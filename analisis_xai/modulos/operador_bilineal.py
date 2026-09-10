#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
El operador relacional como PARAMETRO, no como activacion.

DE DONDE SALE
-------------
La atencion que analiza todo el resto del proyecto es una activacion: sale de
correr el modelo sobre datos, promediar sobre pixeles y normalizar por fila. En
ese camino se pierde casi todo lo relacional, y la descomposicion por aridad lo
cuantifico: 74.7% de la varianza del rollout es efecto de nodo.

Pero el logit del que sale esa activacion tiene una forma algebraica exacta.
En este modelo el token del indice i es

    src_i = c_i + p_i        c_i = token_norm(frame_proj(encoder(x_i)))
                             p_i = pos_embed[i], un vector APRENDIDO por indice

y la atencion de la primera capa calcula, por cabeza,

    logit[i,j] = (W_Q src_i + b_Q) . (W_K src_j + b_K) / sqrt(hd)

Sustituyendo src = c + p, eso se abre en nueve terminos exactos:

    p_i^T M p_j     posicion-posicion   ARIDAD 2, y no depende de los datos
    p_i^T M c_j     posicion-contenido  aridad 2
    c_i^T M p_j     contenido-posicion  aridad 2
    c_i^T M c_j     contenido-contenido aridad 2
    (W_Q p_i).b_K   aridad 1, solo fila
    (W_Q c_i).b_K   aridad 1, solo fila
    b_Q.(W_K p_j)   aridad 1, solo columna
    b_Q.(W_K c_j)   aridad 1, solo columna
    b_Q.b_K         aridad 0, constante

con M = W_Q^T W_K. Los cuatro terminos con sesgo son unarios POR ALGEBRA, no
por como salieron los datos: no pueden llevar informacion de par ni aunque el
modelo quisiera. Eso da un reparto por aridad exacto y derivado, no estimado.

LO QUE INTERESA
---------------
El termino posicion-posicion, P[i,j] = p_i^T M p_j, es una matriz 12x12
construida solo con parametros aprendidos: los doce vectores de identidad de
indice y el operador bilineal. No pasa por softmax, no vive en el simplex, no
tiene efecto sumidero posible y no depende de que pixel se le pase. Es el
candidato mas limpio a "relacion aprendida entre indices" que hay en el modelo,
y hasta ahora nadie lo habia leido.

Ademas M se parte en simetrica mas antisimetrica. La parte antisimetrica ES el
operador de direccion, como parametro y no como diferencia de activaciones.

DOS NOCIONES DE ARIDAD, QUE NO HAY QUE CONFUNDIR
------------------------------------------------
El reparto en nueve terminos es ALGEBRAICO: dice de que argumentos depende cada
termino. Los cuatro con sesgo dependen solo de i o solo de j, asi que son
unarios y no pueden llevar informacion de par ni aunque el modelo quisiera.

Eso NO es lo mismo que la descomposicion de dos vias mu + r_i + c_j + R. Un
termino como c_i^T M c_j depende de los dos indices y aun asi puede ser casi
enteramente efecto de nodo, si por ejemplo todos los c_i se parecen. Por eso se
reporta tambien la descomposicion ANOVA del logit total, que es la cifra
comparable con el 24.3% de aridad 2 medido sobre el rollout.

CONTROL DE ENTRENAMIENTO
------------------------
Si el bloque de atencion apenas se movio de su inicializacion, todo lo anterior
describe ruido de partida y no algo aprendido. Se mide la distancia relativa de
cada bloque de parametros a una inicializacion fresca con la misma semilla de
arquitectura, y se compara el bloque de atencion con el encoder, el MLP y el
decoder. Ademas el espectro de M se contrasta contra el de un modelo sin
entrenar.

LIMITE DECLARADO
----------------
La descomposicion en contenido y posicion solo es limpia en la CAPA 1. La
entrada de la capa 2 es la salida de la capa 1, donde contenido y posicion ya
estan mezclados por el bloque de atencion y el MLP, y no hay forma de
separarlos. De la capa 2 se reporta solo el espectro de M.

USO
    .venv/bin/python operador_bilineal.py --semillas 50
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, "/srv/pfigueroa")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hello as H  # noqa: E402
from certificar_todo import (  # noqa: E402
    CANDIDATOS_MATRICES, CANDIDATOS_PARADIGMAS, CANDIDATOS_STACK, INDEX_NAMES,
    N_IDX, PARES, VAL_RATIO, _buscar, fuerza_por_par, pc_y_r,
    recolectar_fuentes,
)
from analisis_atencion import (  # noqa: E402
    OFF, descomponer_aridad, fracciones_aridad, par_simetrico,
)

TERMINOS = ["pos-pos", "pos-cont", "cont-pos", "cont-cont",
            "fila (p.bK)", "fila (c.bK)", "col (bQ.p)", "col (bQ.c)",
            "constante"]


def frames_validacion(n_max: int) -> Tuple[np.ndarray, object]:
    """Parches de validacion, estandarizados igual que en el entrenamiento."""
    stack = np.load(H.OUTPUT_NPY)
    fechas = H.cargar_fechas_stack(len(stack))
    _tr, _va, _img, scaler = H.process_indices_data(
        stack, seq_length=H.SEQ_LENGTH, dates_millis=fechas)
    n_train = int((1.0 - VAL_RATIO) * len(stack))
    sel = stack[n_train:].astype(np.float32)
    plano = sel.reshape(-1, sel.shape[-1])
    sel = scaler.transform(plano).reshape(sel.shape).astype(np.float32)
    if H.TOKEN_PARCHE and H.TOKEN_PARCHE > 0:
        sel = H.trocear_en_parches(sel, H.TOKEN_PARCHE)
    if len(sel) > n_max:
        sel = sel[np.linspace(0, len(sel) - 1, n_max).astype(int)]
    return sel, scaler


def contenido_de(base, frames: np.ndarray, dev) -> torch.Tensor:
    """c_i = token_norm(frame_proj(encoder(x_i))), promediado sobre el lote.

    Se usa el camino SIN enmascarar. El termino que interesa, posicion-posicion,
    no depende de esto en absoluto; los terminos con contenido se reportan como
    magnitud tipica, y decir de que lote salen es parte de declararlo.
    """
    N = base.num_tokens
    fr = torch.from_numpy(np.ascontiguousarray(
        np.transpose(frames, (0, 3, 1, 2)))).float().to(dev)
    B, _, Hh, Ww = fr.shape
    with torch.no_grad():
        e = base.encoder(fr.reshape(B * N, 1, Hh, Ww)).reshape(B, N, -1)
        c = base.token_norm(base.frame_proj(e))          # (B, N, d)
    return c


def nueve_terminos(sa, c: torch.Tensor, p: torch.Tensor, nh: int
                   ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Los nueve terminos del logit de la capa 1, sumados sobre cabezas.

    Devuelve (terminos, logit_reconstruido). El logit reconstruido se compara
    con el que calcula el modelo para verificar que la descomposicion es exacta
    y no una aproximacion.
    """
    d = c.size(-1)
    hd = d // nh
    W = sa.in_proj_weight.detach()
    b = sa.in_proj_bias.detach()
    WQ, WK, _WV = W.chunk(3, dim=0)
    bQ, bK, _bV = b.chunk(3, dim=0)
    esc = 1.0 / (hd ** 0.5)

    cm = c.mean(dim=0).detach()                                   # (N, d) lote promedio
    T = {k: torch.zeros(cm.size(0), cm.size(0), dtype=torch.float64)
         for k in TERMINOS}
    for h in range(nh):
        s = slice(h * hd, (h + 1) * hd)
        Q_c, Q_p = cm @ WQ[s].T, p @ WQ[s].T             # (N, hd)
        K_c, K_p = cm @ WK[s].T, p @ WK[s].T
        bq, bk = bQ[s], bK[s]
        N = cm.size(0)
        uno = torch.ones(N, 1, dtype=Q_c.dtype, device=Q_c.device)
        pares = (("pos-pos", Q_p @ K_p.T),
                 ("pos-cont", Q_p @ K_c.T),
                 ("cont-pos", Q_c @ K_p.T),
                 ("cont-cont", Q_c @ K_c.T),
                 ("fila (p.bK)", (Q_p @ bk).unsqueeze(1) @ uno.T),
                 ("fila (c.bK)", (Q_c @ bk).unsqueeze(1) @ uno.T),
                 ("col (bQ.p)", uno @ (K_p @ bq).unsqueeze(0)),
                 ("col (bQ.c)", uno @ (K_c @ bq).unsqueeze(0)),
                 ("constante", torch.full((N, N), float(bq @ bk),
                                          dtype=Q_c.dtype)))
        for k, M in pares:
            T[k] += (esc * M).double().cpu()
    total = sum(T.values())
    return ({k: v.numpy() for k, v in T.items()}, total.numpy())


def logit_real(sa, src: torch.Tensor, nh: int) -> np.ndarray:
    """El logit tal como lo calcula CustomTransformerEncoderLayer, para el
    control de exactitud."""
    d = src.size(-1)
    hd = d // nh
    q, k, _v = torch.nn.functional.linear(
        src, sa.in_proj_weight.detach(),
        sa.in_proj_bias.detach()).chunk(3, dim=-1)
    N = src.size(-2)
    qh = q.reshape(-1, N, nh, hd).permute(0, 2, 1, 3)
    kh = k.reshape(-1, N, nh, hd).permute(0, 2, 1, 3)
    lg = (qh @ kh.transpose(-2, -1)) / (hd ** 0.5)
    return lg.sum(dim=1).mean(dim=0).double().cpu().numpy()


def energia_simetrica(A: np.ndarray) -> Dict:
    """Reparto de energia entre parte simetrica y antisimetrica.

    Fuera de la diagonal, que en la parte antisimetrica es cero por definicion
    e inflaria la fraccion simetrica. Vale para cualquier tamano: se usa tanto
    sobre la matriz 12x12 de indices como sobre M de 128x128.
    """
    S = 0.5 * (A + A.T)
    K = 0.5 * (A - A.T)
    fuera = ~np.eye(A.shape[0], dtype=bool)
    es = float((S[fuera] ** 2).sum())
    ea = float((K[fuera] ** 2).sum())
    tot = es + ea
    return dict(frac_simetrica=es / tot if tot > 0 else float("nan"),
                frac_antisimetrica=ea / tot if tot > 0 else float("nan"))


def _icc(T: np.ndarray) -> float:
    entre = float(T.mean(axis=0).var(ddof=1))
    dentro = float(T.var(axis=0, ddof=1).mean())
    return entre / (entre + dentro) if (entre + dentro) > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--semillas", type=int, default=50)
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--salida", default="resultados")
    a = ap.parse_args()

    frames, _sc = frames_validacion(a.frames)
    print("  parches de validacion: " + str(len(frames)))
    stack_all = np.load(_buscar(CANDIDATOS_STACK, False))
    PC, RR = pc_y_r(stack_all)
    pc_pares = np.array([PC[i, j] for i, j in PARES])
    rr_pares = np.array([RR[i, j] for i, j in PARES])

    with open(os.path.join(H.DIR_MATRICES, "val_losses.json")) as f:
        vl = json.load(f)
    semillas = [int(s) for s, _ in
                sorted(vl.items(), key=lambda kv: kv[1])][:a.semillas]
    dev = H.device
    img = ((H.TOKEN_PARCHE, H.TOKEN_PARCHE) if H.TOKEN_PARCHE
           else (frames.shape[1], frames.shape[2]))

    Ps, terminos_ss, control, logits_tot = [], [], [], []
    dist_bloques, esp_nulo = [], []
    espectros = {0: [], 1: []}
    for s in semillas:
        ck = os.path.join(H.CHECKPOINT_DIR, "model_seed_" + str(s) + ".pth")
        if not os.path.exists(ck):
            continue
        m = H.ConvTransformer(num_indices=H.NUM_INDICES,
                              seq_length=H.SEQ_LENGTH, img_size=img,
                              num_heads=4, num_layers=2)
        m.load_state_dict(torch.load(ck, map_location="cpu"))
        m = m.to(dev).eval()
        c = contenido_de(m, frames, dev)
        p = m.pos_embed[0].detach()
        sa = m.transformer.layers[0].self_attn
        nh = sa.num_heads
        T, total = nueve_terminos(sa, c, p, nh)
        # Control de exactitud: la suma de los nueve terminos tiene que dar el
        # logit que calcula el modelo, salvo error de redondeo.
        src = c.mean(dim=0, keepdim=True) + p.unsqueeze(0)
        with torch.no_grad():
            lr = logit_real(sa, src, nh)
        control.append(float(np.abs(total - lr).max()))
        Ps.append(T["pos-pos"])
        logits_tot.append(total)
        ss = {k: float(((v[OFF] - v[OFF].mean()) ** 2).sum())
              for k, v in T.items()}
        tot = sum(ss.values())
        terminos_ss.append({k: (v / tot if tot > 0 else 0.0)
                            for k, v in ss.items()})
        # Control de entrenamiento: cuanto se movio cada bloque respecto de
        # una inicializacion fresca de la misma arquitectura.
        fresco = H.ConvTransformer(num_indices=H.NUM_INDICES,
                                   seq_length=H.SEQ_LENGTH, img_size=img,
                                   num_heads=4, num_layers=2)
        sd_e, sd_f = m.state_dict(), fresco.state_dict()
        grupos = {"atencion (in_proj)": "self_attn.in_proj_weight",
                  "atencion (out_proj)": "self_attn.out_proj.weight",
                  "MLP del bloque": "linear",
                  "encoder conv": "encoder.",
                  "frame_proj": "frame_proj.weight",
                  "pos_embed": "pos_embed",
                  "decoder": "decoder."}
        d1 = {}
        for nom, clave in grupos.items():
            num = den = 0.0
            for k in sd_e:
                if clave in k and sd_e[k].dtype.is_floating_point:
                    a1 = sd_e[k].detach().cpu().float()
                    a0 = sd_f[k].detach().cpu().float()
                    num += float(((a1 - a0) ** 2).sum())
                    den += float((a0 ** 2).sum())
            d1[nom] = (num / den) ** 0.5 if den > 0 else float("nan")
        dist_bloques.append(d1)
        for L in (0, 1):
            saF = fresco.transformer.layers[L].self_attn
            WQf, WKf, _ = saF.in_proj_weight.chunk(3, dim=0)
            Mf = (WQf.T @ WKf).detach().cpu().numpy()
            evf = np.linalg.svd(Mf, compute_uv=False)
            esp_nulo.append(dict(
                capa=L,
                rango_efectivo=float(np.exp(stats.entropy(evf / evf.sum()))),
                **energia_simetrica(Mf)))
        del fresco
        for L in (0, 1):
            saL = m.transformer.layers[L].self_attn
            WQ, WK, _ = saL.in_proj_weight.chunk(3, dim=0)
            M = (WQ.T @ WK).detach().cpu().numpy()
            ev = np.linalg.svd(M, compute_uv=False)
            espectros[L].append(dict(
                rango_efectivo=float(np.exp(stats.entropy(ev / ev.sum()))),
                **energia_simetrica(M)))
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    M_sem = len(Ps)
    print("  semillas leidas: " + str(M_sem)
          + "   control de exactitud (max abs): "
          + ("%.2e" % max(control)))

    P_med = np.mean(Ps, axis=0)
    logit_med = np.mean(logits_tot, axis=0)
    anova_logit = fracciones_aridad(logit_med)
    anova_por_semilla = {k: float(np.mean([fracciones_aridad(x)[k]
                                           for x in logits_tot]))
                         for k in ("frac_fila", "frac_col", "frac_par")}
    dist = {k: float(np.mean([d[k] for d in dist_bloques]))
            for k in dist_bloques[0]}
    nulo_esp = {L: dict(
        rango_efectivo=float(np.mean([e["rango_efectivo"] for e in esp_nulo
                                      if e["capa"] == L])),
        frac_antisimetrica=float(np.mean([e["frac_antisimetrica"]
                                          for e in esp_nulo
                                          if e["capa"] == L])))
        for L in (0, 1)}
    tabla_ss = {k: float(np.mean([t[k] for t in terminos_ss]))
                for k in TERMINOS}

    # Reproducibilidad de P entre semillas.
    Tab = np.stack([p[OFF] for p in Ps])
    rho_sem = [stats.spearmanr(Tab[i], Tab[j]).statistic
               for i in range(M_sem) for j in range(i + 1, M_sem)]
    res_P = dict(icc=_icc(Tab), spearman_medio=float(np.nanmean(rho_sem)),
                 **fracciones_aridad(P_med), **energia_simetrica(P_med))

    # Contra la verdad y contra el otro operador.
    _mu, _r, _c, R_P = descomponer_aridad(P_med)
    par_P = par_simetrico(R_P)
    fuerte_P = np.array([max(abs(P_med[i, j]), abs(P_med[j, i]))
                         for i, j in PARES])
    fuentes = [(e, q) for e, _n, _A, _s, q in recolectar_fuentes(
        _buscar(CANDIDATOS_MATRICES, True),
        _buscar(CANDIDATOS_PARADIGMAS, True), PC, RR) if q and len(q) >= 5]
    comp = {}
    for et, mats in fuentes:
        A = np.mean(mats, axis=0)
        _m2, _r2, _c2, R_A = descomponer_aridad(A)
        comp[et] = dict(
            rho_crudo=float(stats.spearmanr(fuerte_P,
                                            fuerza_por_par(A)).statistic),
            rho_par=float(stats.spearmanr(par_P,
                                          par_simetrico(R_A)).statistic))
    verdad = dict(
        rho_pc_crudo=float(stats.spearmanr(fuerte_P, pc_pares).statistic),
        rho_pc_par=float(stats.spearmanr(par_P, pc_pares).statistic),
        rho_r_crudo=float(stats.spearmanr(fuerte_P, rr_pares).statistic))

    # Contra el conjunto invariante del ICP, si esta calculado.
    icp = {}
    ruta_icp = os.path.join(a.salida, "ANALISIS_ICP.json")
    if os.path.exists(ruta_icp):
        with open(ruta_icp, encoding="utf-8") as f:
            aristas = json.load(f)["aristas"]
        inv = np.array([x["q"] > 0.10 for x in aristas])
        v = np.array([abs(P_med[x["i"], x["j"]]) for x in aristas])
        beta = np.array([abs(x["beta_medio"]) for x in aristas])
        u = stats.mannwhitneyu(v[inv], v[~inv], alternative="two-sided")
        icp = dict(auc=float(u.statistic / (inv.sum() * (~inv).sum())),
                   p=float(u.pvalue),
                   rho_beta=float(stats.spearmanr(v, beta).statistic))

    W = 78
    L = ["=" * W, " EL OPERADOR RELACIONAL COMO PARAMETRO", "",
         " logit[i,j] de la capa 1, abierto en nueve terminos exactos con",
         " src = contenido + posicion y M = W_Q^T W_K. Los cuatro terminos con",
         " sesgo son unarios POR ALGEBRA: no pueden llevar informacion de par.",
         "",
         " Control de exactitud: la suma de los nueve reconstruye el logit del",
         " modelo con error maximo " + ("%.2e" % max(control)) + ".",
         "=" * W, ""]
    L.append("  DE DONDE VIENE LA VARIACION DEL LOGIT  (media de "
             + str(M_sem) + " semillas)")
    L.append("  " + "termino".ljust(18) + "aridad".ljust(10)
             + "fraccion de SS".rjust(16))
    ar = {"pos-pos": "2", "pos-cont": "2", "cont-pos": "2", "cont-cont": "2",
          "fila (p.bK)": "1", "fila (c.bK)": "1", "col (bQ.p)": "1",
          "col (bQ.c)": "1", "constante": "0"}
    for k in TERMINOS:
        L.append("  " + k.ljust(18) + ar[k].ljust(10)
                 + ("%.1f%%" % (100 * tabla_ss[k])).rjust(16))
    u2 = sum(tabla_ss[k] for k in TERMINOS if ar[k] == "2")
    u1 = sum(tabla_ss[k] for k in TERMINOS if ar[k] == "1")
    L.append("  " + "TOTAL aridad 2".ljust(28) + ("%.1f%%" % (100 * u2)).rjust(16))
    L.append("  " + "TOTAL aridad 1".ljust(28) + ("%.1f%%" % (100 * u1)).rjust(16))
    L.append("")
    L.append("  Ese reparto es ALGEBRAICO: dice de que argumentos depende cada")
    L.append("  termino. Un termino de aridad 2 puede seguir siendo casi todo")
    L.append("  efecto de nodo, asi que la cifra comparable con el rollout es")
    L.append("  la descomposicion de dos vias del logit COMPLETO:")
    L.append("")
    L.append("  " + "objeto".ljust(38) + "fila".rjust(9) + "columna".rjust(9)
             + "PAR".rjust(9))
    L.append("  " + "logit de la capa 1 (antes de normalizar)".ljust(38)
             + ("%.1f%%" % (100 * anova_logit["frac_fila"])).rjust(9)
             + ("%.1f%%" % (100 * anova_logit["frac_col"])).rjust(9)
             + ("%.1f%%" % (100 * anova_logit["frac_par"])).rjust(9))
    L.append("  " + "  el mismo, promediado por semilla".ljust(38)
             + ("%.1f%%" % (100 * anova_por_semilla["frac_fila"])).rjust(9)
             + ("%.1f%%" % (100 * anova_por_semilla["frac_col"])).rjust(9)
             + ("%.1f%%" % (100 * anova_por_semilla["frac_par"])).rjust(9))
    L.append("  " + "rollout, ya normalizado (medido antes)".ljust(38)
             + "1.0%".rjust(9) + "74.7%".rjust(9) + "24.3%".rjust(9))
    L.append("")
    L.append("  La diferencia entre esas dos filas es lo que la normalizacion")
    L.append("  y la renormalizacion por fila le hacen a la matriz.")
    L.append("")
    L.append("=" * W)
    L.append("  P[i,j] = p_i^T M p_j   (posicion-posicion, solo parametros)")
    L.append("")
    L.append("  ICC entre semillas                  "
             + ("%.3f" % res_P["icc"]).rjust(8))
    L.append("  Spearman medio entre semillas       "
             + ("%+.3f" % res_P["spearman_medio"]).rjust(8))
    L.append("  fraccion de aridad 2                "
             + ("%.1f%%" % (100 * res_P["frac_par"])).rjust(8))
    L.append("  fraccion simetrica                  "
             + ("%.1f%%" % (100 * res_P["frac_simetrica"])).rjust(8))
    L.append("  fraccion antisimetrica (DIRECCION)  "
             + ("%.1f%%" % (100 * res_P["frac_antisimetrica"])).rjust(8))
    L.append("")
    L.append("  Contra la verdad")
    L.append("    Spearman con |pc|, matriz cruda   "
             + ("%+.3f" % verdad["rho_pc_crudo"]).rjust(8))
    L.append("    Spearman con |pc|, parte de par   "
             + ("%+.3f" % verdad["rho_pc_par"]).rjust(8))
    L.append("    Spearman con |r| (liston)         "
             + ("%+.3f" % verdad["rho_r_crudo"]).rjust(8))
    if icp:
        L.append("    auc contra el invariante del ICP  "
                 + ("%.3f" % icp["auc"]).rjust(8)
                 + "   p " + ("%.4f" % icp["p"]))
        L.append("    Spearman con |beta| del ICP       "
                 + ("%+.3f" % icp["rho_beta"]).rjust(8))
    L.append("")
    L.append("  Contra las fuentes que ya se analizaban")
    L.append("  " + "fuente".ljust(32) + "rho crudo".rjust(11)
             + "rho par".rjust(10))
    for et, d in comp.items():
        L.append("  " + et[:31].ljust(32)
                 + ("%+.3f" % d["rho_crudo"]).rjust(11)
                 + ("%+.3f" % d["rho_par"]).rjust(10))
    L.append("")
    L.append("  Los doce indices ordenados por cuanto RECIBEN en P")
    cP = np.array([P_med[[k for k in range(N_IDX) if k != j], j].mean()
                   for j in range(N_IDX)])
    orden = np.argsort(-cP)
    L.append("    " + "  ".join(INDEX_NAMES[i] + " " + ("%+.3f" % cP[i])
                                for i in orden[:6]))
    L.append("    " + "  ".join(INDEX_NAMES[i] + " " + ("%+.3f" % cP[i])
                                for i in orden[6:]))
    L.append("")
    L.append("=" * W)
    L.append("  CONTROL: SE ENTRENO EL BLOQUE DE ATENCION?")
    L.append("")
    L.append("  Primer intento, que NO SIRVE y se deja escrito por eso:")
    L.append("  distancia relativa a una inicializacion fresca, por bloque.")
    L.append("")
    for k, v in sorted(dist.items(), key=lambda kv: -kv[1]):
        L.append("    " + k.ljust(24) + ("%.3f" % v).rjust(8))
    L.append("")
    L.append("  Casi todos los bloques salen en 1.42, que es raiz de 2. Ese es")
    L.append("  exactamente el valor que dan dos vectores aleatorios")
    L.append("  independientes de norma parecida, y sale igual este o no")
    L.append("  entrenado el bloque: la inicializacion fresca usa otra semilla,")
    L.append("  asi que la comparacion mide que son dos sorteos distintos y")
    L.append("  nada mas. Haria falta guardar el estado inicial de cada")
    L.append("  entrenamiento, que no se hizo. El control valido es el de")
    L.append("  abajo, sobre el espectro.")
    L.append("")
    L.append("=" * W)
    L.append("  ESPECTRO DE M = W_Q^T W_K, POR CAPA")
    L.append("")
    L.append("  " + "capa".ljust(8) + "rango efectivo".rjust(16)
             + "antisimetrica".rjust(15) + "rango SIN entrenar".rjust(20)
             + "antisim. SIN entr.".rjust(20))
    for Lc in (0, 1):
        e = espectros[Lc]
        L.append("  " + ("A" + str(Lc + 1)).ljust(8)
                 + ("%.1f" % np.mean([x["rango_efectivo"] for x in e])).rjust(16)
                 + ("%.1f%%" % (100 * np.mean([x["frac_antisimetrica"]
                                               for x in e]))).rjust(15)
                 + ("%.1f" % nulo_esp[Lc]["rango_efectivo"]).rjust(20)
                 + ("%.1f%%" % (100 * nulo_esp[Lc]
                                ["frac_antisimetrica"])).rjust(20))
    L.append("")
    L.append("  Las dos ultimas columnas son el mismo estadistico sobre un")
    L.append("  modelo recien inicializado. Si coinciden con las dos primeras,")
    L.append("  M no tiene estructura que la inicializacion no tuviera ya.")
    L.append("")
    L.append("  El operador de la capa 2 se lista solo por su espectro: su")
    L.append("  entrada es la salida de la capa 1, donde contenido y posicion")
    L.append("  ya estan mezclados y no se pueden separar.")
    L.append("=" * W)
    txt = "\n".join(L)
    print("\n" + txt)
    os.makedirs(a.salida, exist_ok=True)
    with open(os.path.join(a.salida, "OPERADOR_BILINEAL.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt)
    with open(os.path.join(a.salida, "OPERADOR_BILINEAL.json"), "w",
              encoding="utf-8") as f:
        json.dump(dict(n_semillas=M_sem, control_max=max(control),
                       terminos=tabla_ss, anova_logit=anova_logit,
                       anova_logit_por_semilla=anova_por_semilla,
                       distancia_a_init=dist,
                       espectro_sin_entrenar={str(k): v
                                              for k, v in nulo_esp.items()},
                       pos_pos=res_P, verdad=verdad,
                       icp=icp, comparacion=comp,
                       P=P_med.tolist(),
                       espectros={str(k): v for k, v in espectros.items()}),
                  f, indent=2, ensure_ascii=False)
    np.save(os.path.join(a.salida, "operador_pos_pos.npy"),
            np.stack(Ps).astype(np.float32))
    print("\n  Guardado: "
          + os.path.join(a.salida, "OPERADOR_BILINEAL.txt"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
