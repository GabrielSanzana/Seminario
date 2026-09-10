#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Atencion sin renormalizar, y ponderada por lo que aporta el valor.

DOS ARREGLOS DE LECTURA EN UN MISMO MODULO, PORQUE PIDEN LA MISMA PASADA.

(1) SIN RENORMALIZAR
El modelo se entreno con ATENCION_NORMALIZACION = "sigmoid", que existe
justamente para que una fila pueda sumar menos de 1 cuando el indice decide no
mirar a nadie. Pero la lectura lo deshace en dos sitios: attention_rollout
divide por la suma de fila al final, y la lectura por variante anula la
diagonal y vuelve a dividir. Las matrices guardadas tienen suma de fila 1.0000
con desviacion 5e-08. O sea que el arreglo se aplico al entrenar y se tiro al
medir.

Consecuencia algebraica exacta, no una sospecha: con suma de fila constante y
diagonal excluida, la fila i no ve c_i, y como los efectos de columna suman
cero, la media de fila queda en +c_i/(N-1). Es decir r = c/11 y el efecto
emisor deja de existir como cantidad propia. Medido: corr(r,c) = +1.000 con
razon de desviaciones 0.0909 = 1/11.

Aqui se vuelve a leer de los mismos 50 checkpoints sin ninguna de las dos
divisiones. La suma de fila pasa a ser una observable por derecho propio.

(2) PONDERADA POR EL VALOR
El peso A[i,j] dice cuanto mira i a j, no cuanto RECIBE. Lo que entra de verdad
en el contexto de i es

    c_ij = suma sobre cabezas de  A_h[i,j] * O_h W_V^h x_j

con O_h la franja de out_proj que corresponde a la cabeza h. Un indice puede
tener peso alto y vector de valor casi nulo, y entonces no aporta nada. La
norma ||c_ij|| es la medida de Kobayashi et al. 2020, "Attention is not only a
weight".

Esto no es una idea suelta: es la prediccion que deja el modulo anterior. La
ablacion de aristas mostro que cortar A[v,j] tiene efecto reproducible y de
aridad 2 (81.4%), mientras el peso solo predice ese efecto con rho = +0.369.
Si la explicacion es que lo que importa es A[v,j]*v_j y no A[v,j], entonces
||c_ij|| tiene que predecir Datt bastante mejor que +0.369. Es falsable y se
contrasta aqui.

RESULTADO: LA PREDICCION SALE REFUTADA. Medido sobre 15 semillas, ||c_ij|| da
rho = +0.374 contra Datt y el peso crudo da +0.372: identicos. El motivo es que
las normas de valor son casi uniformes entre indices, de 4.570 a 4.895, un 7%
de rango, asi que ponderar por ellas multiplica la matriz por algo casi
constante. La hipotesis de que la importancia causal vive en A[v,j]*v_j queda
descartada.

Lo que queda en pie es mas fuerte y mas simple: NINGUNA lectura estatica del
bloque de atencion -peso crudo, peso renormalizado, logit, norma de valor,
contribucion de Kobayashi- recupera la estructura causal que si recupera la
intervencion. La sensibilidad del error final a perturbar una arista no es
funcion de la norma de la perturbacion sino de su direccion respecto de lo que
el resto de la red amplifica, o sea de un jacobiano y no de una norma.

HALLAZGO LATERAL, QUE RESULTO SER EL MAS GRANDE
-----------------------------------------------
La aridad de la atencion depende sobre todo de BAJO QUE MASCARA se la lea. El
mismo modelo, las mismas semillas, los mismos parches:

    solo v oculto (lo que lee el pipeline)     fila 48%  col 43%  PAR  9%
    familia de v oculta                        fila 28%  col 19%  PAR 54%

Seis veces mas estructura de par sin tocar el modelo. El motivo es el mismo
atajo de redundancia de siempre: si solo se oculta NDVI, el modelo copia de
EVI2, SAVI o KNDVI, que son casi duplicados, y la fila de atencion sale
generica; si se oculta la familia entera tiene que buscar informacion en otro
sitio y la eleccion pasa a depender del indice. La redundancia no solo estropea
la interpretacion, tapa la estructura DENTRO de la propia matriz.

QUE SE COMPARA
--------------
Cuatro lecturas de la misma pasada, sobre las mismas aristas dirigidas:

    A cruda          el peso, sin renormalizar        (arreglo 1)
    A renormalizada  el peso como lo guarda el pipeline
    norma de valor   ||suma_h O_h W_V^h x_j||, unaria, solo depende de j
    ||c_ij||         la contribucion completa          (arreglo 2)
    A * norma valor  la version barata de lo mismo

y cada una contra: Datt (la ablacion de aristas), la dependencia por ablacion
de entrada, |pc|, y el conjunto invariante del ICP.

USO
    .venv/bin/python atencion_cruda.py --semillas 15
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
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

CAPTURA: List[Dict[str, np.ndarray]] = []
CAPTURAR = False


def _forward_captura(self, src, src_mask=None, src_key_padding_mask=None,
                     is_causal=False, **kwargs):
    """Igual que CustomTransformerEncoderLayer.forward, pero guardando la
    atencion CRUDA y la norma de la contribucion de cada arista."""
    sa = self.self_attn
    B, N, d = src.shape
    nh = sa.num_heads
    hd = d // nh

    q, k, v = F.linear(src, sa.in_proj_weight, sa.in_proj_bias).chunk(3, dim=-1)
    qh = q.reshape(B, N, nh, hd).permute(0, 2, 1, 3)
    kh = k.reshape(B, N, nh, hd).permute(0, 2, 1, 3)
    vh = v.reshape(B, N, nh, hd).permute(0, 2, 1, 3)          # (B,nh,N,hd)

    logits = (qh @ kh.transpose(-2, -1)) / (hd ** 0.5)
    if src_mask is not None:
        logits = logits + src_mask
    attn_weights = H.normalizar_atencion(logits)

    if CAPTURAR:
        with torch.no_grad():
            # u[b,h,j,:] = O_h W_V^h x_j, o sea lo que la cabeza h aporta por
            # el token j UNA VEZ pasado por out_proj. Se hace asi y no con la
            # norma de v_j a secas porque out_proj puede escalar cada cabeza
            # de forma muy distinta.
            Wo = sa.out_proj.weight.detach().reshape(d, nh, hd)
            u = torch.einsum("bhjk,dhk->bhjd", vh.detach(), Wo)  # (B,nh,N,d)
            # ||c_ij|| con c_ij = suma_h A_h[i,j] u[b,h,j]. Se recorre i para
            # no materializar (B,N,N,d) de golpe.
            nor = torch.empty(B, N, N, dtype=torch.float32)
            for i in range(N):
                ci = torch.einsum("bhj,bhjd->bjd",
                                  attn_weights[:, :, i, :].detach(), u)
                nor[:, i, :] = ci.norm(dim=-1).float().cpu()
            CAPTURA.append(dict(
                A=attn_weights.detach().mean(dim=1).float().cpu().numpy(),
                norma=nor.numpy(),
                norma_valor=u.sum(dim=1).norm(dim=-1).float().cpu().numpy()))

    ctx = (attn_weights @ vh).permute(0, 2, 1, 3).reshape(B, N, d)
    src2 = sa.out_proj(ctx)
    src = self.norm1(src + self.dropout1(src2))
    src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
    src = self.norm2(src + self.dropout2(src2))
    return src, attn_weights


def filas_de_variante(M: np.ndarray) -> np.ndarray:
    """(B*N, N, N) -> (N, N) tomando de cada fila de lote su propia variante.

    La fila b del lote de forward_enmascarado es la variante b % N, y de esa
    variante solo se decodifica el canal b % N. La arista que significa algo es
    por tanto [variante, j]: cuanto mira el token OCULTO a cada uno de los
    demas mientras se lo reconstruye. Es la misma convencion que usa la
    ablacion de aristas, para que las dos sean comparables entrada a entrada.
    """
    Bt = M.shape[0]
    var = np.arange(Bt) % N_IDX
    filas = M[np.arange(Bt), var, :]
    return np.stack([filas[var == v].mean(axis=0) for v in range(N_IDX)])


def renormalizar(A: np.ndarray) -> np.ndarray:
    """Lo que hace el pipeline: anular la diagonal y dividir por la suma."""
    B = A.copy()
    np.fill_diagonal(B, 0.0)
    return B / np.maximum(B.sum(axis=1, keepdims=True), 1e-12)


def frames_validacion(n_max: int) -> np.ndarray:
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
    return sel


def _icc(T: np.ndarray) -> float:
    entre = float(T.mean(axis=0).var(ddof=1))
    dentro = float(T.var(axis=0, ddof=1).mean())
    return entre / (entre + dentro) if (entre + dentro) > 0 else 0.0


def dirigidas(M: np.ndarray) -> np.ndarray:
    return np.array([M[i, j] for i in range(N_IDX) for j in range(N_IDX)
                     if i != j])


def main() -> int:
    global CAPTURAR
    ap = argparse.ArgumentParser()
    ap.add_argument("--semillas", type=int, default=15)
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--lote", type=int, default=32)
    ap.add_argument("--mascara", default="familia",
                    choices=("familia", "identidad"),
                    help="condicion de lectura: familia de v oculta (la que "
                         "usa la ablacion) o solo v oculto (la del pipeline)")
    ap.add_argument("--sufijo", default="")
    ap.add_argument("--modelos", default=None,
                    help="carpeta con model_seed_*.pth y val_losses.json; "
                         "por defecto la del pipeline")
    ap.add_argument("--sufijo2", default="")
    ap.add_argument("--datt", default="ablacion_atencion.npy",
                    help="nombre del .npy con la ablacion de aristas del MISMO "
                         "brazo; comparar contra el del otro brazo no diria "
                         "nada")
    ap.add_argument("--seeds-fijas", action="store_true", dest="seeds_fijas",
                    help="usar las primeras semillas de H.SEEDS en vez de las "
                         "mejores por perdida, para comparar dos brazos")
    ap.add_argument("--salida", default="resultados")
    a = ap.parse_args()

    H.CustomTransformerEncoderLayer.forward = _forward_captura
    frames = frames_validacion(a.frames)
    stack_all = np.load(_buscar(CANDIDATOS_STACK, False))
    PC, RR = pc_y_r(stack_all)
    pc_pares = np.array([PC[i, j] for i, j in PARES])

    dir_mod = a.modelos or H.CHECKPOINT_DIR
    dir_val = next(d for d in (a.modelos, os.path.join(dir_mod, "matrices"),
                               H.DIR_MATRICES)
                   if d and os.path.exists(os.path.join(d, "val_losses.json")))
    with open(os.path.join(dir_val, "val_losses.json")) as f:
        vl = json.load(f)
    # Con --modelos se usan las MISMAS semillas que el otro brazo, en el orden
    # de H.SEEDS, y no las mejores por perdida: seleccionar por perdida dentro
    # de cada brazo seria una seleccion distinta en cada uno y ya no se podria
    # atribuir la diferencia al cambio de mascara.
    if a.modelos or a.seeds_fijas:
        semillas = [s for s in list(H.SEEDS) if str(s) in vl][:a.semillas]
    else:
        semillas = [int(s) for s, _ in
                    sorted(vl.items(), key=lambda kv: kv[1])][:a.semillas]
    print("  modelos: " + dir_mod)
    dev = H.device
    img = ((H.TOKEN_PARCHE, H.TOKEN_PARCHE) if H.TOKEN_PARCHE
           else (frames.shape[1], frames.shape[2]))
    ident = torch.eye(N_IDX, dtype=torch.bool, device=dev)
    usar_fam = (a.mascara == "familia"
                and H.MASCARA_MODO == "familia_balanceada")
    base_msk = H._mascara_familias(dev) if usar_fam else ident
    print("  condicion de lectura: "
          + ("familia de v oculta" if usar_fam else "solo v oculto"))

    lecturas = {k: [] for k in ("A cruda", "A renormalizada", "||c_ij||",
                                "A x norma de valor")}
    sumas_fila, normas_valor = [], []
    for s in semillas:
        ck = os.path.join(dir_mod, "model_seed_" + str(s) + ".pth")
        if not os.path.exists(ck):
            continue
        m = H.ConvTransformer(num_indices=H.NUM_INDICES,
                              seq_length=H.SEQ_LENGTH, img_size=img,
                              num_heads=4, num_layers=2)
        m.load_state_dict(torch.load(ck, map_location="cpu"))
        m = m.to(dev).eval()
        acum = {"A": [], "norma": [], "norma_valor": []}
        with torch.no_grad():
            for ini in range(0, len(frames), a.lote):
                fr = torch.from_numpy(np.ascontiguousarray(np.transpose(
                    frames[ini:ini + a.lote], (0, 3, 1, 2)))).float().to(dev)
                mm = base_msk.unsqueeze(0).expand(fr.size(0), N_IDX,
                                                  N_IDX).clone()
                CAPTURA.clear()
                CAPTURAR = True
                _ = m.forward_enmascarado(fr, mm)
                CAPTURAR = False
                # CAPTURA trae una entrada por capa; se promedian las capas,
                # igual que hace por_capa en el pipeline.
                for k in acum:
                    acum[k].append(np.mean([c[k] for c in CAPTURA], axis=0))
                CAPTURA.clear()
        A = filas_de_variante(np.concatenate(acum["A"], axis=0))
        Nc = filas_de_variante(np.concatenate(acum["norma"], axis=0))
        nv = np.concatenate(acum["norma_valor"], axis=0).mean(axis=0)  # (N,)
        lecturas["A cruda"].append(A)
        lecturas["A renormalizada"].append(renormalizar(A))
        lecturas["||c_ij||"].append(Nc)
        lecturas["A x norma de valor"].append(A * nv[None, :])
        sumas_fila.append(A.sum(axis=1))
        normas_valor.append(nv)
        print("  semilla " + str(s) + " lista")
        del m

    M = len(sumas_fila)
    medias = {k: np.mean(v, axis=0) for k, v in lecturas.items()}
    sf = np.mean(sumas_fila, axis=0)
    nv = np.mean(normas_valor, axis=0)

    # Referencias externas.
    Datt = None
    ruta = os.path.join(a.salida, a.datt)
    if os.path.exists(ruta):
        Datt_sem = np.load(ruta)
        Datt = Datt_sem.mean(axis=0)
    fuentes = dict((e, np.mean(q, axis=0)) for e, _n, _A, _s, q
                   in recolectar_fuentes(_buscar(CANDIDATOS_MATRICES, True),
                                         _buscar(CANDIDATOS_PARADIGMAS, True),
                                         PC, RR) if q and len(q) >= 5)
    D_entrada = fuentes.get("dependencia por ablacion (cruda)")
    icp_inv = None
    ruta_icp = os.path.join(a.salida, "ANALISIS_ICP.json")
    if os.path.exists(ruta_icp):
        with open(ruta_icp, encoding="utf-8") as f:
            icp_inv = json.load(f)["aristas"]

    filas = []
    for k, mats in lecturas.items():
        Am = medias[k]
        Tab = np.stack([dirigidas(x) for x in mats])
        f = dict(lectura=k, icc=_icc(Tab), **fracciones_aridad(Am))
        v = dirigidas(Am)
        if Datt is not None:
            f["rho_datt"] = float(stats.spearmanr(v, dirigidas(Datt)).statistic)
            rs = [float(stats.spearmanr(dirigidas(x),
                                        dirigidas(Datt_sem[i])).statistic)
                  for i, x in enumerate(mats) if i < len(Datt_sem)]
            f["rho_datt_por_semilla"] = float(np.mean(rs))
            f["semillas_positivas"] = int(sum(1 for x in rs if x > 0))
            f["n_comparadas"] = len(rs)
        if D_entrada is not None:
            _m, _r, _c, R_A = descomponer_aridad(Am)
            _m2, _r2, _c2, R_D = descomponer_aridad(D_entrada)
            f["rho_ablacion_entrada"] = float(
                stats.spearmanr(dirigidas(Am), dirigidas(D_entrada)).statistic)
            f["rho_par_ablacion"] = float(
                stats.spearmanr(par_simetrico(R_A),
                                par_simetrico(R_D)).statistic)
        f["rho_pc"] = float(stats.spearmanr(fuerza_por_par(Am),
                                            pc_pares).statistic)
        if icp_inv:
            inv = np.array([x["q"] > 0.10 for x in icp_inv])
            vv = np.array([Am[x["i"], x["j"]] for x in icp_inv])
            u = stats.mannwhitneyu(vv[inv], vv[~inv], alternative="two-sided")
            f["auc_icp"] = float(u.statistic / (inv.sum() * (~inv).sum()))
            f["p_icp"] = float(u.pvalue)
        filas.append(f)

    W = 78
    L = ["=" * W, " ATENCION CRUDA Y PONDERADA POR EL VALOR", "",
         " Los 50 modelos se entrenaron con sigmoid, que permite que una fila",
         " sume menos de 1. La lectura del pipeline lo deshace: divide por la",
         " suma de fila en dos sitios y las matrices guardadas tienen suma",
         " 1.0000 con desviacion 5e-08. Aqui se lee sin ninguna division.",
         "",
         " Y se compara el peso contra lo que de verdad entra en el contexto,",
         " ||c_ij|| con c_ij = suma_h A_h[i,j] O_h W_V^h x_j.",
         "=" * W, ""]
    L.append("  semillas: " + str(M) + "    parches: " + str(len(frames))
             + "    condicion: "
             + ("familia de v oculta" if usar_fam else "solo v oculto"))
    L.append("")
    L.append("  SUMA DE FILA DEL PESO CRUDO   (la observable que se perdia)")
    for i in np.argsort(-sf):
        L.append("    " + INDEX_NAMES[i].ljust(14) + ("%.3f" % sf[i]).rjust(8))
    L.append("")
    L.append("  Con sigmoid la fila no tiene por que sumar 1. Alta significa")
    L.append("  que ese indice necesita mirar mucho para reconstruirse.")
    L.append("")
    L.append("  NORMA DEL VALOR POR INDICE   ||suma_h O_h W_V^h x_j||")
    for i in np.argsort(-nv):
        L.append("    " + INDEX_NAMES[i].ljust(14) + ("%.3f" % nv[i]).rjust(8))
    L.append("")
    L.append("=" * W)
    L.append("  LAS CUATRO LECTURAS")
    L.append("")
    L.append("  " + "lectura".ljust(22) + "ICC".rjust(7) + "fila".rjust(8)
             + "col".rjust(8) + "PAR".rjust(8))
    for f in filas:
        L.append("  " + f["lectura"].ljust(22) + ("%.3f" % f["icc"]).rjust(7)
                 + ("%.0f%%" % (100 * f["frac_fila"])).rjust(8)
                 + ("%.0f%%" % (100 * f["frac_col"])).rjust(8)
                 + ("%.0f%%" % (100 * f["frac_par"])).rjust(8))
    L.append("")
    L.append("  La fila 'A renormalizada' es lo que analizaba todo el resto")
    L.append("  del proyecto. La diferencia con 'A cruda' es exactamente lo")
    L.append("  que costaba la division por la suma de fila.")
    L.append("")
    L.append("=" * W)
    L.append("  CONTRA LA ABLACION DE ARISTAS  (la prediccion falsable)")
    L.append("")
    L.append("  Si lo que importa es A[v,j]*v_j y no A[v,j], la norma de la")
    L.append("  contribucion tiene que predecir Datt mejor que el +0.369 que")
    L.append("  daba el peso solo.")
    L.append("")
    L.append("  " + "lectura".ljust(22) + "rho Datt".rjust(10)
             + "por semilla".rjust(13) + "signo +".rjust(9))
    for f in filas:
        if "rho_datt" not in f:
            continue
        L.append("  " + f["lectura"].ljust(22)
                 + ("%+.3f" % f["rho_datt"]).rjust(10)
                 + ("%+.3f" % f["rho_datt_por_semilla"]).rjust(13)
                 + (str(f["semillas_positivas"]) + "/"
                    + str(f["n_comparadas"])).rjust(9))
    L.append("")
    L.append("=" * W)
    L.append("  CONTRA TODO LO DEMAS")
    L.append("")
    L.append("  " + "lectura".ljust(22) + "abl. entrada".rjust(14)
             + "su parte par".rjust(14) + "|pc|".rjust(8) + "auc ICP".rjust(9))
    for f in filas:
        L.append("  " + f["lectura"].ljust(22)
                 + ("%+.3f" % f.get("rho_ablacion_entrada",
                                    float("nan"))).rjust(14)
                 + ("%+.3f" % f.get("rho_par_ablacion",
                                    float("nan"))).rjust(14)
                 + ("%+.3f" % f["rho_pc"]).rjust(8)
                 + ("%.3f" % f.get("auc_icp", float("nan"))).rjust(9))
    L.append("=" * W)
    txt = "\n".join(L)
    print("\n" + txt)
    os.makedirs(a.salida, exist_ok=True)
    with open(os.path.join(a.salida, "ATENCION_CRUDA" + a.sufijo + a.sufijo2 + ".txt"), "w",
              encoding="utf-8") as f:
        f.write(txt)
    with open(os.path.join(a.salida, "ATENCION_CRUDA" + a.sufijo + a.sufijo2 + ".json"), "w",
              encoding="utf-8") as f:
        json.dump(dict(n_semillas=M, suma_fila=sf.tolist(),
                       norma_valor=nv.tolist(), lecturas=filas,
                       matrices={k: v.tolist() for k, v in medias.items()}),
                  f, indent=2, ensure_ascii=False)
    print("\n  Guardado: " + os.path.join(a.salida, "ATENCION_CRUDA" + a.sufijo + a.sufijo2 + ".txt"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
