#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interferir la atencion: cortar una arista y medir cuanto empeora el modelo.

POR QUE HACE FALTA
------------------
La atencion es un objeto OBSERVACIONAL: se mira lo que el modelo hizo. La
dependencia por ablacion es INTERVENCIONAL: se cambia la entrada y se mide el
efecto. Esa diferencia de estatus epistemico, y no una diferencia de calidad,
es la explicacion mas simple de por que la segunda muestra senal de par (84.6%
de su varianza) y la primera no (24.3%, y de esa casi nada reproducible como
relacion).

Este modulo pone a la atencion en el mismo estatus. En vez de leer A[v,j], se
ANULA y se mide cuanto sube el error de reconstruir v:

    Datt[v,j] = MSE(reconstruir v | A[v,j] = 0) - MSE(reconstruir v)

Es la metrica de fidelidad que el informe teorico pide en su Etapa 4, pero por
arista y sin sustituir la matriz entera.

QUE PREGUNTA CONTESTA QUE NINGUNA OTRA CONTESTA
-----------------------------------------------
Si Datt[v,j] es cero, ese peso de atencion NO SE USA: el modelo reconstruye v
igual de bien sin el. Eso es prueba directa de que el peso es decorativo, en vez
de inferirlo de que la matriz no correlaciona con nada. Y si la correlacion
entre A[v,j] y Datt[v,j] es cero, entonces el TAMANO del peso no dice nada
sobre su importancia, que es la suposicion implicita de todo el framework
cuando discretiza con Top-P.

COMO SE HACE EXACTO Y BARATO
----------------------------
forward_enmascarado arma B*N secuencias: la fila b*N+v del lote es la variante
v, donde el token v esta oculto, y de esa variante SOLO se decodifica el canal
v. Asi que anular A[v,j] unicamente en las filas de lote cuya variante es v
aisla el efecto por completo: ninguna otra salida lo ve.

Eso permite cortar las doce aristas (v,j) para v = 0..11 en UNA sola pasada,
cada una en su propia variante. Doce pasadas mas la base, el mismo coste que
matriz_dependencia_ablacion, y no 132.

El corte NO renormaliza la fila. Con ATENCION_NORMALIZACION = "sigmoid" las
filas no tienen por que sumar 1, asi que quitar una entrada es exactamente
"quitar ese canal de informacion". Renormalizar repartiria la masa entre los
demas y mezclaria "quitar j" con "subir a los otros", que es otra intervencion
distinta.

REFERENCIA
----------
Se usa la misma mascara base que matriz_dependencia_ablacion, para que Datt y D
esten en las mismas unidades y la misma condicion y se puedan comparar arista a
arista. Con familia_balanceada eso significa condicionar a que la familia de v
este oculta.

USO
    .venv/bin/python ablacion_atencion.py --semillas 15
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

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

# Columna a cortar, o None. Se lee desde el forward parcheado. Es global de
# modulo y no atributo del modelo porque la capa no tiene referencia al modelo
# que la contiene.
COLUMNA_CORTADA: Optional[int] = None


def _recortar_sin_renormalizar(self, attn: torch.Tensor) -> np.ndarray:
    """_recortar_atencion sin la division por la suma de fila.

    El original renormaliza para que Top-P tenga filas comparables. Aqui hace
    dano: con ATENCION_NORMALIZACION = "sigmoid" la suma de fila es informacion
    real -cuanto necesito mirar ese indice- y renormalizar la borra, ademas de
    hacer incomparables las magnitudes entre filas, que es justo lo que hay que
    comparar contra el efecto de cortar cada arista.
    """
    a = attn.detach()
    N = self.num_tokens
    if self.n_registros > 0:
        self._masa_registros.append(
            float(a[..., :N, N:].sum(dim=-1).mean().cpu()))
        a = a[..., :N, :N]
    return a.cpu().numpy()


def _forward_con_corte(self, src, src_mask=None, src_key_padding_mask=None,
                       is_causal=False, **kwargs):
    """CustomTransformerEncoderLayer.forward, con el corte de arista.

    Copia fiel del original salvo las cuatro lineas del corte. Se parchea la
    clase en tiempo de ejecucion en vez de tocar hello.py, que es compartido.
    """
    sa = self.self_attn
    B, N, d = src.shape
    nh = sa.num_heads
    hd = d // nh

    q, k, v = F.linear(src, sa.in_proj_weight, sa.in_proj_bias).chunk(3, dim=-1)
    qh = q.reshape(B, N, nh, hd).permute(0, 2, 1, 3)
    kh = k.reshape(B, N, nh, hd).permute(0, 2, 1, 3)
    vh = v.reshape(B, N, nh, hd).permute(0, 2, 1, 3)

    logits = (qh @ kh.transpose(-2, -1)) / (hd ** 0.5)
    if src_mask is not None:
        logits = logits + src_mask
    attn_weights = H.normalizar_atencion(logits)          # (B, nh, N, N)

    if COLUMNA_CORTADA is not None:
        # La fila b del lote es la variante b % N, y de esa variante solo se
        # decodifica el canal b % N. Anular [variante, j] en su propia fila
        # corta la arista v->j sin tocar ninguna otra salida.
        attn_weights = attn_weights.clone()
        filas = torch.arange(B, device=attn_weights.device)
        var = filas % N
        attn_weights[filas, :, var, COLUMNA_CORTADA] = 0.0

    ctx = (attn_weights @ vh).permute(0, 2, 1, 3).reshape(B, N, d)
    src2 = sa.out_proj(ctx)

    src = self.norm1(src + self.dropout1(src2))
    src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
    src = self.norm2(src + self.dropout2(src2))
    return src, attn_weights


def error_por_canal(base, frames: np.ndarray, msk, lote: int) -> np.ndarray:
    """MSE de reconstruir cada canal, con la mascara base dada."""
    N = base.num_tokens
    dev = next(base.parameters()).device
    suma = np.zeros(N, dtype=np.float64)
    n_pix = 0
    with torch.no_grad():
        for ini in range(0, len(frames), lote):
            fr = torch.from_numpy(np.ascontiguousarray(np.transpose(
                frames[ini:ini + lote], (0, 3, 1, 2)))).float().to(dev)
            B = fr.size(0)
            m = msk.unsqueeze(0).expand(B, N, N).clone()
            out = base.forward_enmascarado(fr, m)
            err = ((out - fr) ** 2).sum(dim=(0, 2, 3))
            suma += err.detach().cpu().numpy().astype(np.float64)
            n_pix += B * fr.shape[-1] * fr.shape[-2]
    return suma / max(n_pix, 1)


def matriz_ablacion_atencion(base, frames: np.ndarray, msk, lote: int
                             ) -> np.ndarray:
    """Datt[v,j] = cuanto sube el MSE de v al cortar la arista de atencion v->j.
    """
    global COLUMNA_CORTADA
    N = base.num_tokens
    COLUMNA_CORTADA = None
    e0 = error_por_canal(base, frames, msk, lote)
    Datt = np.zeros((N, N), dtype=np.float64)
    for j in range(N):
        COLUMNA_CORTADA = j
        ej = error_por_canal(base, frames, msk, lote)
        Datt[:, j] = ej - e0
    COLUMNA_CORTADA = None
    np.fill_diagonal(Datt, 0.0)
    return Datt


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--semillas", type=int, default=15)
    ap.add_argument("--frames", type=int, default=128)
    ap.add_argument("--lote", type=int, default=32)
    ap.add_argument("--modelos", default=None,
                    help="carpeta con model_seed_*.pth y val_losses.json; "
                         "por defecto la del pipeline")
    ap.add_argument("--sufijo2", default="")
    ap.add_argument("--mascara", default="familia",
                    choices=("familia", "identidad"))
    ap.add_argument("--seeds-fijas", action="store_true", dest="seeds_fijas",
                    help="usar las primeras semillas de H.SEEDS en vez de las "
                         "mejores por perdida, para comparar dos brazos")
    ap.add_argument("--salida", default="resultados")
    a = ap.parse_args()

    H.CustomTransformerEncoderLayer.forward = _forward_con_corte
    H.ConvTransformer._recortar_atencion = _recortar_sin_renormalizar
    print("  forward parcheado con el corte de arista")
    print("  lectura de atencion SIN renormalizar por fila")

    frames = frames_validacion(a.frames)
    print("  parches de validacion: " + str(len(frames)))
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
    print("  referencia: "
          + ("familia de v oculta" if usar_fam else "solo v oculto"))

    Datts, Aatts = [], []
    for s in semillas:
        ck = os.path.join(dir_mod, "model_seed_" + str(s) + ".pth")
        if not os.path.exists(ck):
            continue
        m = H.ConvTransformer(num_indices=H.NUM_INDICES,
                              seq_length=H.SEQ_LENGTH, img_size=img,
                              num_heads=4, num_layers=2)
        m.load_state_dict(torch.load(ck, map_location="cpu"))
        m = m.to(dev).eval()
        Datts.append(matriz_ablacion_atencion(m, frames, base_msk, a.lote))
        # El peso de atencion de la MISMA condicion, para poder preguntar si el
        # tamano del peso predice su efecto.
        with torch.no_grad():
            fr = torch.from_numpy(np.ascontiguousarray(np.transpose(
                frames[:a.lote], (0, 3, 1, 2)))).float().to(dev)
            mm = base_msk.unsqueeze(0).expand(fr.size(0), N_IDX,
                                              N_IDX).clone()
            m.capture_attention = True
            _ = m.forward_enmascarado(fr, mm)
            cache = m.get_attention_weights()
            m.capture_attention = False
        capas = cache[0] if cache else None
        if capas is not None:
            # A[v,j] con v la variante: cada capa llega como (B*N, nh, N, N) y
            # de cada fila de lote interesa la fila v del token oculto, que es
            # la que el corte interviene.
            Bt = capas[0].shape[0]
            var = np.arange(Bt) % N_IDX
            porcapa = []
            for c in capas:
                pl = c.mean(axis=1)[np.arange(Bt), var, :]     # (B*N, N)
                porcapa.append(np.stack([pl[var == v].mean(axis=0)
                                         for v in range(N_IDX)]))
            Aatts.append(np.mean(porcapa, axis=0))
        print("  semilla " + str(s) + " lista")
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    M = len(Datts)
    Dm = np.mean(Datts, axis=0)
    Am = np.mean(Aatts, axis=0) if Aatts else None

    # Nivel de ruido: la cola negativa. Cortar una arista no puede MEJORAR la
    # reconstruccion salvo por ruido de estimacion, asi que la magnitud tipica
    # de los valores negativos mide el suelo.
    neg = Dm[Dm < 0]
    ruido = float(np.abs(neg).mean()) if neg.size else 0.0
    n_efecto = int((Dm > 3 * ruido).sum())

    # Reproducibilidad y aridad.
    Tab = np.stack([d[OFF] for d in Datts])
    entre = float(Tab.mean(axis=0).var(ddof=1))
    dentro = float(Tab.var(axis=0, ddof=1).mean())
    icc = entre / (entre + dentro) if (entre + dentro) > 0 else 0.0
    ar = fracciones_aridad(Dm)

    # Comparaciones.
    _mu, _r, _c, R_D = descomponer_aridad(Dm)
    par_D = par_simetrico(R_D)
    fuerte_D = fuerza_por_par(Dm)
    comp = {}
    fuentes = [(e, q) for e, _n, _A, _s, q in recolectar_fuentes(
        _buscar(CANDIDATOS_MATRICES, True),
        _buscar(CANDIDATOS_PARADIGMAS, True), PC, RR) if q and len(q) >= 5]
    for et, mats in fuentes:
        A = np.mean(mats, axis=0)
        _m2, _r2, _c2, R_A = descomponer_aridad(A)
        comp[et] = dict(
            rho_crudo=float(stats.spearmanr(fuerte_D,
                                            fuerza_por_par(A)).statistic),
            rho_par=float(stats.spearmanr(par_D,
                                          par_simetrico(R_A)).statistic))

    # La prueba central: el peso predice su propio efecto?
    peso_vs_efecto = {}
    if Am is not None:
        va = np.array([Am[i, j] for i, j in
                       [(x, y) for x in range(N_IDX) for y in range(N_IDX)
                        if x != y]])
        vd = np.array([Dm[i, j] for i, j in
                       [(x, y) for x in range(N_IDX) for y in range(N_IDX)
                        if x != y]])
        r = stats.spearmanr(va, vd)
        peso_vs_efecto = dict(rho=float(r.statistic), p=float(r.pvalue))
        # Y por semilla, que es mas exigente que sobre el promedio.
        rs = []
        for Dd, Aa in zip(Datts, Aatts):
            x = np.array([Aa[i, j] for i in range(N_IDX)
                          for j in range(N_IDX) if i != j])
            y = np.array([Dd[i, j] for i in range(N_IDX)
                          for j in range(N_IDX) if i != j])
            rs.append(float(stats.spearmanr(x, y).statistic))
        peso_vs_efecto["rho_medio_por_semilla"] = float(np.mean(rs))
        peso_vs_efecto["semillas_positivas"] = int(sum(1 for x in rs if x > 0))
        peso_vs_efecto["n_semillas"] = len(rs)

    verdad = dict(rho_pc=float(stats.spearmanr(fuerte_D, pc_pares).statistic),
                  rho_pc_par=float(stats.spearmanr(par_D, pc_pares).statistic))
    icp = {}
    ruta = os.path.join(a.salida, "ANALISIS_ICP.json")
    if os.path.exists(ruta):
        with open(ruta, encoding="utf-8") as f:
            aristas = json.load(f)["aristas"]
        inv = np.array([x["q"] > 0.10 for x in aristas])
        v = np.array([Dm[x["i"], x["j"]] for x in aristas])
        u = stats.mannwhitneyu(v[inv], v[~inv], alternative="two-sided")
        icp = dict(auc=float(u.statistic / (inv.sum() * (~inv).sum())),
                   p=float(u.pvalue))

    orden = sorted(((Dm[i, j], i, j) for i in range(N_IDX)
                    for j in range(N_IDX) if i != j), reverse=True)

    W = 78
    L = ["=" * W, " ABLACION DE ARISTAS DE ATENCION", "",
         " Datt[v,j] = MSE(reconstruir v | A[v,j]=0) - MSE(reconstruir v).",
         " El corte se aplica solo en la variante v, que es la unica que",
         " produce el canal v, asi que el efecto queda aislado.",
         "",
         " Convierte la atencion de objeto observacional en intervencional, el",
         " mismo estatus que la dependencia por ablacion.",
         "=" * W, ""]
    L.append("  semillas: " + str(M) + "    parches: " + str(len(frames)))
    L.append("  nivel de ruido (media de la cola negativa): "
             + ("%.3e" % ruido))
    L.append("  aristas con efecto por encima de 3 veces el ruido: "
             + str(n_efecto) + " de 132")
    L.append("")
    if Am is not None:
        sf = Am.sum(axis=1)
        L.append("  Suma de fila del peso crudo (sin renormalizar): "
                 + " ".join("%.2f" % x for x in sf))
        L.append("  Con sigmoid la fila no tiene por que sumar 1. Una fila baja")
        L.append("  significa que ese indice decidio no mirar a casi nadie, y")
        L.append("  esa informacion la borra la renormalizacion del pipeline.")
        L.append("")
    L.append("  ICC entre semillas                  " + ("%.3f" % icc).rjust(8))
    L.append("  fraccion de aridad 2                "
             + ("%.1f%%" % (100 * ar["frac_par"])).rjust(8))
    L.append("  fraccion de fila                    "
             + ("%.1f%%" % (100 * ar["frac_fila"])).rjust(8))
    L.append("  fraccion de columna                 "
             + ("%.1f%%" % (100 * ar["frac_col"])).rjust(8))
    L.append("")
    L.append("=" * W)
    L.append("  EL PESO DE ATENCION PREDICE SU PROPIO EFECTO?")
    L.append("")
    if peso_vs_efecto:
        L.append("  Spearman A[v,j] contra Datt[v,j]    "
                 + ("%+.3f" % peso_vs_efecto["rho"]).rjust(8)
                 + "   p " + ("%.4g" % peso_vs_efecto["p"]))
        L.append("  media por semilla                   "
                 + ("%+.3f" % peso_vs_efecto["rho_medio_por_semilla"]).rjust(8))
        L.append("  semillas con signo positivo         "
                 + (str(peso_vs_efecto["semillas_positivas"]) + "/"
                    + str(peso_vs_efecto["n_semillas"])).rjust(8))
        L.append("")
        L.append("  Si esto es cercano a cero, el TAMANO del peso no dice nada")
        L.append("  sobre su importancia, y discretizar con Top-P selecciona")
        L.append("  aristas por una cantidad que no predice el efecto de")
        L.append("  quitarlas.")
    L.append("")
    L.append("=" * W)
    L.append("  ARISTAS CUYO CORTE MAS DUELE")
    L.append("")
    L.append("  " + "arista".ljust(28) + "Datt".rjust(12)
             + "veces el ruido".rjust(16))
    for val, i, j in orden[:12]:
        L.append("  " + (INDEX_NAMES[i] + "->" + INDEX_NAMES[j]).ljust(28)
                 + ("%+.3e" % val).rjust(12)
                 + ("%.1f" % (val / ruido if ruido > 0 else
                              float("nan"))).rjust(16))
    L.append("")
    L.append("=" * W)
    L.append("  CONTRA LA VERDAD Y CONTRA LAS OTRAS FUENTES")
    L.append("")
    L.append("  Spearman con |pc|, crudo            "
             + ("%+.3f" % verdad["rho_pc"]).rjust(8))
    L.append("  Spearman con |pc|, parte de par     "
             + ("%+.3f" % verdad["rho_pc_par"]).rjust(8))
    if icp:
        L.append("  auc contra el invariante del ICP    "
                 + ("%.3f" % icp["auc"]).rjust(8)
                 + "   p " + ("%.4f" % icp["p"]))
    L.append("")
    L.append("  " + "fuente".ljust(32) + "rho crudo".rjust(11)
             + "rho par".rjust(10))
    for et, d in comp.items():
        L.append("  " + et[:31].ljust(32)
                 + ("%+.3f" % d["rho_crudo"]).rjust(11)
                 + ("%+.3f" % d["rho_par"]).rjust(10))
    L.append("")
    L.append("  La comparacion que importa es con 'dependencia por ablacion':")
    L.append("  las dos son intervenciones y estan en las mismas unidades de")
    L.append("  MSE, pero una interviene la ENTRADA y la otra la ATENCION.")
    L.append("=" * W)
    txt = "\n".join(L)
    print("\n" + txt)
    os.makedirs(a.salida, exist_ok=True)
    with open(os.path.join(a.salida, "ABLACION_ATENCION" + a.sufijo2 + ".txt"), "w",
              encoding="utf-8") as f:
        f.write(txt)
    with open(os.path.join(a.salida, "ABLACION_ATENCION" + a.sufijo2 + ".json"), "w",
              encoding="utf-8") as f:
        json.dump(dict(n_semillas=M, ruido=ruido, n_efecto=n_efecto,
                       icc=icc, aridad=ar, peso_vs_efecto=peso_vs_efecto,
                       verdad=verdad, icp=icp, comparacion=comp,
                       Datt=Dm.tolist()), f, indent=2, ensure_ascii=False)
    np.save(os.path.join(a.salida, "ablacion_atencion" + a.sufijo2 + ".npy"),
            np.stack(Datts).astype(np.float32))
    print("\n  Guardado: "
          + os.path.join(a.salida, "ABLACION_ATENCION" + a.sufijo2 + ".txt"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
