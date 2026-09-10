#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Datt disociado por fase fenologica: se apagan las aristas de regimen?

LA PREDICCION QUE SE CONTRASTA
------------------------------
El grafo de 19 aristas verificadas por intervencion trae una etiqueta que sale
de OTRO lado: el ICP sobre los datos crudos dice, para cada arista, si su
coeficiente sobrevive al cambio de fase fenologica (7 invariantes) o si cambia
(12 de regimen). Esa etiqueta se calculo con |beta| por fecha sobre el dato, sin
tocar el modelo.

Aqui se corre Datt restringiendo la ENTRADA a cada fase por separado y se mira
cuantas fases aguanta cada arista por encima de 3 veces el ruido. Si las dos
mediciones concuerdan -las invariantes aguantan en las cuatro fases, las de
regimen se apagan en alguna- entonces una medida del lado del DATO y otra del
lado del MODELO coinciden, que es la unica forma de validacion cruzada
disponible aqui.

Si no concuerdan, la clasificacion del grafo se queda sin respaldo del modelo y
hay que decirlo.

DOS CONTROLES QUE EL DISENO INGENUO NO TIENE
--------------------------------------------
1. EL SUELO DE RUIDO DEPENDE DEL NUMERO DE PARCHES. Con menos parches el error
   por canal es mas ruidoso y el suelo sube, asi que una arista puede "apagarse"
   sin que cambie nada del fenomeno. Las fases tienen entre 8 y 76 fechas, o sea
   casi un factor 10. Aqui se IGUALA el numero de parches entre fases,
   submuestreando al minimo comun, y ademas el suelo se recalcula dentro de cada
   fase con su propia cola negativa.

2. QUE DATOS SE USAN. Se toman todas las fechas de cada fase, incluidas las de
   entrenamiento. No es fuga: no se mide generalizacion sino que dependencia
   expone el modelo en cada regimen, y restringirse al 20% de validacion dejaria
   fases con dos o tres fechas. Es la misma decision que en reinferir_por_fase.py
   y por el mismo motivo.

USO
    srun --partition=student --qos=student --gres=gpu:1 --mem=16G \\
         .venv/bin/python -u datt_por_fase.py --semillas 15
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, "/srv/pfigueroa")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hello as H  # noqa: E402
from certificar_todo import INDEX_NAMES, N_IDX, VAL_RATIO  # noqa: E402
from analisis_fenologico import FASES  # noqa: E402
import ablacion_atencion as AB  # noqa: E402


def frames_de_fase(stack, meses, ms, scaler, n_max, n_fechas_comun=None):
    """Parches de las fechas de una fase, con el scaler GLOBAL.

    El scaler tiene que ser el ajustado sobre todo el tramo de train y no sobre
    la fase: con uno por fase cada matriz saldria en otra escala y el MSE dejaria
    de ser comparable entre fases, que es justo lo que se quiere comparar.
    """
    idx = np.where(np.isin(meses, ms))[0]
    if len(idx) == 0:
        return None, 0
    # Igualar el numero de FECHAS, no solo de parches. Dos lotes de 128 parches
    # no son equivalentes si uno sale de 9 fechas y el otro de 90: los parches
    # de una misma escena estan correlacionados, asi que el de 9 fechas tiene
    # mucho menos tamano de muestra efectivo y su suelo de ruido sube por eso y
    # no por biologia.
    if n_fechas_comun and len(idx) > n_fechas_comun:
        idx = idx[np.linspace(0, len(idx) - 1, n_fechas_comun).astype(int)]
    sel = stack[idx].astype(np.float32)
    plano = sel.reshape(-1, sel.shape[-1])
    sel = scaler.transform(plano).reshape(sel.shape).astype(np.float32)
    if H.TOKEN_PARCHE and H.TOKEN_PARCHE > 0:
        sel = H.trocear_en_parches(sel, H.TOKEN_PARCHE)
    n_fechas = len(idx)
    if len(sel) > n_max:
        sel = sel[np.linspace(0, len(sel) - 1, n_max).astype(int)]
    return sel, n_fechas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--semillas", type=int, default=15)
    ap.add_argument("--lote", type=int, default=32)
    ap.add_argument("--min-fechas", type=int, default=6, dest="min_fechas")
    ap.add_argument("--umbral", type=float, default=3.0)
    ap.add_argument("--modelos", default=None,
                    help="carpeta con model_seed_*.pth y val_losses.json")
    ap.add_argument("--sufijo", default="")
    ap.add_argument("--salida", default="resultados")
    a = ap.parse_args()

    H.CustomTransformerEncoderLayer.forward = AB._forward_con_corte
    H.ConvTransformer._recortar_atencion = AB._recortar_sin_renormalizar

    stack = np.load(H.OUTPUT_NPY)
    fechas = H.cargar_fechas_stack(len(stack))
    meses = np.array([dt.datetime.utcfromtimestamp(int(m) / 1000).month
                      for m in fechas])
    _tr, _va, _img, scaler = H.process_indices_data(
        stack, seq_length=H.SEQ_LENGTH, dates_millis=fechas)

    # Primer pase: contar parches disponibles por fase para fijar el minimo comun.
    disp = {}
    for nombre, ms in FASES:
        idx = np.where(np.isin(meses, ms))[0]
        if len(idx) < a.min_fechas:
            print("  fuera por pocas fechas: " + nombre + " ("
                  + str(len(idx)) + ")")
            continue
        n_par = len(idx) * ((stack.shape[1] // H.TOKEN_PARCHE) ** 2
                            if H.TOKEN_PARCHE else 1)
        disp[nombre] = (ms, len(idx), n_par)
    n_fechas_comun = min(v[1] for v in disp.values())
    n_comun = min(v[2] for v in disp.values())
    print("  fases: " + str(list(disp)))
    print("  fechas por fase, igualadas a " + str(n_fechas_comun))
    print("  parches por fase, igualados a " + str(n_comun))

    dir_mod = a.modelos or H.CHECKPOINT_DIR
    dir_val = (a.modelos if a.modelos and os.path.exists(
        os.path.join(a.modelos, "val_losses.json")) else H.DIR_MATRICES)
    with open(os.path.join(dir_val, "val_losses.json")) as f:
        vl = json.load(f)
    semillas = [s for s in list(H.SEEDS) if str(s) in vl][:a.semillas]
    print("  modelos: " + dir_mod)
    dev = H.device
    img = ((H.TOKEN_PARCHE, H.TOKEN_PARCHE) if H.TOKEN_PARCHE
           else (stack.shape[1], stack.shape[2]))
    ident = torch.eye(N_IDX, dtype=torch.bool, device=dev)
    usar_fam = (H.ABLACION_CONDICIONADA_FAMILIA
                and H.MASCARA_MODO == "familia_balanceada")
    base_msk = H._mascara_familias(dev) if usar_fam else ident

    por_fase: Dict[str, Dict] = {}
    for nombre, (ms, n_f, _n_par) in disp.items():
        fr, n_usadas = frames_de_fase(stack, meses, ms, scaler, n_comun,
                                      n_fechas_comun)
        print("\n  " + nombre + ": " + str(n_usadas) + " fechas usadas de "
              + str(n_f) + ", " + str(len(fr)) + " parches")
        mats = []
        for s in semillas:
            ck = os.path.join(dir_mod, "model_seed_" + str(s) + ".pth")
            if not os.path.exists(ck):
                continue
            m = H.ConvTransformer(num_indices=H.NUM_INDICES,
                                  seq_length=H.SEQ_LENGTH, img_size=img,
                                  num_heads=4, num_layers=2)
            m.load_state_dict(torch.load(ck, map_location="cpu"))
            m = m.to(dev).eval()
            mats.append(AB.matriz_ablacion_atencion(m, fr, base_msk, a.lote))
            del m
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        # MSE base sin cortar nada: es la cifra que dice si el suelo de ruido
        # de una fase sube por biologia o porque el modelo entreno poco ahi.
        AB.COLUMNA_CORTADA = None
        e0 = []
        for s2 in semillas:
            ck2 = os.path.join(dir_mod, "model_seed_" + str(s2) + ".pth")
            if not os.path.exists(ck2):
                continue
            m2 = H.ConvTransformer(num_indices=H.NUM_INDICES,
                                   seq_length=H.SEQ_LENGTH, img_size=img,
                                   num_heads=4, num_layers=2)
            m2.load_state_dict(torch.load(ck2, map_location="cpu"))
            m2 = m2.to(dev).eval()
            e0.append(float(AB.error_por_canal(m2, fr, base_msk,
                                               a.lote).mean()))
            del m2
        mse_base = float(np.mean(e0)) if e0 else float("nan")
        D = np.mean(mats, axis=0)
        neg = D[D < 0]
        ruido = float(np.abs(neg).mean()) if neg.size else float("nan")
        por_fase[nombre] = dict(D=D, ruido=ruido, mse_base=mse_base,
                                n_fechas=n_usadas,
                                n_parches=int(len(fr)), M=len(mats),
                                n_sobre=int((D > a.umbral * ruido).sum()))
        print("    ruido %.3e   aristas sobre %gx: %d"
              % (ruido, a.umbral, por_fase[nombre]["n_sobre"]))

    # Aristas de referencia: las 19 del grafo global.
    with open(os.path.join(a.salida, "ABLACION_ATENCION_base.json"),
              encoding="utf-8") as f:
        base = json.load(f)
    Dg = np.array(base["Datt"])
    rg = float(base["ruido"])
    ref = [(i, j) for i in range(N_IDX) for j in range(N_IDX)
           if i != j and Dg[i, j] > a.umbral * rg]
    with open(os.path.join(a.salida, "ANALISIS_ICP.json"),
              encoding="utf-8") as f:
        q = {(x["i"], x["j"]): x["q"] for x in json.load(f)["aristas"]}

    nombres = list(por_fase)
    filas = []
    for (i, j) in sorted(ref, key=lambda t: -Dg[t[0], t[1]]):
        veces = {n: por_fase[n]["D"][i, j] / por_fase[n]["ruido"]
                 for n in nombres}
        n_sobre = sum(1 for v in veces.values() if v > a.umbral)
        filas.append(dict(
            i=int(i), j=int(j),
            arista=INDEX_NAMES[i] + " <- " + INDEX_NAMES[j],
            global_veces=float(Dg[i, j] / rg),
            invariante_icp=bool(q.get((i, j), 0.0) > 0.10),
            veces={n: float(v) for n, v in veces.items()},
            fases_sobre_umbral=int(n_sobre)))

    inv = [f for f in filas if f["invariante_icp"]]
    reg = [f for f in filas if not f["invariante_icp"]]
    u = (stats.mannwhitneyu([f["fases_sobre_umbral"] for f in inv],
                            [f["fases_sobre_umbral"] for f in reg],
                            alternative="greater")
         if inv and reg else None)

    W = 82
    L = ["=" * W, " Datt DISOCIADO POR FASE FENOLOGICA", "",
         " Se corre la ablacion de aristas restringiendo la entrada a cada fase.",
         " La etiqueta invariante/regimen viene del ICP sobre los DATOS crudos,",
         " sin tocar el modelo, asi que si las dos coinciden es validacion",
         " cruzada entre una medida del dato y una del modelo.",
         "",
         " Igualados entre fases el numero de FECHAS (" + str(n_fechas_comun)
         + ") y de parches (" + str(n_comun) + "),",
         " y el suelo de ruido recalculado dentro de cada fase. Igualar solo",
         " parches no basta: los de una misma escena estan correlacionados, asi",
         " que 128 parches de 9 fechas valen menos que 128 de 90.",
         "=" * W, ""]
    L.append("  " + "fase".ljust(24) + "fechas".rjust(7) + "parches".rjust(9)
             + "MSE base".rjust(11) + "ruido".rjust(12)
             + "aristas sobre 3x".rjust(18))
    for n in nombres:
        d = por_fase[n]
        L.append("  " + n[:23].ljust(24) + str(d["n_fechas"]).rjust(7)
                 + str(d["n_parches"]).rjust(9)
                 + ("%.4f" % d["mse_base"]).rjust(11)
                 + ("%.2e" % d["ruido"]).rjust(12)
                 + str(d["n_sobre"]).rjust(18))
    L.append("")
    L.append("  LAS 19 ARISTAS DEL GRAFO GLOBAL, FASE POR FASE")
    L.append("  (numeros en multiplos del ruido de esa fase; * = sobre 3x)")
    L.append("")
    L.append("  " + "arista".ljust(26) + "ICP".ljust(6) + "global".rjust(8)
             + "".join(n[:9].rjust(11) for n in nombres) + "fases".rjust(7))
    for f in filas:
        fila = ("  " + f["arista"][:25].ljust(26)
                + ("inv" if f["invariante_icp"] else "reg").ljust(6)
                + ("%.1f" % f["global_veces"]).rjust(8))
        for n in nombres:
            v = f["veces"][n]
            fila += (("%.1f" % v) + ("*" if v > a.umbral else " ")).rjust(11)
        fila += (str(f["fases_sobre_umbral"]) + "/"
                 + str(len(nombres))).rjust(7)
        L.append(fila)
    L.append("")
    L.append("  Fases por encima del umbral, promedio")
    L.append("    invariantes segun el ICP (" + str(len(inv)) + "):  "
             + ("%.2f" % np.mean([f["fases_sobre_umbral"] for f in inv])
                if inv else "-"))
    L.append("    de regimen segun el ICP (" + str(len(reg)) + "):   "
             + ("%.2f" % np.mean([f["fases_sobre_umbral"] for f in reg])
                if reg else "-"))
    if u is not None:
        L.append("    Mann-Whitney unilateral (invariantes aguantan mas): "
                 "p = " + ("%.4f" % u.pvalue))
        L.append("")
        L.append("    La hipotesis es que las invariantes aguanten en mas")
        L.append("    fases. Un p alto significa que la clasificacion del ICP")
        L.append("    no se refleja en la persistencia de Datt, y entonces las")
        L.append("    dos medidas no se validan entre si.")
    L.append("=" * W)
    txt = "\n".join(L)
    print("\n" + txt)
    os.makedirs(a.salida, exist_ok=True)
    with open(os.path.join(a.salida, "DATT_POR_FASE" + a.sufijo + ".txt"), "w",
              encoding="utf-8") as f:
        f.write(txt)
    with open(os.path.join(a.salida, "DATT_POR_FASE" + a.sufijo + ".json"), "w",
              encoding="utf-8") as f:
        json.dump(dict(aristas=filas, n_parches_comun=int(n_comun),
                       fases={n: dict(ruido=por_fase[n]["ruido"],
                                      mse_base=por_fase[n]["mse_base"],
                                      n_fechas=por_fase[n]["n_fechas"],
                                      n_parches=por_fase[n]["n_parches"],
                                      M=por_fase[n]["M"],
                                      D=por_fase[n]["D"].tolist())
                              for n in nombres},
                       p_mannwhitney=(float(u.pvalue) if u is not None
                                      else None)),
                  f, indent=2, ensure_ascii=False)
    print("\n  Guardado: " + os.path.join(a.salida, "DATT_POR_FASE" + a.sufijo + ".txt"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
