#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Re-inferencia por fase fenologica sobre los modelos ya entrenados.

LA PREGUNTA
-----------
El catastro CIREN situa un vivero de vid vinifera de 41 ha en el centro del
recorte, asi que la fenologia del cultivo es verdad externa. Y la estructura
de dependencia de los datos cambia mucho entre fases: la correlacion de Spearman
entre el |pc| de dormancia y el de crecimiento es 0.510, no 1.

Los 50 modelos se entrenaron sobre todas las fechas mezcladas, de modo que su
matriz agregada promedia regimenes que difieren entre si tanto como eso. La
pregunta que este script contesta es si ese promedio es una perdida
irrecuperable o solo una consecuencia de como se LEE el modelo:

    evaluado sobre las fechas de una fase, el modelo produce la estructura
    de esa fase, o produce siempre la misma matriz?

Si la matriz cambia con la fase y ademas se acerca al |pc| de la fase en que se
evalua, entonces el modelo si aprendio los regimenes y el problema estaba en
agregar la lectura, no en el entrenamiento. Si sale la misma matriz siempre, el
modelo colapso los regimenes y la fenologia se perdio de verdad.

Es una prediccion falsable y con verdad externa agronomica, no estadistica
inventada sobre los propios datos.

NOTA SOBRE QUE DATOS SE USAN. Se evalua sobre TODAS las fechas de cada fase,
incluidas las de entrenamiento. No es fuga: no se mide generalizacion sino que
estructura relacional expone el modelo en cada regimen, y restringirse al 20% de
validacion dejaria fases con dos o tres fechas.

USO
    .venv/bin/python reinferir_por_fase.py --semillas 15
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, "/srv/pfigueroa")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hello as H  # noqa: E402
from certificar_todo import PARES, fuerza_por_par, pc_y_r  # noqa: E402
from analisis_fenologico import CULTIVO, FASES  # noqa: E402

INDEX_NAMES = H.INDEX_NAMES


def frames_de(stack: np.ndarray, idx: np.ndarray, scaler) -> np.ndarray:
    """Frames de las fechas `idx`, estandarizados y troceados como en el
    entrenamiento.

    Hay que reproducir exactamente process_indices_data: estandarizar con el
    scaler ajustado sobre el tramo de train COMPLETO -no sobre la fase, que
    daria otra escala y haria incomparables las matrices entre fases- y luego
    trocear en parches de TOKEN_PARCHE. El modelo se entreno con img_size
    (13,13), asi que recibe parches y no frames enteros; pasarle 52x52 da un
    frame_proj de 43264 contra los 4096 del checkpoint.
    """
    sel = stack[idx].astype(np.float32)
    plano = sel.reshape(-1, sel.shape[-1])
    sel = scaler.transform(plano).reshape(sel.shape).astype(np.float32)
    if H.TOKEN_PARCHE and H.TOKEN_PARCHE > 0:
        sel = H.trocear_en_parches(sel, H.TOKEN_PARCHE)
    return sel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--semillas", type=int, default=15)
    ap.add_argument("--max-frames", type=int, default=24, dest="max_frames")
    ap.add_argument("--salida", default="resultados")
    a = ap.parse_args()

    stack = np.load(H.OUTPUT_NPY)
    fechas = H.cargar_fechas_stack(len(stack))
    meses = np.array([dt.datetime.utcfromtimestamp(int(m) / 1000).month
                      for m in fechas])
    print(f"  cultivo: {CULTIVO['especie']} ({CULTIVO['comuna']}), "
          f"{CULTIVO['superficie_ha']} ha")

    with open(os.path.join(H.DIR_MATRICES, "val_losses.json")) as f:
        vl = json.load(f)
    semillas = [int(s) for s, _ in
                sorted(vl.items(), key=lambda kv: kv[1])][:a.semillas]
    print(f"  semillas (mejores por val_loss): {len(semillas)}")

    dev = H.device
    # El modelo se entreno sobre PARCHES, no sobre el frame entero: img_size es
    # (TOKEN_PARCHE, TOKEN_PARCHE). Con (52,52) el frame_proj sale de 43264 y
    # no carga contra los 4096 del checkpoint.
    img = ((H.TOKEN_PARCHE, H.TOKEN_PARCHE) if H.TOKEN_PARCHE
           else (int(stack.shape[1]), int(stack.shape[2])))
    # Scaler global, el mismo que uso el entrenamiento.
    (_tr, _), (_va, _), _img, scaler = H.process_indices_data(
        stack, seq_length=H.SEQ_LENGTH, dates_millis=fechas)
    print(f"  img_size={img}  parches por frame="
          f"{(stack.shape[1] // H.TOKEN_PARCHE) ** 2 if H.TOKEN_PARCHE else 1}")

    # Verdad por fase.
    pcs = {}
    for nombre, ms in FASES:
        idx = np.where(np.isin(meses, ms))[0]
        if len(idx) < 8:
            continue
        PC, _ = pc_y_r(stack[idx])
        pcs[nombre] = np.array([PC[i, j] for i, j in PARES])
    fases = list(pcs)
    print(f"  fases con datos: {fases}")

    resultados = {}
    for nombre in fases:
        ms = dict(FASES)[nombre]
        idx = np.where(np.isin(meses, ms))[0]
        if len(idx) > a.max_frames:
            idx = idx[np.linspace(0, len(idx) - 1, a.max_frames).astype(int)]
        X = frames_de(stack, idx, scaler)
        print(f"\n  {nombre}: {len(idx)} frames")
        abl, att = [], []
        for s in semillas:
            ck = os.path.join(H.CHECKPOINT_DIR, f"model_seed_{s}.pth")
            if not os.path.exists(ck):
                continue
            m = H.ConvTransformer(num_indices=H.NUM_INDICES,
                                  seq_length=H.SEQ_LENGTH, img_size=img,
                                  num_heads=4, num_layers=2)
            m.load_state_dict(torch.load(ck, map_location="cpu"))
            m = m.to(dev).eval()
            try:
                D, _ = H.matriz_dependencia_ablacion(m, X, verbose=False)
                abl.append(np.asarray(D, dtype=np.float64))
            except Exception as e:
                print(f"    ablacion fallo en semilla {s}: "
                      f"{type(e).__name__}: {e}")
            try:
                xin = torch.from_numpy(np.ascontiguousarray(
                    np.transpose(X[:16][:, None, ...], (0, 4, 1, 2, 3)))
                ).float()
                R, _, _ = H.compute_attention_for_batch(m, xin)
                att.append(np.asarray(R, dtype=np.float64))
            except Exception as e:
                print(f"    atencion fallo en semilla {s}: "
                      f"{type(e).__name__}: {e}")
            del m
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        resultados[nombre] = {}
        for et, mats in (("ablacion", abl), ("atencion", att)):
            if len(mats) < 2:
                continue
            fr = fuerza_por_par(np.mean(mats, axis=0))
            resultados[nombre][et] = {
                f: float(stats.spearmanr(fr, pcs[f]).statistic) for f in fases}
            resultados[nombre][et]["n_semillas"] = len(mats)
            print(f"    {et}: " + "  ".join(
                f"{f[:11]} {resultados[nombre][et][f]:+.3f}" for f in fases))

    W = 78
    L = ["=" * W, " RE-INFERENCIA POR FASE FENOLOGICA", "",
         f" Cultivo {CULTIVO['especie']} ({CULTIVO['fuente']}).",
         " Los 50 modelos se entrenaron con todas las fechas mezcladas. Aqui se",
         " los evalua restringiendo la entrada a cada fase, sin reentrenar.",
         "", " Lee la DIAGONAL: si el valor mas alto de cada fila esta en su",
         " propia fase, el modelo expone la estructura del regimen en que se lo",
         " evalua y el promedio agregado era el problema, no el entrenamiento.",
         "=" * W]
    for et in ("ablacion", "atencion"):
        L.append("")
        L.append(f"  {et.upper()}   (filas: fase de evaluacion; "
                 f"columnas: |pc| de referencia)")
        L.append("  " + " " * 24 + "".join(f"{f[:11]:>12}" for f in fases))
        for na in fases:
            d = resultados.get(na, {}).get(et)
            if not d:
                continue
            fila = f"  {na:<24}"
            for nb in fases:
                v = d.get(nb, float('nan'))
                marca = "*" if nb == na else " "
                fila += f"{v:>+11.3f}{marca}"
            L.append(fila)
        aciertos = sum(1 for na in fases
                       if resultados.get(na, {}).get(et)
                       and max(fases, key=lambda nb: resultados[na][et][nb]) == na)
        L.append(f"    fases donde el maximo cae en su propia fase: "
                 f"{aciertos}/{len(fases)}")
    L.append("=" * W)
    txt = "\n".join(L)
    print("\n" + txt)
    os.makedirs(a.salida, exist_ok=True)
    with open(os.path.join(a.salida, "REINFERENCIA_FASE.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt)
    with open(os.path.join(a.salida, "REINFERENCIA_FASE.json"), "w",
              encoding="utf-8") as f:
        json.dump(dict(cultivo=CULTIVO, resultados=resultados), f, indent=2,
                  ensure_ascii=False)
    print(f"\n  Guardado: {os.path.join(a.salida, 'REINFERENCIA_FASE.txt')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
