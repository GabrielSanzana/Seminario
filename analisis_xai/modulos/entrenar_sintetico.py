#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Entrena el ConvTransformer sobre los datos sinteticos y calcula Datt.

Misma arquitectura, mismo enmascarado, mismo procedimiento de ablacion que
sobre el vinedo. Lo unico que cambia es que ahora la estructura verdadera se
conoce, asi que la matriz Datt que sale de aqui se puede puntuar.

SOBRE LAS FAMILIAS ESPECTRALES
------------------------------
El pipeline enmascara por familia, y las familias vienen de las formulas de
los indices reales. Sobre datos sinteticos esa agrupacion es una particion
arbitraria de los doce canales, sin ninguna relacion con el DAG que los
genero. No favorece ni perjudica: es ruido de configuracion. Se deja como
esta, porque lo que se evalua es el metodo tal como se aplico al caso real y
no una version idealizada.

Por si acaso, la ablacion se corre con las dos mascaras base -la de familia y
la identidad- y las dos se puntuan. Si el resultado dependiera de esa
eleccion, habria que decirlo.

USO
    srun --partition=student --qos=student --gres=gpu:1 --mem=16G \\
         .venv/bin/python -u entrenar_sintetico.py --regimen no_lineal
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/srv/pfigueroa")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hello as H  # noqa: E402
import ablacion_atencion as AB  # noqa: E402

N_IDX = 12


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimen", default="no_lineal",
                    choices=("lineal", "no_lineal"))
    ap.add_argument("--semillas", type=int, default=5)
    ap.add_argument("--epocas", type=int, default=H.TOTAL_EPOCHS)
    ap.add_argument("--paciencia", type=int, default=H.PATIENCE)
    ap.add_argument("--lote", type=int, default=64)
    ap.add_argument("--etiqueta", default="",
                    help="sufijo de la carpeta de modelos y del JSON de Datt")
    ap.add_argument("--datos", default="datos_sinteticos")
    a = ap.parse_args()

    destino = os.path.join(a.datos, "modelos_" + a.regimen + a.etiqueta)
    os.makedirs(destino, exist_ok=True)
    stack = np.load(os.path.join(a.datos, "stack_" + a.regimen + ".npy"))
    print("  stack " + str(stack.shape) + "  regimen " + a.regimen)

    semillas = list(H.SEEDS)[:a.semillas]
    for s in semillas:
        ck = os.path.join(destino, "model_seed_" + str(s) + ".pth")
        if os.path.exists(ck):
            print("  semilla %d ya estaba" % s)
            continue
        t0 = time.time()
        H.set_seed(s)
        _m, _v, _h, best = H.train_convtransformer(
            stack, seq_length=H.SEQ_LENGTH, total_epochs=a.epocas,
            batch_size=H.BATCH_SIZE, lr=H.LR_CONVTRANSFORMER,
            patience=a.paciencia, num_workers=0, ckpt_path=ck,
            dates_millis=None)
        print("  semilla %d en %.1f min  val=%.6f"
              % (s, (time.time() - t0) / 60, best))

    # ---- ablacion
    H.CustomTransformerEncoderLayer.forward = AB._forward_con_corte
    H.ConvTransformer._recortar_atencion = AB._recortar_sin_renormalizar

    _tr, _va, _img, scaler = H.process_indices_data(
        stack, seq_length=H.SEQ_LENGTH, dates_millis=None)
    X = scaler.transform(
        stack.astype(np.float32).reshape(-1, stack.shape[-1])
    ).reshape(stack.shape).astype(np.float32)
    frames = (H.trocear_en_parches(X, H.TOKEN_PARCHE)
              if H.TOKEN_PARCHE else X)
    # La ablacion sobre el vinedo usa un tope de parches; aqui se toma una
    # submuestra regular del mismo tamano para que el coste y el suelo de
    # ruido sean comparables entre los dos experimentos.
    if len(frames) > 2976:
        frames = frames[np.linspace(0, len(frames) - 1, 2976).astype(int)]
    print("  %d parches para la ablacion" % len(frames))

    dev = H.device
    img = ((H.TOKEN_PARCHE, H.TOKEN_PARCHE) if H.TOKEN_PARCHE
           else (stack.shape[1], stack.shape[2]))
    mascaras = dict(
        familia=H._mascara_familias(dev),
        identidad=torch.eye(N_IDX, dtype=torch.bool, device=dev))

    salida = {}
    for nombre, msk in mascaras.items():
        mats = []
        for s in semillas:
            ck = os.path.join(destino, "model_seed_" + str(s) + ".pth")
            if not os.path.exists(ck):
                continue
            m = H.ConvTransformer(num_indices=H.NUM_INDICES,
                                  seq_length=H.SEQ_LENGTH, img_size=img,
                                  num_heads=4, num_layers=2)
            m.load_state_dict(torch.load(ck, map_location="cpu"))
            m = m.to(dev).eval()
            mats.append(AB.matriz_ablacion_atencion(m, frames, msk, a.lote))
            del m
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        D = np.mean(mats, axis=0)
        off = ~np.eye(N_IDX, dtype=bool)
        neg = D[off & (D < 0)]
        ruido = float(np.abs(neg).mean()) if neg.size else float("nan")
        salida[nombre] = dict(Datt=D.tolist(), ruido=ruido,
                              M=len(mats),
                              por_semilla=[m.tolist() for m in mats])
        print("  mascara %-10s ruido %.3e  aristas sobre 3x: %d"
              % (nombre, ruido, int((D > 3 * ruido).sum())))

    ruta = os.path.join(a.datos, "DATT_" + a.regimen + a.etiqueta + ".json")
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(salida, f)
    print("  guardado " + ruta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
