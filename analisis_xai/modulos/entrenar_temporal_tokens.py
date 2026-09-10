#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Darle tiempo al transformer sin tocar el pipeline: los retardos como tokens.

EL PROBLEMA
-----------
La tarea de reconstruccion enmascarada descarta las ventanas temporales por
diseno. En process_indices_data:

    if TAREA == "enmascarado":
        # No hay ventanas temporales: cada frame es independiente

y forward_enmascarado codifica cada token con reshape(B*N, 1, H, W), o sea una
imagen y un instante. La arquitectura si admite seq_length > 1 -ConvEncoder
recibe in_channels = seq_length- pero el camino de la tarea enmascarada no lo
usa. Con esa configuracion el modelo no puede captar ninguna relacion
temporal, y el regimen temporal del experimento sintetico lo confirma: todos
los metodos sin acceso al pasado quedan en el azar o por debajo, incluido el
transformer.

Eso no refuta que el transformer capte tiempo. Refuta que se le haya dado.

LA SOLUCION SIN CIRUGIA
-----------------------
En vez de reescribir forward_enmascarado -que es la ruta optimizada del
pipeline, y romperla afectaria a todo lo demas- se le dan los retardos como
CANALES ADICIONALES. Doce indices por tres instantes son 36 tokens:

    tokens  0..11   X(t)
    tokens 12..23   X(t-1)
    tokens 24..35   X(t-2)

El modelo reconstruye cada uno de los 36 a partir de los otros 35, asi que la
atencion y la ablacion viven en una matriz 36x36 donde una arista puede cruzar
instantes. Es exactamente la informacion que se le dio a LOCO con retardos, ni
mas ni menos, asi que la comparacion sigue siendo simetrica.

COMO SE VUELVE A 12x12
----------------------
La verdad es un DAG de 12 nodos. Para puntuar, la fila es el token presente de
i y se suman las columnas de los tres instantes de j:

    S[i,j] = suma_b D[i, j en el instante b]

Sumar y no tomar el maximo es lo que corresponde: LOCO con retardos quita
TODOS los retardos de j a la vez, asi que su cifra es la contribucion total de
la variable j. Se guarda tambien el maximo por si la eleccion importara.

USO
    srun --partition=student --qos=student --gres=gpu:1 --mem=16G \
         .venv/bin/python -u entrenar_temporal_tokens.py --datos datos_rep11
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

N_BASE = 12

H.MASCARA_MODO = "aleatoria"
H.MASCARA_K = 1
H.ABLACION_CONDICIONADA_FAMILIA = False


def apilar_retardos(X, retardos=(1, 2)):
    """(T,H,W,12) a (T,H,W,12*(1+len(retardos))) con el pasado como canales.

    Las primeras fechas no tienen pasado; se repite la primera disponible en
    vez de recortar la serie, para que el numero de fechas siga siendo el
    mismo que en los otros regimenes y las cifras sean comparables.
    """
    bloques = [X]
    for r in retardos:
        prev = np.roll(X, r, axis=0)
        prev[:r] = X[:r]
        bloques.append(prev)
    return np.concatenate(bloques, axis=-1).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", default="datos_sinteticos")
    ap.add_argument("--regimen", default="temporal")
    ap.add_argument("--semillas", type=int, default=5)
    ap.add_argument("--epocas", type=int, default=600)
    ap.add_argument("--paciencia", type=int, default=60)
    ap.add_argument("--lote", type=int, default=64)
    ap.add_argument("--retardos", type=int, nargs="*", default=[1, 2])
    ap.add_argument("--stack", default=None,
                    help="ruta a un .npy cualquiera; con esto se corre sobre "
                         "el vinedo real en vez de sobre el sintetico")
    ap.add_argument("--etiqueta-salida", default=None, dest="etsal")
    a = ap.parse_args()

    if a.stack:
        X = np.load(a.stack)
        # Las fechas de Sentinel-2 no son equiespaciadas: hay revisita de 5
        # dias y huecos por nubes. "t-1" es la escena anterior DISPONIBLE, no
        # un intervalo fijo, asi que el retardo esta emborronado y hay que
        # saber cuanto antes de leer nada. Se reporta la distribucion.
        try:
            f = H.cargar_fechas_stack(len(X))
            if f is not None:
                d = np.diff(np.asarray(f, dtype=np.float64)) / 86400000.0
                print("  huecos entre escenas en dias: mediana %.0f, "
                      "p10 %.0f, p90 %.0f, max %.0f"
                      % (np.median(d), np.percentile(d, 10),
                         np.percentile(d, 90), d.max()))
                print("  ATENCION: el retardo 1 es la escena anterior, no un "
                      "intervalo fijo")
        except Exception as e:  # noqa: BLE001
            print("  sin fechas para reportar huecos (" + type(e).__name__
                  + ")")
    else:
        X = np.load(os.path.join(a.datos, "stack_" + a.regimen + ".npy"))
    Xl = apilar_retardos(X, tuple(a.retardos))
    n_tok = Xl.shape[-1]
    H.NUM_INDICES = n_tok
    # INDEX_NAMES tiene doce nombres y el pipeline lo recorre hasta n_idx en
    # el reporte del baseline lineal, asi que con 36 tokens revienta con
    # IndexError. Se extiende con nombres que dicen indice e instante, que
    # ademas hacen legible cualquier salida que los imprima.
    H.INDEX_NAMES = [("X%02d_t%d" % (k, b)) for b in range(n_tok // N_BASE)
                     for k in range(N_BASE)]
    H.INDEX_FAMILIES = {}
    print("  %s a %s, %d tokens (%d indices x %d instantes)"
          % (X.shape, Xl.shape, n_tok, N_BASE, 1 + len(a.retardos)))

    etsal = a.etsal or (a.regimen + "_tokens")
    destino = os.path.join(a.datos, "modelos_" + etsal)
    os.makedirs(destino, exist_ok=True)
    semillas = list(H.SEEDS)[:a.semillas]
    for s in semillas:
        ck = os.path.join(destino, "model_seed_" + str(s) + ".pth")
        if os.path.exists(ck):
            print("  semilla %d ya estaba" % s)
            continue
        t0 = time.time()
        H.set_seed(s)
        _m, _v, _h, best = H.train_convtransformer(
            Xl, seq_length=1, total_epochs=a.epocas,
            batch_size=H.BATCH_SIZE, lr=H.LR_CONVTRANSFORMER,
            patience=a.paciencia, num_workers=0, ckpt_path=ck,
            dates_millis=None)
        print("  semilla %d en %.1f min  val=%.6f"
              % (s, (time.time() - t0) / 60, best))

    H.CustomTransformerEncoderLayer.forward = AB._forward_con_corte
    H.ConvTransformer._recortar_atencion = AB._recortar_sin_renormalizar

    _tr, _va, _img, scaler = H.process_indices_data(
        Xl, seq_length=1, dates_millis=None)
    Z = scaler.transform(Xl.reshape(-1, n_tok)).reshape(Xl.shape)
    Z = Z.astype(np.float32)
    frames = (H.trocear_en_parches(Z, H.TOKEN_PARCHE)
              if H.TOKEN_PARCHE else Z)
    if len(frames) > 2976:
        frames = frames[np.linspace(0, len(frames) - 1, 2976).astype(int)]
    print("  %d parches para la ablacion" % len(frames))

    dev = H.device
    img = ((H.TOKEN_PARCHE, H.TOKEN_PARCHE) if H.TOKEN_PARCHE
           else (Xl.shape[1], Xl.shape[2]))
    msk = torch.eye(n_tok, dtype=torch.bool, device=dev)

    salida = {}
    for nombre in ("atencion", "entrada"):
        mats = []
        for s in semillas:
            ck = os.path.join(destino, "model_seed_" + str(s) + ".pth")
            if not os.path.exists(ck):
                continue
            m = H.ConvTransformer(num_indices=n_tok, seq_length=1,
                                  img_size=img, num_heads=4, num_layers=2)
            m.load_state_dict(torch.load(ck, map_location="cpu"))
            m = m.to(dev).eval()
            if nombre == "atencion":
                mats.append(AB.matriz_ablacion_atencion(m, frames, msk,
                                                        a.lote))
            else:
                AB.COLUMNA_CORTADA = None
                _n, cruda = H.matriz_dependencia_ablacion(
                    m, frames, lote=a.lote, verbose=False)
                mats.append(np.asarray(cruda, dtype=np.float64))
            del m
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if not mats:
            continue
        G = np.mean(mats, axis=0)
        np.fill_diagonal(G, 0.0)
        n_lag = n_tok // N_BASE
        # fila = token presente de i; columna = suma sobre los instantes de j
        S = np.zeros((N_BASE, N_BASE))
        Smax = np.zeros((N_BASE, N_BASE))
        for i in range(N_BASE):
            for j in range(N_BASE):
                if i == j:
                    continue
                v = [G[i, b * N_BASE + j] for b in range(n_lag)]
                S[i, j] = float(np.sum(v))
                Smax[i, j] = float(np.max(v))
        off = ~np.eye(N_BASE, dtype=bool)
        neg = S[off & (S < 0)]
        r = float(np.abs(neg).mean()) if neg.size else float("nan")
        salida[nombre] = dict(Datt=S.tolist(), Datt_max=Smax.tolist(),
                              G=G.tolist(), M=len(mats), n_tokens=n_tok,
                              ruido=r)
        print("  ablacion de %-9s (%d tokens) ruido %.3e  sobre 3x: %d"
              % (nombre, n_tok, r, int((S > 3 * r).sum())))

    ruta = os.path.join(a.datos, "DATT_" + etsal + ".json")
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(salida, f)
    print("  guardado " + ruta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
