#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests de que el error por fecha reproduce el camino ya publicado.

POR QUE
-------
errores_por_fecha.py reimplementa el calculo del error: en vez de sumar sobre
todos los frames como error_por_canal, guarda el eje del frame y promedia
despues. Si esa reimplementacion no da exactamente lo mismo, todos los
controles de control_fases.py estarian midiendo otra cosa que las cifras del
README, y la comparacion entre las dos no significaria nada.

Un modulo nuevo que contradice al viejo puede estar bien o mal; lo que no puede
es que nadie lo haya comprobado. Aqui se comprueba contra tres referencias:

  la funcion original, sobre los mismos frames, hasta el ultimo decimal;
  el Datt global publicado en ABLACION_ATENCION_base.json;
  el MSE por fase publicado en DATT_POR_FASE.json.

USO
    srun --partition=student --qos=student --gres=gpu:1 --mem=16G \\
         .venv/bin/python -u test_errores.py
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import traceback

import numpy as np
import torch

sys.path.insert(0, "/srv/pfigueroa")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hello as H  # noqa: E402
from certificar_todo import N_IDX  # noqa: E402
from analisis_fenologico import FASES  # noqa: E402
import ablacion_atencion as AB  # noqa: E402
import errores_por_fecha as EF  # noqa: E402
from datt_por_fase import frames_de_fase  # noqa: E402

FALLOS, CORRIDOS = [], []


def prueba(fn):
    nombre = fn.__name__.replace("t_", "").replace("_", " ")
    try:
        CORRIDOS.append((nombre, True, fn() or ""))
    except Exception as e:  # noqa: BLE001
        CORRIDOS.append((nombre, False, str(e)))
        FALLOS.append((nombre, traceback.format_exc()))
    return fn


def cierto(c, m):
    if not c:
        raise AssertionError(m)


# ------------------------------------------------------------------ montaje

H.CustomTransformerEncoderLayer.forward = AB._forward_con_corte
H.ConvTransformer._recortar_atencion = AB._recortar_sin_renormalizar

STACK = np.load(H.OUTPUT_NPY)
FECHAS = H.cargar_fechas_stack(len(STACK))
MESES = np.array([dt.datetime.utcfromtimestamp(int(m) / 1000).month
                  for m in FECHAS])
_TR, _VA, _IMG, SCALER = H.process_indices_data(
    STACK, seq_length=H.SEQ_LENGTH, dates_millis=FECHAS)

DEV = H.device
IMG = ((H.TOKEN_PARCHE, H.TOKEN_PARCHE) if H.TOKEN_PARCHE
       else (STACK.shape[1], STACK.shape[2]))
USAR_FAM = (H.ABLACION_CONDICIONADA_FAMILIA
            and H.MASCARA_MODO == "familia_balanceada")
MSK = (H._mascara_familias(DEV) if USAR_FAM
       else torch.eye(N_IDX, dtype=torch.bool, device=DEV))

_SEMILLA = list(H.SEEDS)[0]
_CK = os.path.join(H.CHECKPOINT_DIR, "model_seed_" + str(_SEMILLA) + ".pth")
MODELO = H.ConvTransformer(num_indices=H.NUM_INDICES,
                           seq_length=H.SEQ_LENGTH, img_size=IMG,
                           num_heads=4, num_layers=2)
MODELO.load_state_dict(torch.load(_CK, map_location="cpu"))
MODELO = MODELO.to(DEV).eval()

X = SCALER.transform(
    STACK.astype(np.float32).reshape(-1, STACK.shape[-1])
).reshape(STACK.shape).astype(np.float32)
FRAMES = H.trocear_en_parches(X, H.TOKEN_PARCHE)
POR_FECHA = (STACK.shape[1] // H.TOKEN_PARCHE) * (STACK.shape[2]
                                                  // H.TOKEN_PARCHE)


# -------------------------------------------------------------------- tests

@prueba
def t_igual_a_la_funcion_original():
    """El test que decide. errores_por_frame promediado sobre frames tiene que
    dar error_por_canal hasta precision numerica, sobre los mismos frames y
    con la misma mascara."""
    fr = FRAMES[:96]
    AB.COLUMNA_CORTADA = None
    viejo = AB.error_por_canal(MODELO, fr, MSK, 32)
    nuevo = EF.errores_por_frame(MODELO, fr, MSK, 32).mean(axis=0)
    err = float(np.abs(viejo - nuevo).max())
    cierto(err < 1e-6, "difieren en %.3e" % err)
    return "maxima diferencia %.2e sobre %d parches" % (err, len(fr))


@prueba
def t_igual_tambien_con_una_arista_cortada():
    """La equivalencia tiene que valer con el corte puesto, que es el estado en
    el que se calcula Datt."""
    fr = FRAMES[:96]
    peor = 0.0
    for j in (0, 5, 11):
        AB.COLUMNA_CORTADA = j
        viejo = AB.error_por_canal(MODELO, fr, MSK, 32)
        nuevo = EF.errores_por_frame(MODELO, fr, MSK, 32).mean(axis=0)
        peor = max(peor, float(np.abs(viejo - nuevo).max()))
    AB.COLUMNA_CORTADA = None
    cierto(peor < 1e-6, "difieren en %.3e" % peor)
    return "tres cortes, maxima diferencia %.2e" % peor


@prueba
def t_el_parche_k_viene_de_la_fecha_correcta():
    """El troceado es (T*nh*nw, P, P, C) recorriendo primero la fecha, asi que
    el parche k viene de la fecha k // 16. Si eso no fuera cierto, cada error
    quedaria asignado a otra fecha y ninguna cifra lo delataria."""
    for t in (0, 37, len(STACK) - 1):
        directo = H.trocear_en_parches(X[t:t + 1], H.TOKEN_PARCHE)
        tramo = FRAMES[t * POR_FECHA:(t + 1) * POR_FECHA]
        cierto(np.array_equal(directo, tramo),
               "la fecha %d no ocupa el tramo esperado" % t)
    return "verificado en 3 fechas, %d parches cada una" % POR_FECHA


@prueba
def t_agrupar_por_fecha_no_cambia_la_media():
    """Promediar primero dentro de la fecha y luego entre fechas da lo mismo
    que promediar de golpe, porque todas las fechas tienen 16 parches. Si el
    stack tuviera fechas con distinto numero de parches dejaria de valer."""
    fr = FRAMES[:16 * 12]
    e = EF.errores_por_frame(MODELO, fr, MSK, 32)
    plano = e.mean(axis=0)
    por_fecha = np.add.reduceat(
        e, np.arange(0, len(e), POR_FECHA), axis=0) / POR_FECHA
    err = float(np.abs(plano - por_fecha.mean(axis=0)).max())
    cierto(err < 1e-9, "difieren en %.3e" % err)
    return "maxima diferencia %.2e" % err


@prueba
def t_reproduce_el_mse_por_fase_publicado():
    """Contra DATT_POR_FASE.json: el MSE base por fase que salio del camino
    viejo, con su misma regla de igualar fechas."""
    ruta = "resultados/DATT_POR_FASE.json"
    if not os.path.exists(ruta):
        return "saltado, no esta " + ruta
    with open(ruta, encoding="utf-8") as f:
        pub = json.load(f)["fases"]
    npz = "resultados/ERRORES_POR_FECHA_base.npz"
    if not os.path.exists(npz):
        return "saltado, no esta " + npz
    d = np.load(npz, allow_pickle=True)
    E = d["E"].astype(np.float64)
    mse = E[:, :, :, 0].mean(axis=(0, 2))     # por fecha
    n_comun = min(v["n_fechas"] for v in pub.values())
    peor, det = 0.0, []
    for nombre, ms in FASES:
        if nombre not in pub:
            continue
        idx = np.where(np.isin(MESES, ms))[0]
        if len(idx) > n_comun:
            idx = idx[np.linspace(0, len(idx) - 1, n_comun).astype(int)]
        mio = float(mse[idx].mean())
        suyo = float(pub[nombre]["mse_base"])
        peor = max(peor, abs(mio - suyo) / suyo)
        det.append("%s %.4f vs %.4f" % (nombre.split()[0], mio, suyo))
    cierto(peor < 0.02, "se van hasta %.1f%%: %s" % (100 * peor, "; ".join(det)))
    return "maxima desviacion %.2f%% sobre %d fases" % (100 * peor, len(det))


@prueba
def t_frames_de_fase_usa_el_scaler_global():
    """Un scaler por fase pondria cada matriz en otra escala y el MSE dejaria
    de ser comparable entre fases, que es justo lo que se compara."""
    ms = dict(FASES)["dormancia (may-ago)"]
    fr, _n = frames_de_fase(STACK, MESES, ms, SCALER, 10 ** 9, None)
    idx = np.where(np.isin(MESES, ms))[0]
    esperado = H.trocear_en_parches(X[idx], H.TOKEN_PARCHE)
    err = float(np.abs(fr - esperado).max())
    cierto(err < 1e-5, "difiere en %.3e del troceado con el scaler global"
           % err)
    return "coincide con el scaler global, diferencia %.2e" % err


@prueba
def t_el_tensor_guardado_no_tiene_agujeros():
    for suf in ("_base", "_bal", "_ctrl", "_bal2"):
        ruta = "resultados/ERRORES_POR_FECHA" + suf + ".npz"
        if not os.path.exists(ruta):
            continue
        d = np.load(ruta, allow_pickle=True)
        E = d["E"]
        cierto(np.isfinite(E).all(), suf + " tiene valores no finitos")
        cierto((E > 0).all(), suf + " tiene errores no positivos")
        cierto(E.shape[1] == len(MESES),
               suf + " no tiene una fila por fecha")
        cierto(len(d["semillas"]) == E.shape[0],
               suf + " no cuadra el numero de semillas")
    return "cuatro brazos finitos, positivos y con 186 fechas"


@prueba
def t_los_brazos_comparten_semillas():
    """Comparar brazos emparejando por semilla solo vale si son las mismas."""
    sem = {}
    for suf in ("_base", "_bal", "_ctrl", "_bal2"):
        ruta = "resultados/ERRORES_POR_FECHA" + suf + ".npz"
        if os.path.exists(ruta):
            sem[suf] = sorted(np.load(ruta, allow_pickle=True)["semillas"]
                              .tolist())
    if len(sem) < 2:
        return "saltado, menos de dos brazos"
    ref = list(sem.values())[0]
    for suf, s in sem.items():
        cierto(s == ref, suf + " usa otras semillas: " + str(s))
    return "%d brazos con las mismas %d semillas" % (len(sem), len(ref))


def main() -> int:
    W = 78
    print("=" * W)
    print(" TESTS DEL ERROR POR FECHA CONTRA EL CAMINO YA PUBLICADO")
    print("=" * W)
    for nombre, ok, det in CORRIDOS:
        print("  %s  %-42s %s" % ("ok  " if ok else "FALLA", nombre[:42],
                                  det[:90]))
    print("-" * W)
    print("  %d de %d pasan" % (len(CORRIDOS) - len(FALLOS), len(CORRIDOS)))
    for nombre, tb in FALLOS:
        print("\n  FALLA " + nombre + "\n" + tb)
    print("=" * W)
    return 1 if FALLOS else 0


if __name__ == "__main__":
    sys.exit(main())
