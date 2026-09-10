#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
El error de reconstruccion FECHA POR FECHA, con y sin cada arista cortada.

POR QUE ESTE MODULO EXISTE
--------------------------
datt_por_fase.py agrega por fase dentro de la misma pasada por GPU: escoge las
fechas de una fase, promedia, y devuelve cinco numeros. Eso obliga a volver a
correr el modelo cada vez que se quiere cambiar de agrupacion, y deja sin
respuesta las preguntas que deciden si el efecto por fase es real:

  cuanta de la diferencia entre fases es ruido entre semillas;
  si un grupo CUALQUIERA de 8 fechas da el mismo salto que dormancia;
  si lo que predice el error es la fenologia o la cantidad de entrenamiento;
  cuanto cambia el resultado segun que fechas se escojan al igualar tamanos.

Todas se contestan con la misma cantidad: el error por fecha. Aqui se calcula
una sola vez y se guarda, y control_fases.py hace el resto sin tocar la GPU.

QUE SE GUARDA
-------------
    E[semilla, fecha, canal, corte]

con corte 0 = sin cortar nada y corte j+1 = con la columna j de la atencion en
cero. De ahi salen, por simple promedio sobre el eje de fechas:

    el MSE base de cualquier conjunto de fechas
    la matriz Datt de cualquier conjunto de fechas
    el suelo de ruido de cualquier conjunto de fechas

Es decir, cualquier agrupacion -las cinco fases, un placebo al azar, una
submuestra- sale del mismo tensor, sin volver a evaluar el modelo. Eso es lo
que hace posible correr dos mil placebos.

UNA FECHA, NO UN PARCHE
-----------------------
El promedio sobre los 16 parches de una escena se hace aqui y no despues. Los
parches de una misma fecha comparten atmosfera, angulo solar y estado del
cultivo, asi que no son muestras independientes: tratarlos como tales inflaria
cualquier intervalo de confianza. La unidad de muestreo es la fecha.

USO
    srun --partition=student --qos=student --gres=gpu:1 --mem=16G \\
         .venv/bin/python -u errores_por_fecha.py --modelos resultados/fase_ctrl \\
         --sufijo _ctrl
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/srv/pfigueroa")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hello as H  # noqa: E402
from certificar_todo import INDEX_NAMES, N_IDX  # noqa: E402
import ablacion_atencion as AB  # noqa: E402


def errores_por_frame(base, frames: np.ndarray, msk, lote: int) -> np.ndarray:
    """MSE de cada (frame, canal), sin promediar entre frames.

    Es error_por_canal pero conservando el eje del frame. Promediando despues
    sobre el eje 0 se recupera exactamente lo que devuelve aquella, asi que las
    cifras siguen siendo las mismas y comparables con lo ya publicado.
    """
    N = base.num_tokens
    dev = next(base.parameters()).device
    out = np.zeros((len(frames), N), dtype=np.float64)
    with torch.no_grad():
        for ini in range(0, len(frames), lote):
            fr = torch.from_numpy(np.ascontiguousarray(np.transpose(
                frames[ini:ini + lote], (0, 3, 1, 2)))).float().to(dev)
            B = fr.size(0)
            m = msk.unsqueeze(0).expand(B, N, N).clone()
            rec = base.forward_enmascarado(fr, m)
            err = ((rec - fr) ** 2).mean(dim=(2, 3))
            out[ini:ini + B] = err.detach().cpu().numpy()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--semillas", type=int, default=15)
    ap.add_argument("--lote", type=int, default=64)
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

    # El scaler es el ajustado sobre el tramo de train cronologico, el mismo
    # que usan datt_por_fase.py y ablacion_atencion.py. Tiene que ser uno solo
    # para todas las fechas: con un scaler por fase cada matriz saldria en otra
    # escala y el MSE dejaria de ser comparable entre fases, que es justo lo
    # que se quiere comparar.
    X = stack.astype(np.float32)
    plano = X.reshape(-1, X.shape[-1])
    X = scaler.transform(plano).reshape(X.shape).astype(np.float32)
    n_fechas = len(X)
    if H.TOKEN_PARCHE and H.TOKEN_PARCHE > 0:
        nh = X.shape[1] // H.TOKEN_PARCHE
        nw = X.shape[2] // H.TOKEN_PARCHE
        por_fecha = nh * nw
        frames = H.trocear_en_parches(X, H.TOKEN_PARCHE)
    else:
        por_fecha = 1
        frames = X
    # trocear_en_parches devuelve (T*nh*nw, P, P, C) recorriendo primero la
    # fecha, asi que el parche k viene de la fecha k // por_fecha. Se comprueba
    # en vez de suponerse: si el orden cambiara, cada error quedaria asignado a
    # la fecha equivocada y ninguna cifra de salida lo delataria.
    if len(frames) != n_fechas * por_fecha:
        raise ValueError("el troceado no da %d parches por fecha: %d parches "
                         "para %d fechas" % (por_fecha, len(frames), n_fechas))
    fecha_de_parche = np.repeat(np.arange(n_fechas), por_fecha)
    print("  %d fechas, %d parches por fecha, %d parches"
          % (n_fechas, por_fecha, len(frames)))

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

    usadas = []
    E = np.zeros((len(semillas), n_fechas, N_IDX, N_IDX + 1), dtype=np.float32)
    for k, s in enumerate(semillas):
        ck = os.path.join(dir_mod, "model_seed_" + str(s) + ".pth")
        if not os.path.exists(ck):
            print("  falta " + ck)
            continue
        m = H.ConvTransformer(num_indices=H.NUM_INDICES,
                              seq_length=H.SEQ_LENGTH, img_size=img,
                              num_heads=4, num_layers=2)
        m.load_state_dict(torch.load(ck, map_location="cpu"))
        m = m.to(dev).eval()
        for corte in range(N_IDX + 1):
            AB.COLUMNA_CORTADA = None if corte == 0 else corte - 1
            e = errores_por_frame(m, frames, base_msk, a.lote)
            # de parche a fecha: los 16 parches de una escena no son muestras
            # independientes, la unidad es la fecha
            E[len(usadas), :, :, corte] = np.add.reduceat(
                e, np.arange(0, len(e), por_fecha), axis=0) / por_fecha
        AB.COLUMNA_CORTADA = None
        usadas.append(int(s))
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("  semilla %d lista (%d de %d)" % (s, len(usadas), len(semillas)))

    if not usadas:
        raise SystemExit("ningun checkpoint cargado en " + dir_mod)
    E = E[:len(usadas)]

    # NDVI medio por fecha, en unidades originales y no normalizadas: es el
    # regresor "biologia" con el que se compite contra "exposicion".
    i_ndvi = INDEX_NAMES.index("NDVI") if "NDVI" in INDEX_NAMES else 0
    ndvi = stack[..., i_ndvi].reshape(n_fechas, -1).mean(axis=1)

    # Varianza espacial de cada canal en cada fecha, en unidades del scaler.
    # Es el MSE que sacaria un predictor trivial: el que ignora los otros once
    # indices y devuelve la media espacial del canal que le tapan. Sin esta
    # referencia, un MSE alto no distingue "el modelo falla" de "esa escena es
    # mas heterogenea y cualquiera fallaria". El invierno de un vinedo, con
    # suelo desnudo entre hileras y sombras largas, es un buen candidato a lo
    # segundo, asi que la distincion no es teorica.
    var_esp = X.reshape(n_fechas, -1, X.shape[-1]).var(axis=1)

    os.makedirs(a.salida, exist_ok=True)
    ruta = os.path.join(a.salida, "ERRORES_POR_FECHA" + a.sufijo + ".npz")
    np.savez_compressed(
        ruta, E=E, semillas=np.array(usadas), meses=meses,
        fechas_millis=np.asarray(fechas), ndvi=ndvi.astype(np.float32),
        var_espacial=var_esp.astype(np.float32),
        n_por_fecha=por_fecha, indices=np.array(INDEX_NAMES),
        modelos=dir_mod)
    print("\n  guardado " + ruta)
    print("  E con forma " + str(E.shape)
          + "  (semilla, fecha, canal, corte; corte 0 = sin cortar)")
    base = E[:, :, :, 0].mean(axis=(0, 2))
    print("  MSE base global %.4f, entre fechas de %.4f a %.4f"
          % (base.mean(), base.min(), base.max()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
