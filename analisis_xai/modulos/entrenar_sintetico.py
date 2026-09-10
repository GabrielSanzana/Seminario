#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Entrena el ConvTransformer sobre los datos sinteticos y calcula Datt.

Misma arquitectura, mismo enmascarado, mismo procedimiento de ablacion que
sobre el vinedo. Lo unico que cambia es que ahora la estructura verdadera se
conoce, asi que la matriz Datt que sale de aqui se puede puntuar.

DOS ERRORES DE LA PRIMERA VERSION, Y SU ARREGLO
-----------------------------------------------
1. EL ENMASCARADO POR FAMILIA HACIA INAPRENDIBLE UN CUARTO DEL GRAFO.
   MASCARA_MODO="familia_balanceada" oculta SIEMPRE la familia entera del
   indice que se reconstruye. Sobre el vinedo eso es deliberado: mata el
   atajo colineal entre indices que comparten bandas. Sobre datos sinteticos
   las familias son una particion arbitraria de los doce canales, y una
   arista verdadera entre dos canales de la misma familia queda fuera del
   alcance del modelo por construccion: nunca ve uno con el otro disponible.
   Medido sobre los cinco DAG del experimento, 23 de 95 aristas verdaderas,
   el 24%. La primera version llamo a esto "ruido de configuracion". No lo
   es: es un techo sobre la puntuacion, y explica parte de la diferencia que
   se atribuyo al metodo.

   Arreglo: MASCARA_MODO="aleatoria" con MASCARA_K=1, o sea ocultar solo el
   canal objetivo. El modelo aprende E[X_i | los otros once], cuya estructura
   de dependencia es exactamente el manto de Markov que se quiere recuperar.

2. SE COMPARABA CORTAR UNA ARISTA CONTRA QUITAR UNA VARIABLE.
   Datt corta una arista de la matriz de atencion. LOCO con arboles quita la
   variable entera. No son la misma pregunta: con conexiones residuales y una
   rama convolucional, cortar una arista de atencion deja abiertos otros
   caminos por los que la misma informacion vuelve a entrar.

   El competidor justo del LOCO con arboles no es Datt sino la ablacion de
   ENTRADA hecha con el mismo transformer, que el pipeline ya tenia:

       D[i,j] = MSE(reconstruir i | i,j ocultos) - MSE(reconstruir i | i)

   Aqui se calculan las dos y se puntuan por separado. Si la ablacion de
   entrada del transformer alcanza a la de arboles, entonces la arquitectura
   no era el problema y lo que fallaba era leer la atencion. Si tampoco
   alcanza, el problema si es la arquitectura o su entrenamiento.

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

# Enmascarado neutro para el experimento sintetico: ocultar solo el canal
# objetivo. Se fija antes de construir nada porque _mascara_familias y la
# generacion de mascaras de entrenamiento leen estas globales.
H.MASCARA_MODO = "aleatoria"
H.MASCARA_K = 1
H.ABLACION_CONDICIONADA_FAMILIA = False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimen", default="no_lineal",
                    choices=("lineal", "no_lineal", "espacial", "temporal"))
    ap.add_argument("--semillas", type=int, default=5)
    ap.add_argument("--epocas", type=int, default=H.TOTAL_EPOCHS)
    ap.add_argument("--paciencia", type=int, default=H.PATIENCE)
    ap.add_argument("--lote", type=int, default=64)
    ap.add_argument("--etiqueta", default="",
                    help="sufijo de la carpeta de modelos y del JSON de Datt")
    ap.add_argument("--datos", default="datos_sinteticos")
    # El parche de 13x13 pasa por dos convoluciones de stride 2, asi que al
    # llegar a la atencion quedan 4x4 posiciones. Un laplaciano es alta
    # frecuencia y puede sencillamente no sobrevivir a esa reduccion, lo que
    # explicaria por que el regimen espacial es el unico donde el modelo no
    # aprende nada. Con --parche se contrasta esa hipotesis en vez de
    # suponerla.
    ap.add_argument("--parche", type=int, default=None,
                    help="TOKEN_PARCHE alternativo, p.ej. 26")
    a = ap.parse_args()
    if a.parche:
        H.TOKEN_PARCHE = a.parche
        print("  TOKEN_PARCHE = %d" % a.parche)

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
    # Con el enmascarado neutro la mascara base es la identidad: se oculta
    # solo el canal objetivo. Ya no hay variante "familia" que puntuar porque
    # las familias no se usan.
    msk = torch.eye(N_IDX, dtype=torch.bool, device=dev)

    def _cargar(s):
        ck = os.path.join(destino, "model_seed_" + str(s) + ".pth")
        if not os.path.exists(ck):
            return None
        m = H.ConvTransformer(num_indices=H.NUM_INDICES,
                              seq_length=H.SEQ_LENGTH, img_size=img,
                              num_heads=4, num_layers=2)
        m.load_state_dict(torch.load(ck, map_location="cpu"))
        return m.to(dev).eval()

    salida, off = {}, ~np.eye(N_IDX, dtype=bool)
    for nombre in ("atencion", "entrada"):
        mats = []
        for s in semillas:
            m = _cargar(s)
            if m is None:
                continue
            if nombre == "atencion":
                # cortar UNA arista de la matriz de atencion
                mats.append(AB.matriz_ablacion_atencion(m, frames, msk,
                                                        a.lote))
            else:
                # quitar la VARIABLE de la entrada: el competidor justo del
                # LOCO con arboles, hecho con el mismo transformer
                AB.COLUMNA_CORTADA = None
                _norm, cruda = H.matriz_dependencia_ablacion(
                    m, frames, lote=a.lote, verbose=False)
                mats.append(np.asarray(cruda, dtype=np.float64))
            del m
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if not mats:
            continue
        D = np.mean(mats, axis=0)
        np.fill_diagonal(D, 0.0)
        neg = D[off & (D < 0)]
        ruido = float(np.abs(neg).mean()) if neg.size else float("nan")
        salida[nombre] = dict(Datt=D.tolist(), ruido=ruido, M=len(mats),
                              por_semilla=[x.tolist() for x in mats])
        print("  ablacion de %-9s ruido %.3e  aristas sobre 3x: %d"
              % (nombre, ruido, int((D > 3 * ruido).sum())))

    ruta = os.path.join(a.datos, "DATT_" + a.regimen + a.etiqueta + ".json")
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(salida, f)
    print("  guardado " + ruta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
