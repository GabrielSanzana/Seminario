#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reentrenar con exposicion balanceada entre fases fenologicas.

QUE CONFOUND ATACA
------------------
datt_por_fase.py encontro que el suelo de ruido de Datt es 5 a 7 veces mayor en
dormancia y postcosecha, y se interpreto como respuesta del modelo a la
ausencia de hoja. Al medir el MSE base por fase aparecio otra explicacion:

    fase           fechas de train    MSE base
    dormancia              8           0.2918
    postcosecha            7           0.1747
    brotacion             17           0.1696
    crecimiento           40           0.1159
    maduracion            76           0.1094

El modelo reconstruye peor donde menos entreno, y el suelo de ruido sigue al
MSE base casi en el mismo orden. Igualar fechas y parches de EVALUACION, que es
lo que hacia datt_por_fase, no toca ese desbalance: la exposicion de
ENTRENAMIENTO va de 8 a 76 fechas.

QUE HACE
--------
Sobremuestrea los frames de entrenamiento para que las cinco fases pesen igual
en la perdida. Cada fase se repite hasta alcanzar el tamano de la mayor (76),
asi que la epoca pasa de 148 a 380 frames y ninguna fecha se descarta.

Se parchea process_indices_data en vez de tocar el DataLoader porque hay dos
caminos de carga -_LoaderVRAM en GPU y DataLoader en CPU- y sobremuestrear
antes vale para los dos sin duplicar codigo.

LO QUE ESTE EXPERIMENTO NO PUEDE CONTESTAR
------------------------------------------
Sobremuestrear iguala el PESO en el gradiente, no la INFORMACION. Las 8 fechas
de dormancia repetidas nueve veces siguen siendo 8 escenas distintas. Asi que:

  si el MSE de dormancia BAJA al nivel del resto, el problema era que el modelo
  apenas la miraba, y la lectura biologica del experimento anterior se cae;

  si SIGUE alto, queda descartado que fuera falta de atencion, pero no se
  distingue entre biologia -sin hoja los indices degeneran- y que 8 escenas no
  alcancen para aprender el regimen. Eso pediria mas fechas, no mas repeticion.

RESULTADO: EL CONFOUND DE EXPOSICION QUEDA DESCARTADO
------------------------------------------------------
Balanceo aplicado: dormancia x10.00, postcosecha x11.25, brotacion x4.09,
crecimiento x1.58, maduracion x1.00. Epoca de 148 a 450 frames, 15 semillas.

Razones contra maduracion, que es la fase con mas fechas de entrenamiento:

    fase           MSE base  ruido | MSE bal  ruido bal
    dormancia         2.67    7.02 |   2.87     6.75
    postcosecha       1.60    4.51 |   1.71     4.03
    crecimiento       1.06    0.94 |   0.96     0.64
    maduracion        1.00    1.00 |   1.00     1.00

Igualar el peso de cada fase en la perdida NO cierra la brecha: dormancia sigue
2.9 veces peor y su suelo de ruido sigue 6.8 veces por encima. La explicacion
"el modelo apenas miraba esa fase" queda descartada.

Lo que no se separa, y estaba declarado antes de correr: biologia -sin hoja los
indices degeneran- frente a que 8 escenas distintas no alcancen. Repetirlas
diez veces iguala el gradiente pero no anade informacion. Distinguirlo pide mas
fechas de invierno, no mas repeticion.

EL BRAZO BALANCEADO ES PEOR MODELO, Y SIRVE IGUAL
-------------------------------------------------
El MSE sube en las cinco fases, entre 2.3 y 2.6 veces. Con dormancia repetida
diez veces y postcosecha once, la diversidad efectiva por paso de gradiente
baja y el modelo generaliza peor. Asi que este brazo no es una estimacion mejor
de nada: es un CONTROL, y como control cumple, porque las razones entre fases
se mantienen.

DOS RESULTADOS DEL BRAZO BASE QUE ESTE CONTROL TUMBA
-----------------------------------------------------
1. La arista ARI <- KNDVI en brotacion daba Datt = -6.0 veces el ruido, o sea
   que cortarla mejoraba la reconstruccion, y se habia propuesto como "ruta mal
   aprendida". En el brazo balanceado da +5.6. Cambia de signo, asi que no es
   un defecto estable del modelo sino un efecto del reparto de fechas de ese
   entrenamiento. Se retira.

2. La prueba contra las etiquetas del ICP empeora: invariantes 1.57 fases
   contra 2.17 de las de regimen, p = 0.8364. En el brazo base era 3.00 contra
   2.83 con p = 0.4134. En ninguno de los dos hay apoyo, y en el balanceado la
   diferencia va en el sentido contrario al esperado.

FALLO DEL PRIMER CONTROL, Y SU ARREGLO
--------------------------------------
La primera version comparaba contra el brazo base dejando la particion
cronologica, y ahi el bloque de validacion tiene 17 fechas de crecimiento y 14
de maduracion contra 1 de dormancia y 1 de postcosecha. El early stopping y la
seleccion de checkpoint optimizan ese reparto, asi que un brazo que gasta un
quinto de su gradiente en dormancia queda penalizado por un criterio donde esa
fase casi no aparece. El control estaba sesgado en contra de lo que medía.

Arreglo: particion train/val tomando el ultimo 20% DENTRO de cada fase
(--estratificado) y sobremuestreo tambien del val (--balancear-val), de modo
que el criterio de parada pese igual las cinco. Y un brazo de control con el
mismo split y el mismo criterio pero sin sobremuestrear el train
(--sin-balancear-train), para que la unica diferencia sea esa.

Cuatro brazos, razones del MSE base contra maduracion:

    brazo                  dormancia  postcosecha   suelo dormancia
    cronologico  base         2.67       1.60            7.02
    cronologico  bal          2.87       1.71            6.75
    estratificado ctrl        2.06       1.48            3.48
    estratificado bal         2.46       1.54            4.93

La comparacion limpia es la del par estratificado, que solo difiere en el
sobremuestreo del train: dormancia pasa de 2.06 a 2.46 y su suelo de 3.48 a
4.93. Balancear la exposicion no cierra la brecha, la ensancha un poco.

Lo que si cambio el arreglo: la brecha es MENOR de lo que decia el primer
control. Dormancia es 2.1 veces peor que maduracion, no 2.7, y su suelo de
ruido 3.5 veces mayor, no 7. Parte del efecto original era la particion
cronologica. El resto se mantiene.

La arista ARI <- KNDVI en brotacion da -6.0 solo en el brazo cronologico
original y +5.6, +9.0 y +4.4 en los otros tres. Queda confirmado que era
artefacto de esa particion.

La prueba contra el ICP no encuentra apoyo en ninguno de los cuatro brazos, y
en tres de ellos las invariantes aguantan MENOS fases que las de regimen
(p entre 0.41 y 0.92).

USO
    srun --partition=student --qos=student --gres=gpu:1 --mem=16G \\
         .venv/bin/python -u entrenar_balanceado.py --semillas 15
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/srv/pfigueroa")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hello as H  # noqa: E402
from analisis_fenologico import FASES  # noqa: E402

_ORIG_PID = H.process_indices_data
MESES = None
BALANCEAR = False           # sobremuestrear el TRAIN
ESTRATIFICADO = False       # particion train/val por fase en vez de cronologica
BALANCEAR_VAL = False       # sobremuestrear el VAL, para que el criterio de
                            # early stopping pese igual las cinco fases
REPORTE = {}


def _indices_balanceados(meses_tr: np.ndarray) -> np.ndarray:
    """Indices de frames de train repetidos para igualar el peso de cada fase."""
    grupos = []
    for nombre, ms in FASES:
        idx = np.where(np.isin(meses_tr, ms))[0]
        if len(idx):
            grupos.append((nombre, idx))
    objetivo = max(len(i) for _n, i in grupos)
    salida, rep = [], {}
    for nombre, idx in grupos:
        veces = int(np.ceil(objetivo / len(idx)))
        ext = np.tile(idx, veces)[:objetivo]
        salida.append(ext)
        rep[nombre] = dict(originales=int(len(idx)), tras_balanceo=int(len(ext)),
                           factor=round(objetivo / len(idx), 2))
    REPORTE.update(rep)
    return np.concatenate(salida)


def _split_estratificado(meses_all, frac_val=0.2):
    """Particion train/val tomando el ultimo frac_val de CADA fase.

    La particion original es cronologica: el ultimo 20% de fechas va a val. Con
    eso el bloque de validacion queda con 17 fechas de crecimiento y 14 de
    maduracion contra 1 de dormancia y 1 de postcosecha, asi que el early
    stopping y la seleccion de checkpoint optimizan casi solo la estacion de
    crecimiento. Un brazo que reparte gradiente entre las cinco fases queda
    penalizado por un criterio donde dos de ellas casi no aparecen, que es
    justo lo que el experimento quiere medir.

    Se toma el ultimo tramo DENTRO de cada fase y no una muestra al azar: las
    fechas vecinas se parecen mucho, y con muestreo aleatorio una fecha de train
    y su vecina de val serian casi la misma escena.
    """
    tr, va = [], []
    for _nombre, ms in FASES:
        idx = np.where(np.isin(meses_all, ms))[0]
        if not len(idx):
            continue
        corte = max(1, int(round((1 - frac_val) * len(idx))))
        tr.append(idx[:corte])
        va.append(idx[corte:] if corte < len(idx) else idx[-1:])
    return np.sort(np.concatenate(tr)), np.sort(np.concatenate(va))


def _pid_balanceado(stack, seq_length=1, dates_millis=None, **kw):
    out = _ORIG_PID(stack, seq_length=seq_length, dates_millis=dates_millis,
                    **kw)
    (Xtr, Ytr), (Xva, Yva), img, scaler = out
    if MESES is None or not (BALANCEAR or ESTRATIFICADO or BALANCEAR_VAL):
        return out

    if ESTRATIFICADO:
        # Se recompone el orden original -el cronologico- y se reparte por
        # fase. El scaler sigue siendo el que ajusto la particion cronologica;
        # es el MISMO para los dos brazos, asi que la comparacion es limpia,
        # pero conviene declararlo.
        X = np.concatenate([Xtr, Xva], axis=0)
        Y = (np.concatenate([Ytr, Yva], axis=0) if Ytr is not None else None)
        i_tr, i_va = _split_estratificado(MESES[:len(X)])
        Xtr, Xva = X[i_tr], X[i_va]
        Ytr = Y[i_tr] if Y is not None else None
        Yva = Y[i_va] if Y is not None else None
        meses_tr, meses_va = MESES[i_tr], MESES[i_va]
    else:
        meses_tr = MESES[:len(Xtr)]
        meses_va = MESES[len(Xtr):len(Xtr) + len(Xva)]

    if BALANCEAR:
        idx = _indices_balanceados(meses_tr)
        Xtr = Xtr[idx]
        Ytr = Ytr[idx] if Ytr is not None else Ytr
    if BALANCEAR_VAL:
        # Sin esto el criterio de parada pesa las fases como vengan en el
        # bloque de validacion, y ese reparto es justo el confound.
        idx = _indices_balanceados(meses_va)
        Xva = Xva[idx]
        Yva = Yva[idx] if Yva is not None else Yva
    return (Xtr, Ytr), (Xva, Yva), img, scaler


def main() -> int:
    global MESES, BALANCEAR, ESTRATIFICADO, BALANCEAR_VAL
    ap = argparse.ArgumentParser()
    ap.add_argument("--semillas", type=int, default=15)
    ap.add_argument("--epocas", type=int, default=H.TOTAL_EPOCHS)
    ap.add_argument("--destino", default="resultados/fase_balanceada")
    ap.add_argument("--estratificado", action="store_true",
                    help="particion train/val por fase en vez de cronologica")
    ap.add_argument("--balancear-val", action="store_true",
                    dest="balancear_val",
                    help="sobremuestrear tambien el val, para que el early "
                         "stopping pese igual las cinco fases")
    ap.add_argument("--sin-balancear-train", action="store_true",
                    dest="sin_train",
                    help="brazo de control: mismo split y mismo criterio, sin "
                         "sobremuestrear el entrenamiento")
    a = ap.parse_args()

    os.makedirs(a.destino, exist_ok=True)
    stack = np.load(H.OUTPUT_NPY)
    fechas = H.cargar_fechas_stack(len(stack))
    MESES = np.array([dt.datetime.utcfromtimestamp(int(m) / 1000).month
                      for m in fechas])
    H.process_indices_data = _pid_balanceado
    ESTRATIFICADO = a.estratificado
    BALANCEAR_VAL = a.balancear_val
    print("  split: " + ("estratificado por fase" if ESTRATIFICADO
                         else "cronologico"))
    print("  val: " + ("balanceado" if BALANCEAR_VAL else "como venga"))
    print("  train: " + ("sin balancear" if a.sin_train else "balanceado"))

    semillas = list(H.SEEDS)[:a.semillas]
    print("  semillas: " + str(semillas))

    for s in semillas:
        ck = os.path.join(a.destino, "model_seed_" + str(s) + ".pth")
        if os.path.exists(ck):
            print("  semilla " + str(s) + " ya estaba")
            continue
        t0 = time.time()
        H.set_seed(s)
        BALANCEAR = not a.sin_train
        try:
            _m, _v, _h, best = H.train_convtransformer(
                stack, seq_length=H.SEQ_LENGTH, total_epochs=a.epocas,
                batch_size=H.BATCH_SIZE, lr=H.LR_CONVTRANSFORMER,
                patience=H.PATIENCE, num_workers=0, ckpt_path=ck,
                dates_millis=fechas)
        finally:
            BALANCEAR = False
        with open(os.path.join(a.destino, "val_" + str(s) + ".json"), "w") as f:
            json.dump({str(s): float(best)}, f)
        print("  semilla %d lista en %.1f min  val=%.6f"
              % (s, (time.time() - t0) / 60, best))

    vl = {}
    for nom in os.listdir(a.destino):
        if nom.startswith("val_") and nom.endswith(".json"):
            with open(os.path.join(a.destino, nom)) as f:
                vl.update(json.load(f))
    with open(os.path.join(a.destino, "val_losses.json"), "w") as f:
        json.dump(vl, f, indent=2)
    with open(os.path.join(a.destino, "balanceo.json"), "w",
              encoding="utf-8") as f:
        json.dump(REPORTE, f, indent=2, ensure_ascii=False)
    print("\n  balanceo aplicado:")
    for k, v in REPORTE.items():
        print("    %-24s %3d -> %3d  (x%.2f)"
              % (k, v["originales"], v["tras_balanceo"], v["factor"]))
    print("  val_losses.json con " + str(len(vl)) + " semillas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
