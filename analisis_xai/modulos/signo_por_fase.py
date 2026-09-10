#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
El signo intra-familia, contra una referencia que no mezcla regimenes.

EL PROBLEMA QUE CIERRA
----------------------
atencion_por_grupos.py encontro que dentro de las familias espectrales las
cinco lecturas de atencion correlacionan NEGATIVO con |pc| en 47 a 50 de las 50
semillas, y la ablacion de entrada POSITIVO en 50 de 50. Señal reproducible y
grande. Pero al repetirlo contra el |beta| del ICP los signos se invierten.

La causa esta medida: en los 14 pares intra-familia, |pc| y |beta| correlacionan
solo +0.209 (p = 0.47). Y el motivo tambien: 11 de esos 14 pares son "de
regimen" segun el ICP, o sea que su coeficiente cambia entre fases fenologicas.
|pc| se estima agrupando las 148 fechas, asi que para esos pares promedia cinco
regimenes incompatibles y no es un blanco estacionario. Solo 3 de los 14 son
invariantes, que no dan para nada.

O sea que el signo medido no es una propiedad de la atencion sino de que
referencia se elija, y con las dos referencias disponibles la pregunta no tiene
respuesta.

LA SALIDA
---------
Estimar |pc| DENTRO de cada fase por separado. Dentro de una fase no hay mezcla
de regimenes por construccion, asi que el blanco si es estacionario. Si el
signo de la correlacion intra-familia sale igual en las cinco fases, la
contaminacion queda descartada y el signo es de la atencion. Si cambia de fase
en fase, entonces no hay un signo que reportar y hay que decirlo.

Es una replica interna con cinco muestras independientes de fechas, que es mas
fuerte que cualquier p sobre una sola estimacion agrupada.

RESULTADO: EL SIGNO SE SOSTIENE
-------------------------------
Postcosecha se cae por tener 7 fechas, asi que quedan cuatro fases. Dentro de
las familias espectrales, TODAS las fuentes conservan su signo en las cuatro:

    atencion capa A1      -0.39  -0.26  -0.42  -0.40    4/4
    atencion capa A2      -0.35  -0.26  -0.35  -0.32    4/4
    atencion A2*A1        -0.35  -0.27  -0.27  -0.23    4/4
    atencion A1+A2        -0.36  -0.26  -0.36  -0.35    4/4
    atencion max cabezas  -0.29  -0.20  -0.29  -0.27    4/4
    Datt cortar arista    -0.41  -0.41  -0.36  -0.29    4/4
    dependencia ablacion  +0.54  +0.42  +0.48  +0.61    4/4
    dependencia cruda     +0.53  +0.48  +0.32  +0.34    4/4

Con |pc| estimado dentro de cada fase no hay mezcla de regimenes por
construccion, asi que la contaminacion del blanco agrupado queda descartada: el
signo es de la fuente. Cuatro lecturas de atencion en negativo y las dos
ablaciones de entrada en positivo, cada una en las cuatro fases y en 47 a 50 de
las 50 semillas.

La excepcion es el rollout, que da 0/4: cambia de signo entre fases y por tanto
no tiene señal intra-familia que reportar. Encaja con que sea la lectura mas
contaminada, con su 0.25*I y su renormalizacion por fila.

QUE NO RESUELVE
---------------
Contra el |beta| del ICP los signos salen invertidos, y esta replica no explica
por que. Lo que si establece es que la inversion no viene de promediar
regimenes, porque el signo aguanta fase por fase. La diferencia queda entre dos
estimadores de dependencia -|pc| normalizado por los dos extremos frente a
|beta| como maximo de dos cocientes con cola pesada, de 0.002 a 9.677- y sigue
abierta.

USO
    .venv/bin/python signo_por_fase.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from typing import Dict, List

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from certificar_todo import (  # noqa: E402
    CANDIDATOS_MATRICES, CANDIDATOS_PARADIGMAS, CANDIDATOS_STACK,
    INDEX_FAMILIES, INDEX_NAMES, N_IDX, PARES, VAL_RATIO, _buscar,
    fuerza_por_par, pc_y_r, recolectar_fuentes,
)
from analisis_fenologico import FASES  # noqa: E402
from atencion_por_grupos import (  # noqa: E402
    BLOQUES_BIO, cargar_datt, etiquetas, mascara_pares,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-fechas", type=int, default=8, dest="min_fechas")
    ap.add_argument("--salida", default="resultados")
    a = ap.parse_args()

    stack_p = _buscar(CANDIDATOS_STACK, False)
    stack = np.load(stack_p)
    fechas = np.load(os.path.join(os.path.dirname(stack_p), "metadata.npz"),
                     allow_pickle=True)["dates_millis"]
    n_train = int((1.0 - VAL_RATIO) * len(stack))
    meses = np.array([dt.datetime.utcfromtimestamp(int(m) / 1000).month
                      for m in fechas[:n_train]])

    PC, RR = pc_y_r(stack)
    refs = {"global (todas las fechas)":
            np.array([PC[i, j] for i, j in PARES])}
    for nombre, ms in FASES:
        sel = np.where(np.isin(meses, ms))[0]
        if len(sel) < a.min_fechas:
            print("  fuera por pocas fechas: " + nombre + " ("
                  + str(len(sel)) + ")")
            continue
        P, _ = pc_y_r(stack[sel])
        refs[nombre] = np.array([P[i, j] for i, j in PARES])
    print("  referencias: " + str(list(refs)))

    fuentes = [(e, p) for e, _n, _A, _s, p in recolectar_fuentes(
        _buscar(CANDIDATOS_MATRICES, True),
        _buscar(CANDIDATOS_PARADIGMAS, True), PC, RR) if p and len(p) >= 5]
    fuentes = fuentes + cargar_datt(a.salida)

    agrup = {"espectral": etiquetas(INDEX_FAMILIES),
             "biofisica": etiquetas(BLOQUES_BIO)}

    filas = []
    for gnom, g in agrup.items():
        dentro = mascara_pares(g)
        for etiqueta, mats in fuentes:
            v = fuerza_por_par(np.mean(mats, axis=0))
            f = dict(agrupacion=gnom, fuente=etiqueta, M=len(mats))
            signos = []
            for rnom, ref in refs.items():
                r = float(stats.spearmanr(v[dentro], ref[dentro]).statistic)
                f["rho_" + rnom] = r
                # Test de signos entre semillas para esta fase.
                rs = [float(stats.spearmanr(fuerza_por_par(m)[dentro],
                                            ref[dentro]).statistic)
                      for m in mats]
                rs = [x for x in rs if np.isfinite(x)]
                k = max(sum(1 for x in rs if x < 0),
                        sum(1 for x in rs if x > 0))
                f["p_" + rnom] = float(stats.binomtest(
                    k, len(rs), 0.5, alternative="greater").pvalue)
                if rnom != "global (todas las fechas)":
                    signos.append(np.sign(r))
            f["fases_mismo_signo"] = (int(abs(sum(signos)))
                                      if signos else 0)
            f["n_fases"] = len(signos)
            filas.append(f)

    W = 78
    L = ["=" * W, " EL SIGNO INTRA-FAMILIA, FASE POR FASE", "",
         " |pc| agrupado sobre las 148 fechas no es blanco estacionario para",
         " los pares intra-familia: 11 de esos 14 son 'de regimen' segun el",
         " ICP y su coeficiente cambia entre fases. Aqui se re-estima |pc|",
         " DENTRO de cada fase, donde por construccion no hay mezcla.",
         "",
         " La columna que decide es la ultima: en cuantas de las fases el",
         " signo coincide. Si es 5 de 5, el signo es de la fuente. Si no, no",
         " hay un signo que reportar.",
         "=" * W, ""]
    nombres = [r for r in refs if r != "global (todas las fechas)"]
    for gnom in agrup:
        sub = [f for f in filas if f["agrupacion"] == gnom]
        L.append("  AGRUPACION " + gnom.upper())
        L.append("  " + "fuente".ljust(28) + "global".rjust(9)
                 + "".join(n[:9].rjust(11) for n in nombres)
                 + "acuerdo".rjust(9))
        for f in sub:
            fila = ("  " + f["fuente"][:27].ljust(28)
                    + ("%+.2f" % f["rho_global (todas las fechas)"]).rjust(9))
            for n in nombres:
                fila += ("%+.2f" % f["rho_" + n]).rjust(11)
            fila += (str(f["fases_mismo_signo"]) + "/"
                     + str(f["n_fases"])).rjust(9)
            L.append(fila)
        L.append("")
    L.append("  'acuerdo' = cuantas fases dan el mismo signo, de las que hay.")
    L.append("=" * W)
    txt = "\n".join(L)
    print("\n" + txt)
    os.makedirs(a.salida, exist_ok=True)
    with open(os.path.join(a.salida, "SIGNO_POR_FASE.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt)
    with open(os.path.join(a.salida, "SIGNO_POR_FASE.json"), "w",
              encoding="utf-8") as f:
        json.dump(filas, f, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
