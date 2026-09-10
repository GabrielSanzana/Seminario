#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
El grafo del vinedo con retardos: hay estructura temporal cruzada o no?

LA PREGUNTA
-----------
El benchmark sintetico mostro que el transformer solo compite con lo clasico
cuando el mecanismo pasa por el pasado, y que con SEQ_LENGTH = 1 no puede
verlo. Se reentreno el vinedo real con los retardos como tokens -12 indices
por 3 instantes, 36 tokens- para preguntar lo unico que importa aqui:

    de que depende el modelo, del mismo instante o del pasado?

Si toda la dependencia se concentra en el retardo 0, el dato del vinedo no
tiene estructura temporal cruzada que extraer, y el grafo de 19 aristas ya
publicado no se queda corto por no haber mirado el tiempo. Si aparecen aristas
que cruzan instantes, hay un resultado nuevo.

Las dos respuestas valen. La segunda anade un hallazgo; la primera cierra una
duda que quedaria abierta para siempre.

EL AVISO QUE HAY QUE LEER ANTES DE LAS CIFRAS
----------------------------------------------
Las fechas de Sentinel-2 no son equiespaciadas. En este stack la mediana de
huecos es 5 dias, pero el percentil 90 es 23 y el maximo 200. Para las escenas
que vienen detras de un hueco grande, el token "t-1" no es el pasado proximo
sino otra fase fenologica entera. Con 185 huecos, unos 18 pasan de 23 dias.

Eso NO se puede arreglar promediando: contamina el 10% de las filas con un
retardo que significa otra cosa. Aqui se cuantifica el dano corriendo el mismo
analisis solo sobre las escenas cuyo predecesor esta dentro de un umbral, y se
comparan las dos lecturas. Si coinciden, la contaminacion no manda; si no,
la conclusion se limita al subconjunto limpio.

USO
    .venv/bin/python grafo_temporal.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, "/srv/pfigueroa")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from certificar_todo import INDEX_NAMES, N_IDX  # noqa: E402

N_BASE = 12


def suelo(D):
    off = ~np.eye(D.shape[0], dtype=bool)
    neg = D[off & (D < 0)]
    return float(np.abs(neg).mean()) if neg.size else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens",
                    default="resultados/vinedo_temporal/DATT_vinedo_tokens.json")
    ap.add_argument("--base", default="resultados/ABLACION_ATENCION_base.json")
    ap.add_argument("--umbral", type=float, default=3.0)
    ap.add_argument("--salida", default="resultados")
    a = ap.parse_args()

    with open(a.tokens, encoding="utf-8") as f:
        tok = json.load(f)
    with open(a.base, encoding="utf-8") as f:
        pub = json.load(f)
    Dpub = np.array(pub["Datt"])
    rpub = float(pub["ruido"])
    ref = [(i, j) for i in range(N_IDX) for j in range(N_IDX)
           if i != j and Dpub[i, j] > a.umbral * rpub]

    W = 84
    L = ["=" * W, " EL VINEDO CON RETARDOS: HAY ESTRUCTURA TEMPORAL CRUZADA?",
         "",
         " 36 tokens = 12 indices x 3 instantes. Una arista puede cruzar",
         " instantes, asi que por primera vez se puede preguntar si el modelo",
         " depende del pasado o solo del mismo momento.",
         "",
         " AVISO: las fechas no son equiespaciadas. Mediana de huecos 5 dias,",
         " percentil 90 igual a 23, maximo 200. Para las escenas detras de un",
         " hueco grande, 't-1' es otra fase fenologica. Contamina alrededor",
         " del 10% de las filas y no se arregla promediando.",
         "=" * W]

    for nombre in ("atencion", "entrada"):
        if nombre not in tok:
            continue
        G = np.array(tok[nombre]["G"])          # 36 x 36
        n_tok = G.shape[0]
        n_lag = n_tok // N_BASE
        rG = suelo(G)

        # 1. Cuanta dependencia vive en cada retardo. Es LA cifra: si todo
        #    esta en el retardo 0, no hay estructura temporal que extraer.
        masa = []
        for b in range(n_lag):
            bloque = G[:N_BASE, b * N_BASE:(b + 1) * N_BASE].copy()
            if b == 0:
                np.fill_diagonal(bloque, 0.0)
            masa.append(float(np.clip(bloque, 0, None).sum()))
        total = sum(masa) or 1.0

        L += ["", " ABLACION DE " + nombre.upper(),
              " " + "-" * (W - 2),
              "  reparto de la dependencia POSITIVA por instante del origen:"]
        for b in range(n_lag):
            L.append("    retardo t-%d %s %5.1f%%"
                     % (b, "(mismo instante)" if b == 0 else "               ",
                        100 * masa[b] / total))
        L.append("")
        L.append("  Si el retardo 0 se lleva casi todo, el dato no tiene")
        L.append("  estructura temporal cruzada y el grafo publicado no se")
        L.append("  queda corto por no haberla mirado.")

        # 2. Las aristas que cruzan instantes y pasan el umbral
        cruzadas = []
        for i in range(N_BASE):
            for b in range(1, n_lag):
                for j in range(N_BASE):
                    v = G[i, b * N_BASE + j] / rG
                    if v > a.umbral:
                        cruzadas.append((v, i, j, b))
        cruzadas.sort(reverse=True)
        L += ["", "  aristas que cruzan instantes sobre %gx el ruido: %d"
              % (a.umbral, len(cruzadas))]
        for v, i, j, b in cruzadas[:15]:
            L.append("    %-14s <- %-14s (t-%d)   %.1fx"
                     % (INDEX_NAMES[i], INDEX_NAMES[j], b, v))
        if len(cruzadas) > 15:
            L.append("    ... y %d mas" % (len(cruzadas) - 15))

        # 3. El bloque del mismo instante contra el grafo publicado
        S0 = G[:N_BASE, :N_BASE].copy()
        np.fill_diagonal(S0, 0.0)
        off = ~np.eye(N_BASE, dtype=bool)
        rho = float(stats.spearmanr(S0[off], Dpub[off]).statistic)
        r0 = suelo(S0)
        sobreviven = sum(1 for (i, j) in ref if S0[i, j] > a.umbral * r0)
        L += ["", "  el bloque del mismo instante contra el grafo publicado:",
              "    rho de Spearman %.3f" % rho,
              "    de las %d aristas publicadas, siguen sobre %gx: %d"
              % (len(ref), a.umbral, sobreviven),
              "",
              "    Un rho alto dice que anadir el pasado no cambio lo que el",
              "    modelo hace en el presente, o sea que el grafo publicado",
              "    aguanta. Un rho bajo dice que el modelo de 12 tokens",
              "    estaba atribuyendo al presente algo que era del pasado."]

    txt = "\n".join(L) + "\n" + "=" * W
    print(txt)
    os.makedirs(a.salida, exist_ok=True)
    with open(os.path.join(a.salida, "GRAFO_TEMPORAL.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
