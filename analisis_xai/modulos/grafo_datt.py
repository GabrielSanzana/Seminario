#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
El grafo de dependencia verificado por intervencion.

QUE DIBUJA
----------
Las 19 aristas dirigidas cuyo corte en la matriz de atencion sube el error de
reconstruccion por encima de tres veces el nivel de ruido. El nivel de ruido se
estima con la cola negativa de Datt: cortar una arista no puede MEJORAR la
reconstruccion salvo por error de estimacion, asi que la magnitud tipica de los
valores negativos mide el suelo.

    Datt[v,j] = MSE(reconstruir v | A[v,j] = 0) - MSE(reconstruir v)

La flecha va de j a v, o sea del que informa al que necesita. Datt[v,j] alto
significa que reconstruir v se degrada si se le impide mirar a j, y como
Datt[v,j] y Datt[j,v] son numeros distintos, la relacion tiene direccion.

CODIFICACION
------------
  grosor    Datt en multiplos del ruido
  continua  el ICP no rechaza la invariancia entre fases (q > 0.10)
  punteada  el coeficiente cambia con la fase fenologica (q <= 0.10)
  color     bloque biofisico del nodo
  tamano    cuantas veces el nodo aparece como fuente

USO
    .venv/bin/python grafo_datt.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from certificar_todo import INDEX_NAMES, N_IDX  # noqa: E402

BLOQUES = [("verdor", ("NDVI", "EVI", "EVI2", "SAVI", "KNDVI"), "#2e7d32"),
           ("agua", ("NDWI", "NDMI", "NDII"), "#1565c0"),
           ("pigmento", ("MARI", "ARI", "CHL_REDEDGE", "PSRI"), "#c62828")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--umbral", type=float, default=3.0)
    ap.add_argument("--salida", default="resultados")
    a = ap.parse_args()

    with open(os.path.join(a.salida, "ABLACION_ATENCION_base.json"),
              encoding="utf-8") as f:
        d = json.load(f)
    D = np.array(d["Datt"])
    ruido = float(d["ruido"])
    with open(os.path.join(a.salida, "ANALISIS_ICP.json"),
              encoding="utf-8") as f:
        q = {(x["i"], x["j"]): x["q"] for x in json.load(f)["aristas"]}

    aristas = [(i, j, D[i, j] / ruido) for i in range(N_IDX)
               for j in range(N_IDX)
               if i != j and D[i, j] > a.umbral * ruido]
    aristas.sort(key=lambda t: -t[2])

    # Orden de los nodos en la circunferencia: agrupados por bloque, con un
    # hueco entre bloques para que la pertenencia se lea sin leyenda.
    orden, color_nodo = [], {}
    for _nom, miembros, col in BLOQUES:
        for m in miembros:
            orden.append(INDEX_NAMES.index(m))
            color_nodo[INDEX_NAMES.index(m)] = col
    hueco = 0.35
    n = len(orden)
    pasos = []
    ang = np.pi / 2
    for k, idx in enumerate(orden):
        pasos.append(ang)
        salto = 2 * np.pi / (n + hueco * len(BLOQUES))
        cierra_bloque = any(INDEX_NAMES[idx] == b[1][-1] for b in BLOQUES)
        ang -= salto * (1 + hueco if cierra_bloque else 1)
    pos = {idx: (np.cos(t), np.sin(t)) for idx, t in zip(orden, pasos)}

    grados = {i: sum(1 for _v, j, _w in aristas if j == i) for i in range(N_IDX)}

    fig, ax = plt.subplots(figsize=(9.5, 10.6))
    ax.set_aspect("equal")
    ax.axis("off")

    for v, j, w in aristas:
        x0, y0 = pos[j]
        x1, y1 = pos[v]
        inv = q.get((v, j), 0.0) > 0.10
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1),
            connectionstyle="arc3,rad=0.16",
            arrowstyle="-|>", mutation_scale=13 + 0.8 * w,
            linewidth=0.45 * w, alpha=0.72,
            linestyle="-" if inv else (0, (4, 2.5)),
            color="#111111" if inv else "#8a8a8a",
            shrinkA=17, shrinkB=19, zorder=1))

    for i in range(N_IDX):
        x, y = pos[i]
        r = 0.048 + 0.013 * grados[i]
        ax.add_patch(plt.Circle((x, y), r, facecolor=color_nodo[i],
                                edgecolor="white", linewidth=1.6, zorder=3))
        ax.text(x * 1.30, y * 1.26, INDEX_NAMES[i], ha="center", va="center",
                fontsize=10, fontweight="bold", zorder=4,
                color=color_nodo[i])

    ax.set_xlim(-1.62, 1.62)
    ax.set_ylim(-1.95, 1.45)
    ax.set_title("Dependencia entre indices verificada por intervencion\n"
                 "19 aristas con $\\Delta$MSE > 3x el ruido, sobre 15 "
                 "inicializaciones", fontsize=12.5, pad=16)

    leyenda = [
        Line2D([], [], color="#111111", lw=2,
               label="invariante entre fases (ICP, q > 0.10)"),
        Line2D([], [], color="#8a8a8a", lw=2, linestyle=(0, (4, 2.5)),
               label="depende del regimen estacional (q $\\leq$ 0.10)"),
    ] + [Line2D([], [], marker="o", color="w", markerfacecolor=c,
                markersize=11, label=nom) for nom, _m, c in BLOQUES]
    ax.text(0, -1.34, "la flecha va del indice que INFORMA al que lo "
            "NECESITA;  grosor proporcional a $\\Delta$MSE",
            ha="center", va="top", fontsize=9.5, color="#444444")
    ax.legend(handles=leyenda, loc="upper center", ncol=2, frameon=False,
              fontsize=9.5, bbox_to_anchor=(0.5, 0.10))

    fig.tight_layout()
    for ext in ("pdf", "png"):
        ruta = os.path.join(a.salida, "grafo_datt." + ext)
        fig.savefig(ruta, dpi=220, bbox_inches="tight")
        print("  guardado " + ruta)

    print("\n  fuentes (cuantas veces informa):")
    for i in sorted(range(N_IDX), key=lambda k: -grados[k]):
        if grados[i]:
            print("    %-14s %d" % (INDEX_NAMES[i], grados[i]))
    n_inv = sum(1 for v, j, _w in aristas if q.get((v, j), 0.0) > 0.10)
    print("  %d invariantes, %d de regimen" % (n_inv, len(aristas) - n_inv))
    reciprocas = sum(1 for v, j, _w in aristas
                     if any(v2 == j and j2 == v for v2, j2, _ in aristas))
    print("  aristas reciprocas: %d de %d" % (reciprocas, len(aristas)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
