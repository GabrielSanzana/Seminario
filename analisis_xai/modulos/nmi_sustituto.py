#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
El acoplamiento NMI = 1.0 de la atencion, es tambien un artefacto unario?

La descomposicion por aridad mostro que la topologia, la curvatura y el grafo
Top-P de la atencion los reproduce un sustituto A_sur = mu + r_i + c_j que no
contiene ninguna relacion de par. Falta el ultimo descriptor de esa lista: el
acoplamiento perfecto entre aristas (NMI = 1.0, phi = +1.0) que se reporto
dentro del grupo G1 y se interpreto como "el modelo trata ese bloque como una
unidad indivisible".

Hay un motivo mecanico para sospechar que tambien es unario. Si la matriz de
una semilla es casi mu + r_i + c_j, entonces todas las aristas que apuntan al
mismo destino j comparten el mismo c_j y suben o bajan juntas de semilla en
semilla. Al discretizar con Top-P entran o salen del grafo a la vez, y su NMI
sale 1 sin que exista ninguna relacion entre esas dos aristas.

Este script contesta la pregunta con el mismo procedimiento de
analisis_sustitucion.py aplicado dos veces: a las matrices reales y a sus
sustitutos unarios semilla a semilla.

USO
    .venv/bin/python nmi_sustituto.py --top-p 0.15
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from certificar_todo import (  # noqa: E402
    CANDIDATOS_MATRICES, CANDIDATOS_PARADIGMAS, CANDIDATOS_STACK,
    _buscar, familia_de_indice, pc_y_r, recolectar_fuentes,
)
from analisis_atencion import sustituto_unario  # noqa: E402
from analisis_sustitucion import (  # noqa: E402
    analizar_sustitucion, nmi_binaria, presencia_por_semilla,
)


def acoplamientos_perfectos(mats, top_p):
    """Cuantos pares de aristas alcanzan NMI = 1, y cuantos comparten destino.

    Compartir destino es la firma del artefacto: dos aristas i->j y k->j se
    mueven juntas si lo que las mueve es c_j y no una relacion entre ellas.
    """
    P, dirigidas = presencia_por_semilla(mats, top_p)
    M, E = P.shape
    var = [k for k in range(E) if 0 < P[:, k].sum() < M]
    n_perf = n_destino = n_origen = 0
    for a in range(len(var)):
        for b in range(a + 1, len(var)):
            ka, kb = var[a], var[b]
            if nmi_binaria(P[:, ka], P[:, kb]) >= 0.999:
                n_perf += 1
                if dirigidas[ka][1] == dirigidas[kb][1]:
                    n_destino += 1
                if dirigidas[ka][0] == dirigidas[kb][0]:
                    n_origen += 1
    return dict(n_aristas_variables=len(var), n_pares_nmi_1=n_perf,
                n_mismo_destino=n_destino, n_mismo_origen=n_origen,
                frac_mismo_destino=(n_destino / n_perf) if n_perf else
                float("nan"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-p", type=float, default=0.15, dest="top_p")
    ap.add_argument("--permutaciones", type=int, default=300)
    ap.add_argument("--salida", default="resultados")
    a = ap.parse_args()

    stack = np.load(_buscar(CANDIDATOS_STACK, False))
    PC, RR = pc_y_r(stack)
    fuentes = [(e, p) for e, _n, _A, _s, p in recolectar_fuentes(
        _buscar(CANDIDATOS_MATRICES, True),
        _buscar(CANDIDATOS_PARADIGMAS, True), PC, RR) if p and len(p) >= 5]
    fam = familia_de_indice()
    rng = np.random.default_rng(0)

    filas = []
    for etiqueta, mats in fuentes:
        sur = [sustituto_unario(m) for m in mats]
        f = dict(etiqueta=etiqueta, M=len(mats))
        for et, ms in (("real", mats), ("sustituto", sur)):
            r = analizar_sustitucion(ms, a.top_p, fam, a.permutaciones, rng)
            p = acoplamientos_perfectos(ms, a.top_p)
            f[et] = dict(nmi_media=r.get("nmi_media", float("nan")),
                         n_sustitutivas=r.get("n_sustitutivas", 0),
                         n_complementarias=r.get("n_complementarias", 0),
                         **p)
        filas.append(f)
        print("  " + etiqueta + " listo")

    W = 78
    L = ["=" * W, " ES UNARIO TAMBIEN EL ACOPLAMIENTO NMI = 1?", "",
         " A_sur = mu + r_i + c_j no contiene ninguna relacion de par. Si aun",
         " asi produce los mismos acoplamientos perfectos, el hallazgo de",
         " 'bloque indivisible' en G1 era un efecto de la normalizacion por",
         " fila y no una propiedad aprendida.",
         "",
         " La columna que lo delata es 'mismo destino': dos aristas i->j y k->j",
         " comparten c_j, asi que entran y salen del Top-P a la vez sin que",
         " exista relacion alguna entre ellas.",
         "=" * W, ""]
    L.append("  " + "fuente".ljust(26) + "NMI media".rjust(11)
             + "pares NMI=1".rjust(13) + "mismo dest".rjust(12)
             + "n compl.".rjust(10) + "n sust.".rjust(9))
    for f in filas:
        for et in ("real", "sustituto"):
            d = f[et]
            nom = (f["etiqueta"][:19] + "  " + et) if et == "real" else \
                  (" " * 19 + "  " + et)
            L.append("  " + nom[:25].ljust(26)
                     + ("%.3f" % d["nmi_media"]).rjust(11)
                     + str(d["n_pares_nmi_1"]).rjust(13)
                     + str(d["n_mismo_destino"]).rjust(12)
                     + str(d["n_complementarias"]).rjust(10)
                     + str(d["n_sustitutivas"]).rjust(9))
        L.append("")
    L.append("=" * W)
    txt = "\n".join(L)
    print("\n" + txt)
    os.makedirs(a.salida, exist_ok=True)
    with open(os.path.join(a.salida, "NMI_SUSTITUTO.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(txt)
    with open(os.path.join(a.salida, "NMI_SUSTITUTO.json"), "w",
              encoding="utf-8") as fh:
        json.dump(filas, fh, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
