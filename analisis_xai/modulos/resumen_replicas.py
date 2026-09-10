#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Agrega las replicas del benchmark de recuperacion.

POR QUE HACE FALTA AGREGAR
--------------------------
Con un solo DAG de 19 aristas, la exactitud de orientacion se estima con 19
ensayos: el error estandar bajo la hipotesis nula es 0.115. O sea que 0.63 y
0.37 caben los dos dentro del azar, y las dos lecturas opuestas -"orienta" y
"orienta al reves"- serian igual de defendibles. Declarar cualquiera de las
dos con ese n seria el mismo error que este trabajo lleva corrigiendo desde el
principio.

Con cuatro DAG independientes son 76 ensayos y el error estandar baja a 0.057.
Sigue sin ser mucho, pero ya distingue 0.5 de 0.7.

QUE SE AGREGA Y COMO
--------------------
Las areas bajo la curva se promedian entre replicas. La orientacion NO se
promedia: se suman aciertos y ensayos y se hace un solo test binomial sobre el
total, que es lo correcto cuando cada replica aporta un numero distinto de
aristas verdaderas.

USO
    .venv/bin/python resumen_replicas.py --dirs datos_rep11 datos_rep12 ...
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from scipy import stats

METODOS = ["correlacion", "correlacion parcial", "PC (Fisher z)",
           "informacion mutua", "distancia de corr.", "LOCO con arboles",
           "LOCO con vecindario", "Datt identidad largo",
           "Datt familia largo"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*", default=None)
    ap.add_argument("--salida", default="resultados")
    a = ap.parse_args()
    dirs = a.dirs or sorted(glob.glob("datos_rep*"))

    W = 84
    L = ["=" * W, " RECUPERACION DE LA ESTRUCTURA VERDADERA, AGREGADO SOBRE "
         + str(len(dirs)) + " DAG", "",
         " Cada replica es un DAG distinto de 19 aristas sobre 12 nodos, con",
         " sus propios mecanismos y su propio entrenamiento de 5 semillas.",
         "",
         " La orientacion se agrega sumando aciertos y ensayos, no promediando",
         " proporciones. Azar 0.5. Con un solo DAG el error estandar es 0.115",
         " y no se puede afirmar nada; aqui baja a 0.057.",
         "=" * W]

    todo = {}
    for reg in ("lineal", "no_lineal", "espacial"):
        filas = {}
        for m in METODOS:
            aucs, mors, pks, ac, tot, n_rep = [], [], [], 0, 0, 0
            for d in dirs:
                f = os.path.join(d, "BENCHMARK_RECUPERACION.json")
                if not os.path.exists(f):
                    continue
                j = json.load(open(f, encoding="utf-8"))
                if reg not in j or m not in j[reg]:
                    continue
                r = j[reg][m]
                aucs.append(r["auc_esqueleto"])
                mors.append(r["auc_moral"])
                pks.append(r["precision_en_k"])
                n_rep += 1
                if r.get("orientacion") is not None:
                    n = r["n_aristas_orientadas"]
                    ac += int(round(r["orientacion"] * n))
                    tot += n
            if not aucs:
                continue
            p = (float(stats.binomtest(ac, tot, 0.5,
                                       alternative="greater").pvalue)
                 if tot else None)
            filas[m] = dict(auc_esqueleto=float(np.mean(aucs)),
                            sd_auc=float(np.std(aucs, ddof=1))
                            if len(aucs) > 1 else 0.0,
                            auc_moral=float(np.mean(mors)),
                            precision_en_k=float(np.mean(pks)),
                            orientacion=(ac / tot) if tot else None,
                            n_orientadas=tot, p_orientacion=p,
                            n_replicas=n_rep)
        if not filas:
            continue
        todo[reg] = filas
        L += ["", " REGIMEN " + reg.upper().replace("_", " ")
              + "   (%d replicas)" % max(f["n_replicas"]
                                         for f in filas.values()),
              " " + "-" * (W - 2),
              "  " + "metodo".ljust(24) + "auc esq".rjust(9) + "sd".rjust(7)
              + "auc moral".rjust(11) + "prec@k".rjust(8)
              + "orientacion".rjust(13) + "n".rjust(5) + "p".rjust(9)]
        for m, f in filas.items():
            L.append("  " + m[:23].ljust(24)
                     + ("%.3f" % f["auc_esqueleto"]).rjust(9)
                     + ("%.3f" % f["sd_auc"]).rjust(7)
                     + ("%.3f" % f["auc_moral"]).rjust(11)
                     + ("%.2f" % f["precision_en_k"]).rjust(8)
                     + (("%.3f" % f["orientacion"]) if f["orientacion"]
                        is not None else "simetrico").rjust(13)
                     + (str(f["n_orientadas"]) if f["n_orientadas"]
                        else "-").rjust(5)
                     + (("%.4f" % f["p_orientacion"])
                        if f["p_orientacion"] is not None else "-").rjust(9))
    L.append("=" * W)
    txt = "\n".join(L)
    print(txt)
    os.makedirs(a.salida, exist_ok=True)
    with open(os.path.join(a.salida, "RECUPERACION_REPLICAS.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt)
    with open(os.path.join(a.salida, "RECUPERACION_REPLICAS.json"), "w",
              encoding="utf-8") as f:
        json.dump(todo, f, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
