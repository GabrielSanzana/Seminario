#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analisis por fase fenologica, con el cultivo como verdad externa.

DE DONDE SALE LA VERDAD EXTERNA
-------------------------------
El catastro de viveros CIREN 2024 (viveros_2024_h19.shp, 3185 poligonos) situa
un vivero de VID VINIFERA de 41 ha en Pencahue, region del Maule, a 0.00 km del
centro del recorte de este caso de estudio. Las coordenadas del stack
(-71.830319, -35.450253) convertidas a UTM 19S dan 243117, 6073342, y el
recorte de 52x52 pixeles de 10 m cubre 27 ha dentro de ese poligono.

Eso resuelve la limitacion que las conclusiones de la tesis declaran como
insalvable: que para indices espectrales de Sentinel-2 no existe una referencia
externa contra la cual comparar. Existe, y dice que el cultivo es vid.

QUE CAMBIA SABER QUE ES VID
---------------------------
La vid es un cultivo perenne, caducifolio y conducido en hileras, con suelo
desnudo entre ellas. En el hemisferio sur su ciclo es:

    dormancia        mayo a agosto      sin hoja, la senal es suelo y sarmiento
    brotacion        septiembre-octubre  aparece area foliar
    crecimiento      noviembre-diciembre  dosel cerrado, maxima actividad
    maduracion       enero-marzo         envero, la hoja empieza a senescer
    postcosecha      abril               caida de hoja

Esto obliga a reinterpretar dos resultados anteriores:

  1. Se atribuyo a NUBOSIDAD invernal que las diez escenas de mayor residuo
     cayeran siete de diez entre mayo y septiembre. En vid, mayo a agosto es
     DORMANCIA: la planta no tiene hojas y todos los indices de vegetacion
     colapsan por biologia, no por nubes. Las dos causas se confunden en el
     mismo periodo y el analisis anterior no podia separarlas.

  2. El 39.8% de pixeles con NDVI > 0.3 y el 57.3% clasificados como suelo
     desnudo dejan de ser un reparto arbitrario: son la firma geometrica
     esperable de un vinedo en hileras, donde el suelo entre hileras es
     permanente y no una anomalia.

QUE HACE ESTE MODULO
--------------------
Calcula la estructura de dependencia de los datos POR FASE y pregunta a cual de
ellas se parece la matriz que el modelo aprendio. El modelo se entreno sobre
todas las fechas a la vez, asi que su matriz es un promedio de regimenes; si se
parece mucho mas a una fase concreta, esa fase domina lo aprendido, y eso es
informacion sobre que capturo el modelo que ninguna metrica agregada da.

USO
    .venv/bin/python analisis_fenologico.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from certificar_todo import (  # noqa: E402
    CANDIDATOS_MATRICES, CANDIDATOS_PARADIGMAS, CANDIDATOS_STACK, INDEX_NAMES,
    N_IDX, PARES, VAL_RATIO, _buscar, dependencia_no_lineal,
    direccion_heterocedastica, fuerza_por_par, pc_y_r, puestos,
    recolectar_fuentes,
)

# Ciclo de la vid en el hemisferio sur. Los cortes son los estados
# fenologicos estandar del cultivo, no una particion elegida por conveniencia:
# cambiarlos cambiaria el significado de cada grupo, no solo su tamano.
FASES: List[Tuple[str, Tuple[int, ...]]] = [
    ("dormancia (may-ago)", (5, 6, 7, 8)),
    ("brotacion (sep-oct)", (9, 10)),
    ("crecimiento (nov-dic)", (11, 12)),
    ("maduracion (ene-mar)", (1, 2, 3)),
    ("postcosecha (abr)", (4,)),
]

# Verdad externa del catastro CIREN 2024, para dejarla en la salida.
CULTIVO = dict(fuente="CIREN viveros_2024_h19.shp",
               especie="Vid Vinifera", comuna="Pencahue", region="Maule",
               superficie_ha=41.05, distancia_km=0.0,
               utm19s=(243117, 6073342))


def meses_del_stack(fechas_millis: np.ndarray, n: int) -> np.ndarray:
    return np.array([dt.datetime.utcfromtimestamp(int(m) / 1000).month
                     for m in fechas_millis[:n]])


def estructura_de(stack: np.ndarray) -> Dict[str, np.ndarray]:
    """|pc|, |r| y direccion heterocedastica de un subconjunto de fechas."""
    PC, RR = pc_y_r(stack)
    pc = np.array([PC[i, j] for i, j in PARES])
    rr = np.array([RR[i, j] for i, j in PARES])
    D = direccion_heterocedastica(stack, n_muestra=3000, semilla=0,
                                  centrar_escena=True)
    d = np.array([D[i, j] for i, j in PARES])
    return dict(pc=pc, rr=rr, dir=d)


def main() -> int:
    ap = argparse.ArgumentParser(description="Analisis por fase fenologica.")
    ap.add_argument("--stack", default=None)
    ap.add_argument("--matrices", default=None)
    ap.add_argument("--paradigmas", default=None)
    ap.add_argument("--metadata", default=None)
    ap.add_argument("--salida", default="resultados")
    a = ap.parse_args()

    stack_p = a.stack or _buscar(CANDIDATOS_STACK, False)
    stack = np.load(stack_p)
    meta_p = a.metadata or os.path.join(os.path.dirname(stack_p),
                                        "metadata.npz")
    fechas = np.load(meta_p, allow_pickle=True)["dates_millis"]
    n_train = int((1.0 - VAL_RATIO) * len(stack))
    meses = meses_del_stack(fechas, n_train)
    print(f"  cultivo (CIREN): {CULTIVO['especie']}, {CULTIVO['comuna']}, "
          f"{CULTIVO['superficie_ha']} ha, a {CULTIVO['distancia_km']} km")
    print(f"  fechas de entrenamiento: {n_train}")

    # Estructura global y por fase.
    glob = estructura_de(stack)
    por_fase = {}
    for nombre, ms in FASES:
        sel = np.where(np.isin(meses, ms))[0]
        print(f"  {nombre:<24} {len(sel):>3} fechas")
        if len(sel) < 8:
            print("    muy pocas fechas, se salta")
            continue
        por_fase[nombre] = dict(n=len(sel), **estructura_de(stack[sel]))

    # A que fase se parece cada fuente aprendida.
    PC, RR = pc_y_r(stack)
    dirm = a.matrices or _buscar(CANDIDATOS_MATRICES, True)
    dirp = a.paradigmas or _buscar(CANDIDATOS_PARADIGMAS, True)
    fuentes = [(e, np.mean(p, axis=0)) for e, _n, _A, _s, p
               in recolectar_fuentes(dirm, dirp, PC, RR) if p and len(p) >= 5]

    filas = []
    for etiqueta, A in fuentes:
        fr = fuerza_por_par(A)
        fila = dict(etiqueta=etiqueta,
                    rho_global=float(stats.spearmanr(fr, glob["pc"]).statistic))
        for nombre, e in por_fase.items():
            fila[nombre] = float(stats.spearmanr(fr, e["pc"]).statistic)
        filas.append(fila)

    W = 78
    L = ["=" * W, " ANALISIS POR FASE FENOLOGICA", "",
         f" Verdad externa: {CULTIVO['fuente']}",
         f" Cultivo {CULTIVO['especie']} en {CULTIVO['comuna']}, "
         f"{CULTIVO['superficie_ha']} ha, a {CULTIVO['distancia_km']} km del",
         f" centro del recorte (UTM19S {CULTIVO['utm19s'][0]}, "
         f"{CULTIVO['utm19s'][1]}).",
         "",
         " Las conclusiones de la tesis declaran que para indices Sentinel-2 no",
         " existe referencia externa. El catastro CIREN es una: dice que cultivo",
         " hay en el pixel.",
         "=" * W, ""]
    L.append("  Cuanto cambia la ESTRUCTURA DE LOS DATOS entre fases")
    L.append(f"  {'fase':<24}{'fechas':>7}   Spearman de |pc| contra la fase")
    nombres = list(por_fase)
    L.append("  " + " " * 31 + "".join(f"{n[:9]:>11}" for n in nombres))
    for na in nombres:
        fila = f"  {na:<24}{por_fase[na]['n']:>7}   "
        for nb in nombres:
            fila += f"{stats.spearmanr(por_fase[na]['pc'], por_fase[nb]['pc']).statistic:>11.3f}"
        L.append(fila)
    L.append("")
    L.append("  Si estas correlaciones son altas, la dependencia entre indices")
    L.append("  no cambia con la fenologia y agregar todas las fechas no pierde")
    L.append("  nada. Si son bajas, el modelo entreno sobre regimenes distintos")
    L.append("  mezclados y su matriz es un promedio de cosas incompatibles.")
    L.append("")
    L.append("  A que fase se parece lo que APRENDIO cada fuente")
    L.append(f"  {'fuente':<30}{'global':>8}"
             + "".join(f"{n[:9]:>11}" for n in nombres))
    for f in filas:
        L.append(f"  {f['etiqueta']:<30}{f['rho_global']:>+8.3f}"
                 + "".join(f"{f.get(n, float('nan')):>+11.3f}"
                           for n in nombres))
    L.append("")
    L.append("  Una fuente que se parezca mucho mas a una fase concreta")
    L.append("  aprendio ese regimen y no el promedio, aunque se entreno con")
    L.append("  todas las fechas.")
    L.append("=" * W)
    txt = "\n".join(L)
    print("\n" + txt)

    os.makedirs(a.salida, exist_ok=True)
    with open(os.path.join(a.salida, "ANALISIS_FENOLOGICO.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt)
    with open(os.path.join(a.salida, "ANALISIS_FENOLOGICO.json"), "w",
              encoding="utf-8") as f:
        json.dump(dict(cultivo=CULTIVO, fuentes=filas,
                       fases={k: dict(n=v["n"]) for k, v in por_fase.items()}),
                  f, indent=2, ensure_ascii=False)
    print(f"\n  Guardado: {os.path.join(a.salida, 'ANALISIS_FENOLOGICO.txt')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
