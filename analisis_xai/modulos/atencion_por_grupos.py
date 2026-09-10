#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
La atencion evaluada donde el teorema de imposibilidad permite que funcione.

EL ERROR DE TODAS LAS EVALUACIONES ANTERIORES
---------------------------------------------
Todo el proyecto evaluo las fuentes sobre los 66 pares a la vez. Eso mezcla dos
regimenes que un resultado teorico separa:

  "The Attribution Impossibility: No Feature Ranking Is Faithful, Stable, and
  Complete Under Collinearity" (arXiv 2605.21492) prueba que, con variables
  colineales, ninguna asignacion de importancia puede ser a la vez fiel,
  estable entre semillas y completa. Y da la salida: agregar sobre modelos
  entrenados de forma independiente SACRIFICANDO la completitud DENTRO de cada
  grupo colineal, o sea declarando empates entre variables simetricas y
  ordenando solo ENTRE grupos.

Los doce indices de este caso tienen correlacion media r = 0.888. Los pares
intra-grupo son, por tanto, irrankeables por construccion, y meterlos en la
evaluacion mete ruido garantizado que hunde cualquier correlacion global. Todos
los "la atencion no correlaciona con nada" de este proyecto se midieron con
esos pares dentro.

Este modulo repite las evaluaciones separando los dos regimenes. La prediccion
es concreta y falsable: la atencion debe funcionar bastante mejor ENTRE grupos
que DENTRO, y si no lo hace, entonces el problema no era la colinealidad.

TRES GRUPOS DE PRUEBAS
----------------------
1. PARTICION DEL RANKING. Spearman contra la verdad, calculado por separado
   sobre los pares intra-grupo y los inter-grupo, con nulo de permutacion que
   respeta el subconjunto. Dos agrupaciones: la espectral (que bandas comparten)
   y la biofisica (que magnitud miden).

2. ESTABILIDAD POR MITADES. Se parten las 50 semillas en dos mitades al azar,
   se calcula el ranking en cada una y se correlacionan. Cien repeticiones. Es
   la medida directa de la "estabilidad" del teorema, y tiene que subir al
   restringir a inter-grupo si la explicacion es la colinealidad.

3. RESOLUCION DE GRUPO. Se colapsa la matriz 12x12 a una k x k promediando las
   entradas de cada bloque, y se compara contra la verdad colapsada igual. El
   teorema dice que a nivel de grupo la atribucion SI es identificable, asi que
   aqui es donde la atencion tiene su mejor oportunidad.

RESULTADO: LA PREDICCION SALE AL REVES, Y ESO ES LA NOTICIA
-----------------------------------------------------------
La prediccion era que la atencion funcionara mejor ENTRE grupos. Sale lo
contrario: entre grupos no hay nada en ninguna fuente (todos los q por encima
de 0.32), y todo lo que hay esta DENTRO de los grupos colineales.

Test de signos sobre 50 semillas, dentro de las familias espectrales, contra
|pc| (14 pares):

    atencion capa A1            -0.459    50/50 negativas   q = 3.6e-15
    atencion capa A2            -0.407    50/50 negativas   q = 3.6e-15
    atencion A1+A2              -0.433    50/50 negativas   q = 3.6e-15
    atencion max sobre cabezas  -0.354    50/50 negativas   q = 3.6e-15
    atencion A2*A1              -0.327    47/50 negativas   q = 6.6e-11
    dependencia por ablacion    +0.613     0/50 negativas   q = 3.6e-15

Las cinco lecturas de atencion coinciden en signo NEGATIVO y la ablacion en
POSITIVO, cada una en las 50 semillas. Por eso ninguna evaluacion anterior vio
nada: promediar los dos regimenes suma una senal negativa dentro de grupo con
ruido fuera de grupo y da cero.

QUE SIGNIFICA EL SIGNO NEGATIVO
-------------------------------
Dentro de una familia los indices son casi duplicados. La ablacion, con la
familia oculta como referencia, mide cuanto ayuda DEVOLVER a un hermano: cuanto
mas dependen entre si, mas ayuda, y sale positivo. La atencion hace lo
contrario: cuanto mas sustituible es j por otro hermano, mas reparte el peso, y
cuanto mas conditionalmente dependiente es el par, MENOS peso especifico le
hace falta. El peso de atencion mide no-sustituibilidad al margen, que dentro
de un conjunto redundante es lo inverso de la dependencia condicional.

Encaja con lo ya medido: a nivel de nodo la atencion pesa redundancia
(rho +0.853 contra la correlacion marginal media) y a nivel de par, dentro de
familia, la anti-pesa.

LO QUE HAY QUE RETIRAR DE LA MOTIVACION DE ARRIBA
-------------------------------------------------
El teorema de imposibilidad NO explica lo que pasa aqui. Su mecanismo es la
inestabilidad entre modelos, y la estabilidad por mitades sale entre 0.92 y
0.99 en TODOS los casos, dentro y fuera de grupo. Las semillas coinciden. El
ranking no es inestable: es estable y no coincide con |pc| cuando se lo mide
mezclando regimenes. La cita sirve para situar el problema de la colinealidad,
no para explicar este resultado, y hay que declararlo asi.

DOS CAUTELAS
------------
Las familias espectrales son exactamente lo que oculta la mascara de
entrenamiento (familia_balanceada), asi que ese regimen es el que el modelo vio
por diseno. Lo que la mascara NO le da es el ORDEN de dependencia dentro de la
familia, que es lo que aqui se recupera; aun asi conviene declararlo.

En la agrupacion biofisica, que la mascara no usa, el rollout da +0.482 con
46/50 semillas y max sobre cabezas da -0.612 con 45/50: dos lecturas de la
MISMA atencion con signos opuestos sobre la misma particion. Eso es una
inconsistencia sin resolver, no un resultado limpio.

USO
    .venv/bin/python atencion_por_grupos.py --repeticiones 200
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from certificar_todo import (  # noqa: E402
    CANDIDATOS_MATRICES, CANDIDATOS_PARADIGMAS, CANDIDATOS_STACK,
    INDEX_FAMILIES, INDEX_NAMES, N_IDX, PARES, _buscar, fuerza_por_par,
    pc_y_r, recolectar_fuentes,
)
from analisis_atencion import OFF, descomponer_aridad, par_simetrico  # noqa: E402

# Agrupacion biofisica: que magnitud mide cada indice. Es distinta de la
# espectral y no la usa la mascara de entrenamiento, asi que recuperar
# estructura respecto de ella no es un control positivo del canal.
BLOQUES_BIO = {
    "verdor": ("NDVI", "EVI", "EVI2", "SAVI", "KNDVI"),
    "agua": ("NDWI", "NDMI", "NDII"),
    "pigmento": ("MARI", "ARI", "CHL_REDEDGE", "PSRI"),
}


def etiquetas(agrupacion: Dict[str, Tuple[str, ...]]) -> np.ndarray:
    g = np.full(N_IDX, -1, dtype=int)
    for k, (_n, miembros) in enumerate(agrupacion.items()):
        for m in miembros:
            if m in INDEX_NAMES:
                g[INDEX_NAMES.index(m)] = k
    for i in range(N_IDX):
        if g[i] < 0:
            g[i] = 100 + i
    return g


def mascara_pares(g: np.ndarray) -> np.ndarray:
    """True en los pares INTRA-grupo, de los 66 no dirigidos."""
    return np.array([g[i] == g[j] for i, j in PARES])


def _rho(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 4:
        return float("nan")
    return float(stats.spearmanr(x, y).statistic)


def _p_perm_indices(v: np.ndarray, ref: np.ndarray, sub: np.ndarray,
                    n_perm: int, rng) -> float:
    """Nulo por permutacion de las 12 etiquetas de indice, evaluado en el
    subconjunto `sub`.

    Permutar los valores de los pares sueltos seria mas facil pero rompe que
    los 66 pares compartan nodos; permutar los indices lo respeta. El
    subconjunto se recalcula en cada permutacion, asi que el nulo tambien
    incorpora que la particion intra/inter cambia al barajar.
    """
    M = np.zeros((N_IDX, N_IDX))
    for k, (i, j) in enumerate(PARES):
        M[i, j] = M[j, i] = v[k]
    obs = abs(_rho(v[sub], ref[sub]))
    if not np.isfinite(obs):
        return float("nan")
    cuenta = 0
    for _ in range(n_perm):
        o = rng.permutation(N_IDX)
        P = M[np.ix_(o, o)]
        w = np.array([P[i, j] for i, j in PARES])
        r = abs(_rho(w[sub], ref[sub]))
        if np.isfinite(r) and r >= obs:
            cuenta += 1
    return float((cuenta + 1) / (n_perm + 1))


def estabilidad_por_mitades(mats: List[np.ndarray], sub: np.ndarray,
                            n_rep: int, rng) -> Dict:
    """Se parten las semillas en dos mitades y se correlacionan los rankings.

    Es la definicion operativa de "estable" del teorema: si dos conjuntos
    independientes de modelos ordenan igual los pares, el ranking es
    reproducible; si no, la fuente no tiene un ranking propio que reportar.
    """
    F = np.stack([fuerza_por_par(m) for m in mats])
    M = len(F)
    vals = []
    for _ in range(n_rep):
        o = rng.permutation(M)
        a, b = o[: M // 2], o[M // 2:]
        va = F[a].mean(axis=0)[sub]
        vb = F[b].mean(axis=0)[sub]
        vals.append(_rho(va, vb))
    vals = np.array([v for v in vals if np.isfinite(v)])
    return dict(media=float(vals.mean()) if vals.size else float("nan"),
                p05=float(np.percentile(vals, 5)) if vals.size else float("nan"),
                n=int(sub.sum()))


def por_semilla(mats, ref, sub):
    """Correlacion dentro del subconjunto, semilla a semilla, con test de signos.

    Es la prueba de alta potencia. El p de permutacion de etiquetas se calcula
    sobre 14 o 19 pares y tiene poca resolucion; que 45 de 50 modelos
    entrenados por separado coincidan en el SIGNO es una afirmacion distinta y
    mucho mas dificil de conseguir por azar. Las dos preguntas son legitimas y
    ninguna sustituye a la otra: una pregunta si el ranking difiere de barajar
    indices, la otra si el efecto se reproduce entre inicializaciones.
    """
    rs = [_rho(fuerza_por_par(m)[sub], ref[sub]) for m in mats]
    rs = [r for r in rs if np.isfinite(r)]
    if len(rs) < 5:
        return dict(media=float("nan"), n_neg=0, n=0, p_signos=float("nan"))
    n_neg = sum(1 for r in rs if r < 0)
    n_pos = len(rs) - n_neg
    # Bilateral: interesa que coincidan en un signo, cualquiera de los dos.
    k = max(n_neg, n_pos)
    return dict(media=float(np.mean(rs)), n_neg=n_neg, n=len(rs),
                p_signos=float(stats.binomtest(k, len(rs), 0.5,
                                               alternative="greater").pvalue))


def bh(ps):
    """Benjamini-Hochberg sobre la tabla completa de contrastes."""
    ps = np.asarray(ps, dtype=float)
    ok = np.where(np.isfinite(ps))[0]
    q = np.full(len(ps), np.nan)
    if not len(ok):
        return q
    v = ps[ok]
    o = np.argsort(v)
    m = len(v)
    aj = np.minimum.accumulate((v[o] * m / (np.arange(m) + 1))[::-1])[::-1]
    out = np.empty(m)
    out[o] = np.minimum(aj, 1.0)
    q[ok] = out
    return q


def cargar_datt(salida: str) -> List[Tuple[str, List[np.ndarray]]]:
    """Las matrices de ablacion de ARISTAS de atencion, por semilla.

    Entran como una fuente mas para poder pasarlas por la misma particion. Son
    15 semillas y no 50, asi que el test de signos tiene menos potencia: con
    15/15 el p es 6.1e-05, que sigue bastando, pero no es comparable en fuerza
    con los 50/50 de las fuentes observacionales.
    """
    out = []
    for nom, etq in (("ablacion_atencion_base.npy",
                      "Datt cortar arista (base)"),
                     ("ablacion_atencion_alea.npy",
                      "Datt cortar arista (mask alea)")):
        r = os.path.join(salida, nom)
        if os.path.exists(r):
            m = np.load(r)
            out.append((etq, [m[i] for i in range(len(m))]))
    return out


def colapsar(A: np.ndarray, g: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """Matriz k x k promediando las entradas fuera de diagonal de cada bloque.

    La celda (a,b) con a != b promedia todas las A[i,j] con i en a y j en b. La
    diagonal de bloque promedia las A[i,j] con i,j en el mismo grupo e i != j, y
    queda como dato descriptivo: no entra en la comparacion porque es justo el
    regimen que el teorema declara irrankeable.
    """
    ks = sorted(set(g.tolist()))
    K = len(ks)
    C = np.zeros((K, K))
    for a, ga in enumerate(ks):
        for b, gb in enumerate(ks):
            sel = [(i, j) for i in range(N_IDX) for j in range(N_IDX)
                   if i != j and g[i] == ga and g[j] == gb]
            C[a, b] = np.mean([A[i, j] for i, j in sel]) if sel else np.nan
    return C, ks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--permutaciones", type=int, default=2000)
    ap.add_argument("--repeticiones", type=int, default=200)
    ap.add_argument("--salida", default="resultados")
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    stack = np.load(_buscar(CANDIDATOS_STACK, False))
    PC, RR = pc_y_r(stack)
    pc_pares = np.array([PC[i, j] for i, j in PARES])
    fuentes = [(e, p) for e, _n, _A, _s, p in recolectar_fuentes(
        _buscar(CANDIDATOS_MATRICES, True),
        _buscar(CANDIDATOS_PARADIGMAS, True), PC, RR) if p and len(p) >= 5]

    # |beta| del ICP como segunda verdad. Es mejor referencia que |pc| en un
    # sentido concreto: sale de estimar el coeficiente en CADA fecha y no de
    # agregar todas, asi que no promedia los cinco regimenes fenologicos.
    beta_pares = None
    ruta = os.path.join(a.salida, "ANALISIS_ICP.json")
    if os.path.exists(ruta):
        with open(ruta, encoding="utf-8") as f:
            ar = {(x["i"], x["j"]): x for x in json.load(f)["aristas"]}
        beta_pares = np.array([max(abs(ar[(i, j)]["beta_medio"]),
                                   abs(ar[(j, i)]["beta_medio"]))
                               for i, j in PARES])
        # Solo los pares INVARIANTES del ICP: la estructura del cultivo que no
        # cambia con la fenologia. Es la verdad mas exigente que hay aqui.
        inv = np.array([(ar[(i, j)]["q"] > 0.10) and (ar[(j, i)]["q"] > 0.10)
                        for i, j in PARES])
    else:
        inv = None

    verdades = {"|pc|": pc_pares}
    if beta_pares is not None:
        verdades["|beta| ICP"] = beta_pares

    fuentes = fuentes + cargar_datt(a.salida)
    print("  fuentes: " + str(len(fuentes)) + "   verdades: "
          + str(list(verdades)))

    agrup = {"espectral": etiquetas(INDEX_FAMILIES),
             "biofisica": etiquetas(BLOQUES_BIO)}

    filas = []
    for vnom, ref in verdades.items():
        for nombre, g in agrup.items():
            dentro = mascara_pares(g)
            for etiqueta, mats in fuentes:
                A = np.mean(mats, axis=0)
                v = fuerza_por_par(A)
                f = dict(verdad=vnom, agrupacion=nombre, fuente=etiqueta,
                         M=len(mats), n_dentro=int(dentro.sum()),
                         n_fuera=int((~dentro).sum()))
                for tag, sub in (("dentro", dentro), ("fuera", ~dentro)):
                    f["rho_" + tag] = _rho(v[sub], ref[sub])
                    f["p_" + tag] = _p_perm_indices(v, ref, sub,
                                                    a.permutaciones, rng)
                    ps = por_semilla(mats, ref, sub)
                    f["sem_media_" + tag] = ps["media"]
                    f["sem_neg_" + tag] = ps["n_neg"]
                    f["sem_n_" + tag] = ps["n"]
                    f["sem_p_" + tag] = ps["p_signos"]
                    f["estab_" + tag] = estabilidad_por_mitades(
                        mats, sub, a.repeticiones, rng)["media"]
                filas.append(f)

    # Multiplicidad sobre TODA la tabla, que ahora son verdades x agrupaciones
    # x fuentes x (dentro, fuera).
    qs = bh([f["p_dentro"] for f in filas] + [f["p_fuera"] for f in filas])
    qs2 = bh([f["sem_p_dentro"] for f in filas]
             + [f["sem_p_fuera"] for f in filas])
    for i, f in enumerate(filas):
        f["q_dentro"] = float(qs[i])
        f["q_fuera"] = float(qs[len(filas) + i])
        f["sem_q_dentro"] = float(qs2[i])
        f["sem_q_fuera"] = float(qs2[len(filas) + i])

    # Resolucion de grupo.
    grupo = []
    for nombre, g in agrup.items():
        Cpc, ks = colapsar(PC, g)
        fuera_diag = ~np.eye(len(ks), dtype=bool)
        for etiqueta, mats in fuentes:
            C, _ = colapsar(np.mean(mats, axis=0), g)
            x = C[fuera_diag]
            y = Cpc[fuera_diag]
            ok = np.isfinite(x) & np.isfinite(y)
            d = dict(agrupacion=nombre, fuente=etiqueta, k=len(ks),
                     n_celdas=int(ok.sum()),
                     rho_pc=_rho(x[ok], y[ok]))
            if beta_pares is not None:
                Bm = np.zeros((N_IDX, N_IDX))
                with open(ruta, encoding="utf-8") as fh:
                    for z in json.load(fh)["aristas"]:
                        Bm[z["i"], z["j"]] = abs(z["beta_medio"])
                Cb, _ = colapsar(Bm, g)
                yb = Cb[fuera_diag]
                ok2 = np.isfinite(x) & np.isfinite(yb)
                d["rho_beta"] = _rho(x[ok2], yb[ok2])
            grupo.append(d)

    W = 78
    L = ["=" * W, " LA ATENCION DONDE EL TEOREMA PERMITE QUE FUNCIONE", "",
         " Con variables colineales, ninguna atribucion puede ser fiel, estable",
         " y completa a la vez (arXiv 2605.21492). La salida que propone el",
         " propio teorema es agregar sobre modelos independientes y renunciar a",
         " ordenar DENTRO de cada grupo colineal.",
         "",
         " Estos indices tienen r = 0.888 de correlacion media. Todas las",
         " evaluaciones anteriores de este proyecto metieron los pares",
         " intra-grupo, que son irrankeables por construccion, en la misma",
         " correlacion global que los inter-grupo.",
         "=" * W, ""]
    for vnom in verdades:
        for nombre in agrup:
            sub = [f for f in filas if f["verdad"] == vnom
                   and f["agrupacion"] == nombre]
            if not sub:
                continue
            L.append("  CONTRA " + vnom + "   ---   agrupacion "
                     + nombre.upper() + "   (" + str(sub[0]["n_dentro"])
                     + " pares dentro, " + str(sub[0]["n_fuera"]) + " fuera)")
            L.append("")
            L.append("  " + "fuente".ljust(30) + "M".rjust(4)
                     + "rho DENTRO".rjust(11) + "signo".rjust(9)
                     + "q signos".rjust(10) + "rho fuera".rjust(11)
                     + "q signos".rjust(10))
            for f in sub:
                neg, n = f["sem_neg_dentro"], f["sem_n_dentro"]
                signo = ("%d-/%d" % (neg, n)) if neg > n / 2 else \
                        ("%d+/%d" % (n - neg, n))
                L.append("  " + f["fuente"][:29].ljust(30)
                         + str(f["M"]).rjust(4)
                         + ("%+.3f" % f["rho_dentro"]).rjust(11)
                         + signo.rjust(9)
                         + ("%.2g" % f["sem_q_dentro"]).rjust(10)
                         + ("%+.3f" % f["rho_fuera"]).rjust(11)
                         + ("%.2g" % f["sem_q_fuera"]).rjust(10))
            L.append("")
    L.append("=" * W)
    L.append("  RESOLUCION DE GRUPO: la matriz colapsada a k x k")
    L.append("")
    L.append("  " + "agrupacion".ljust(12) + "fuente".ljust(30)
             + "celdas".rjust(8) + "rho |pc|".rjust(10) + "rho |beta|".rjust(12))
    for d in grupo:
        L.append("  " + d["agrupacion"].ljust(12) + d["fuente"][:29].ljust(30)
                 + str(d["n_celdas"]).rjust(8)
                 + ("%+.3f" % d["rho_pc"]).rjust(10)
                 + ("%+.3f" % d.get("rho_beta", float("nan"))).rjust(12))
    L.append("")
    L.append("  A nivel de grupo la atribucion SI es identificable segun el")
    L.append("  teorema, asi que esta es la mejor oportunidad de la atencion.")
    L.append("  Pocas celdas, asi que un rho alto aqui vale poco por si solo:")
    L.append("  hay que leerlo junto con la estabilidad de arriba.")
    L.append("=" * W)
    txt = "\n".join(L)
    print("\n" + txt)
    os.makedirs(a.salida, exist_ok=True)
    with open(os.path.join(a.salida, "ATENCION_POR_GRUPOS.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt)
    with open(os.path.join(a.salida, "ATENCION_POR_GRUPOS.json"), "w",
              encoding="utf-8") as f:
        json.dump(dict(pares=filas, grupos=grupo), f, indent=2,
                  ensure_ascii=False)
    print("\n  Guardado: "
          + os.path.join(a.salida, "ATENCION_POR_GRUPOS.txt"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
