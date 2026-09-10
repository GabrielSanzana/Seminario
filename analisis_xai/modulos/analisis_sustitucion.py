#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Metricas de la seccion 13 del informe: sustitucion, reciprocidad, estabilidad
por nodo y persistencia. INDEPENDIENTE del pipeline.

POR QUE ESTAS Y NO OTRAS
------------------------
El informe propone cuatro metricas adicionales (§13). Dos ya estan en
analisis_grafo.py (modularidad del grafo de consenso, PageRank de consenso).
Las otras dos faltaban, y una de ellas ataca justo el hallazgo central de esta
corrida:

  NMI ENTRE ARISTAS  Mide si dos aristas aparecen y desaparecen de forma
      coordinada a lo largo de las M semillas. El informe dice que sirve para
      "caracterizar la estructura de alternativas del modelo", y eso es
      exactamente el diagnostico que falta: se midio que la atencion falla
      porque con canales colineales el peso NO esta identificado -repartirlo
      entre NDVI y SAVI da el mismo error-, pero no se habia medido si el
      modelo efectivamente ELIGE entre ellos. Si NDVI-X y SAVI-X son
      sustitutivas, aparecen en semillas complementarias y su phi es negativa.
      Aqui las 50 semillas dejan de ser ruido a promediar y pasan a ser el
      diseno experimental.

  RECIPROCIDAD  Fraccion de aristas cuya reciproca tambien esta. Alta indica
      relaciones simetricas (lo tipico de una correlacion), baja indica
      direccionalidad. Es un resumen de una linea de algo que costo tres
      modulos medir con la corriente neta de Markov.

  OVERLAP@K POR NODO (§6.2)  El Jaccard global ya se midio; su version por
      nodo dice QUE indices tienen vecindario estable y cuales no, que es un
      diagnostico y no un numero agregado.

  PERSISTENCIA Phi(e) CON TEST BINOMIAL (§7)  El propio informe advierte que
      Phi por si sola no distingue una arista recurrente por azar de una
      recurrente por estructura, y exige el test binomial. Se implementa con
      la correccion por multiplicidad que el informe no menciona.

CORRECCION AL INFORME. El texto de §13 dice "Si NMI(e1, e2) es alta y negativa,
las aristas son sustitutivas". La informacion mutua es no negativa por
definicion: I(X;Y) >= 0 siempre. Lo que distingue sustitutivas de
complementarias no es el signo de la NMI sino el de la correlacion phi entre
las dos variables binarias. Aqui se reportan las dos cosas: NMI para la fuerza
del acoplamiento y phi para su sentido.

USO
    .venv/bin/python analisis_sustitucion.py --top-p 0.15
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from certificar_todo import (  # noqa: E402
    CANDIDATOS_MATRICES, CANDIDATOS_PARADIGMAS, CANDIDATOS_STACK, INDEX_NAMES,
    N_IDX, N_PARES, PARES, _buscar, familia_de_indice, pc_y_r,
    recolectar_fuentes,
)


def presencia_por_semilla(mats: List[np.ndarray], top_p: float
                          ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """(M, n_aristas) booleana: que aristas dirigidas elige cada semilla.

    Se discretiza cada matriz por separado, que es el punto: el consenso se
    construye DESPUES, y lo que interesa aqui es la variabilidad entre semillas,
    no suprimirla.
    """
    dirigidas = [(i, j) for i in range(N_IDX) for j in range(N_IDX) if i != j]
    P = np.zeros((len(mats), len(dirigidas)), dtype=bool)
    for m, A in enumerate(mats):
        v = np.array([abs(A[i, j]) for i, j in dirigidas])
        corte = np.quantile(v, 1 - top_p)
        P[m] = v >= corte
    return P, dirigidas


def nmi_binaria(x: np.ndarray, y: np.ndarray) -> float:
    """Informacion mutua normalizada entre dos variables binarias."""
    n = len(x)
    if n == 0:
        return 0.0
    conj = np.zeros((2, 2))
    for a in (0, 1):
        for b in (0, 1):
            conj[a, b] = np.sum((x == a) & (y == b)) / n
    px, py = conj.sum(1), conj.sum(0)

    def H(p):
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    hx, hy = H(px), H(py)
    if hx <= 0 or hy <= 0:
        return 0.0
    mi = 0.0
    for a in (0, 1):
        for b in (0, 1):
            if conj[a, b] > 0 and px[a] > 0 and py[b] > 0:
                mi += conj[a, b] * np.log(conj[a, b] / (px[a] * py[b]))
    return float(max(mi, 0.0) / np.sqrt(hx * hy))


def analizar_sustitucion(mats: List[np.ndarray], top_p: float,
                         fam: np.ndarray, n_perm: int,
                         rng: np.random.Generator) -> Dict:
    P, dirigidas = presencia_por_semilla(mats, top_p)
    M, E = P.shape
    variables = [k for k in range(E) if 0 < P[:, k].sum() < M]
    if len(variables) < 4:
        return dict(n_aristas_variables=len(variables))

    pares_sust, pares_comp = [], []
    nmis = []
    for a in range(len(variables)):
        for b in range(a + 1, len(variables)):
            ka, kb = variables[a], variables[b]
            x, y = P[:, ka], P[:, kb]
            nmi = nmi_binaria(x, y)
            phi = float(stats.pearsonr(x.astype(float),
                                       y.astype(float)).statistic)
            nmis.append(nmi)
            reg = dict(
                e1=f"{INDEX_NAMES[dirigidas[ka][0]]}->{INDEX_NAMES[dirigidas[ka][1]]}",
                e2=f"{INDEX_NAMES[dirigidas[kb][0]]}->{INDEX_NAMES[dirigidas[kb][1]]}",
                nmi=nmi, phi=phi,
                misma_familia=bool(
                    fam[dirigidas[ka][1]] == fam[dirigidas[kb][1]]),
                mismo_origen=bool(dirigidas[ka][0] == dirigidas[kb][0]))
            (pares_sust if phi < 0 else pares_comp).append(reg)

    # Nulo: permutar las semillas de una de las dos series rompe el
    # acoplamiento y deja las frecuencias marginales intactas.
    nulo = []
    for _ in range(min(n_perm, 300)):
        a, b = rng.choice(variables, 2, replace=False)
        nulo.append(nmi_binaria(P[:, a], rng.permutation(P[:, b])))
    nulo = np.array(nulo)
    umbral = float(np.percentile(nulo, 99))

    pares_sust.sort(key=lambda r: -r["nmi"])
    pares_comp.sort(key=lambda r: -r["nmi"])
    sust_sig = [r for r in pares_sust if r["nmi"] > umbral]
    comp_sig = [r for r in pares_comp if r["nmi"] > umbral]
    return dict(
        n_aristas_variables=len(variables),
        nmi_media=float(np.mean(nmis)), nmi_umbral_nulo=umbral,
        n_sustitutivas=len(sust_sig), n_complementarias=len(comp_sig),
        frac_sust_mismo_origen=(
            float(np.mean([r["mismo_origen"] for r in sust_sig]))
            if sust_sig else float("nan")),
        frac_sust_misma_familia=(
            float(np.mean([r["misma_familia"] for r in sust_sig]))
            if sust_sig else float("nan")),
        top_sustitutivas=sust_sig[:8], top_complementarias=comp_sig[:5])


def reciprocidad(A: np.ndarray, top_p: float) -> float:
    """Fraccion de aristas dirigidas cuya reciproca tambien esta en el grafo."""
    dirigidas = [(i, j) for i in range(N_IDX) for j in range(N_IDX) if i != j]
    v = np.array([abs(A[i, j]) for i, j in dirigidas])
    corte = np.quantile(v, 1 - top_p)
    S = {dirigidas[k] for k in range(len(dirigidas)) if v[k] >= corte}
    if not S:
        return float("nan")
    return float(sum(1 for (i, j) in S if (j, i) in S) / len(S))


def overlap_por_nodo(mats: List[np.ndarray], K: int = 3) -> Dict:
    """Fraccion del Top-K de cada fila que se conserva entre dos semillas."""
    tops = []
    for A in mats:
        fila = []
        for i in range(N_IDX):
            v = np.abs(A[i]).copy()
            v[i] = -np.inf
            fila.append(set(np.argsort(-v)[:K].tolist()))
        tops.append(fila)
    por_nodo = np.zeros(N_IDX)
    n = 0
    for a in range(len(tops)):
        for b in range(a + 1, len(tops)):
            for i in range(N_IDX):
                por_nodo[i] += len(tops[a][i] & tops[b][i]) / K
            n += 1
    por_nodo /= max(n, 1)
    orden = np.argsort(-por_nodo)
    return dict(overlap_medio=float(por_nodo.mean()),
                por_nodo={INDEX_NAMES[i]: float(por_nodo[i])
                          for i in range(N_IDX)},
                mas_estables=[INDEX_NAMES[i] for i in orden[:4]],
                menos_estables=[INDEX_NAMES[i] for i in orden[-4:]])


def persistencia_con_test(mats: List[np.ndarray], top_p: float) -> Dict:
    """Phi(e) por arista, con test binomial y correccion BH.

    El informe advierte que Phi sola no distingue recurrencia por azar de
    recurrencia por estructura. Bajo el nulo cada arista se elige con
    probabilidad top_p en cada semilla de forma independiente, asi que el
    numero de apariciones es binomial(M, top_p).
    """
    P, dirigidas = presencia_por_semilla(mats, top_p)
    M = len(mats)
    phi = P.mean(axis=0)
    ps = np.array([stats.binomtest(int(P[:, k].sum()), M, top_p,
                                   alternative="greater").pvalue
                   for k in range(P.shape[1])])
    orden = np.argsort(ps)
    m = len(ps)
    q = np.minimum.accumulate(
        (ps[orden] * m / np.arange(1, m + 1))[::-1])[::-1]
    qs = np.empty(m)
    qs[orden] = np.clip(q, 0, 1)
    sig = np.where(qs < 0.05)[0]
    sig = sig[np.argsort(-phi[sig])]
    return dict(
        n_significativas=int(len(sig)),
        aristas=[dict(arista=f"{INDEX_NAMES[dirigidas[k][0]]}->"
                             f"{INDEX_NAMES[dirigidas[k][1]]}",
                      phi=float(phi[k]), q=float(qs[k])) for k in sig[:12]])


def main() -> int:
    ap = argparse.ArgumentParser(description="Metricas §13 y §6.2 del informe.")
    ap.add_argument("--stack", default=None)
    ap.add_argument("--matrices", default=None)
    ap.add_argument("--paradigmas", default=None)
    ap.add_argument("--salida", default="resultados")
    ap.add_argument("--top-p", type=float, default=0.15, dest="top_p")
    ap.add_argument("--permutaciones", type=int, default=300)
    ap.add_argument("--semilla", type=int, default=0)
    a = ap.parse_args()

    stack_p = a.stack or _buscar(CANDIDATOS_STACK, False)
    PC, RR = pc_y_r(np.load(stack_p))
    dirm = a.matrices or _buscar(CANDIDATOS_MATRICES, True)
    dirp = a.paradigmas or _buscar(CANDIDATOS_PARADIGMAS, True)
    fam = familia_de_indice()
    fuentes = [(e, p) for e, _n, _A, _s, p in
               recolectar_fuentes(dirm, dirp, PC, RR) if p and len(p) >= 5]
    print(f"  fuentes: {len(fuentes)}   Top-P = {a.top_p}")

    rng = np.random.default_rng(a.semilla)
    filas = []
    for etiqueta, mats in fuentes:
        print(f"    {etiqueta}")
        A = np.mean(mats, axis=0)
        filas.append(dict(
            etiqueta=etiqueta, n_semillas=len(mats),
            reciprocidad=reciprocidad(A, a.top_p),
            **analizar_sustitucion(mats, a.top_p, fam, a.permutaciones, rng),
            **{f"ov_{k}": v for k, v in overlap_por_nodo(mats).items()},
            **{f"pers_{k}": v for k, v in
               persistencia_con_test(mats, a.top_p).items()}))

    W = 78
    L = ["=" * W, " SUSTITUCION, RECIPROCIDAD Y ESTABILIDAD POR NODO",
         "", " Metricas de la seccion 13 y 6.2 del informe del framework.",
         f" Grafo discretizado con Top-P = {a.top_p}.", "=" * W, ""]
    L.append(f"  {'fuente':<30}{'recipr':>8}{'sustit':>8}{'compl':>8}"
             f"{'overlap':>9}{'persist':>9}")
    for r in filas:
        L.append(f"  {r['etiqueta']:<30}{r['reciprocidad']:>8.3f}"
                 f"{r.get('n_sustitutivas', 0):>8}"
                 f"{r.get('n_complementarias', 0):>8}"
                 f"{r['ov_overlap_medio']:>9.3f}"
                 f"{r['pers_n_significativas']:>9}")
    L.append("")
    L.append("  recipr = fraccion de aristas con su reciproca presente. Alta")
    L.append("     indica relaciones simetricas, o sea correlacionales; baja")
    L.append("     indica direccionalidad.")
    L.append("  sustit / compl = pares de aristas acopladas entre semillas por")
    L.append("     encima del percentil 99 del nulo. Sustitutivas son las que")
    L.append("     se excluyen (phi<0): el modelo ELIGE entre ellas y el peso")
    L.append("     no esta identificado. Complementarias aparecen juntas.")
    L.append("  overlap = fraccion del Top-3 de cada fila que sobrevive entre")
    L.append("     dos semillas.")
    L.append("  persist = aristas con frecuencia mayor que el azar, test")
    L.append("     binomial con correccion de Benjamini-Hochberg.")
    for r in filas:
        L.append("")
        L.append("-" * W)
        L.append(f" {r['etiqueta']}  ({r['n_semillas']} semillas)")
        L.append(f"  aristas que varian entre semillas: "
                 f"{r.get('n_aristas_variables', 0)} de {N_IDX * (N_IDX - 1)}")
        if r.get("top_sustitutivas"):
            L.append(f"  SUSTITUTIVAS (el modelo elige una u otra), "
                     f"{r['n_sustitutivas']} significativas:")
            for t in r["top_sustitutivas"][:6]:
                marca = " [mismo origen]" if t["mismo_origen"] else ""
                L.append(f"    {t['e1']:>22}  vs {t['e2']:<22}"
                         f" NMI={t['nmi']:.3f} phi={t['phi']:+.2f}{marca}")
            L.append(f"    de las sustitutivas, comparten origen el "
                     f"{r['frac_sust_mismo_origen'] * 100:.0f}% y familia de "
                     f"destino el {r['frac_sust_misma_familia'] * 100:.0f}%")
        if r.get("top_complementarias"):
            L.append(f"  COMPLEMENTARIAS (aparecen juntas), "
                     f"{r['n_complementarias']} significativas:")
            for t in r["top_complementarias"][:4]:
                L.append(f"    {t['e1']:>22}  y  {t['e2']:<22}"
                         f" NMI={t['nmi']:.3f} phi={t['phi']:+.2f}")
        L.append(f"  estabilidad por nodo (overlap@3): mas estables "
                 f"{', '.join(r['ov_mas_estables'])}")
        L.append(f"    menos estables {', '.join(r['ov_menos_estables'])}")
        if r["pers_aristas"]:
            L.append("  aristas persistentes (binomial + BH):")
            for t in r["pers_aristas"][:6]:
                L.append(f"    {t['arista']:<26} phi={t['phi']:.2f} "
                         f"q={t['q']:.4g}")
    L.append("=" * W)
    txt = "\n".join(L)
    print("\n" + txt)
    os.makedirs(a.salida, exist_ok=True)
    with open(os.path.join(a.salida, "ANALISIS_SUSTITUCION.json"), "w",
              encoding="utf-8") as f:
        json.dump(filas, f, indent=2, ensure_ascii=False)
    with open(os.path.join(a.salida, "ANALISIS_SUSTITUCION.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt)
    print(f"\n  Guardado: "
          f"{os.path.join(a.salida, 'ANALISIS_SUSTITUCION.txt')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
