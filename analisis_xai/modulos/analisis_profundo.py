#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analisis geometrico y topologico profundo del grafo. INDEPENDIENTE del pipeline.

QUE HACE Y POR QUE CADA COSA
----------------------------
Los modulos anteriores miden orden de aristas, reproducibilidad, comunidades y
Markov. Aqui se atacan las propiedades que no se ven en ninguna de esas: la
forma del grafo a distintas escalas, su geometria local, y cuanta de su
estructura espectral supera lo que produciria el azar con las mismas
marginales.

  HOMOLOGIA PERSISTENTE   Se filtra el grafo bajando el umbral de peso y se
      registra cuando nacen y mueren las componentes (H0) y los ciclos (H1).
      Un ciclo que sobrevive un rango largo de umbrales es estructura real; uno
      que aparece y muere enseguida es ruido. Responde "que forma tiene esto"
      sin fijar ningun corte, que es el problema que tuvo todo el proyecto con
      Top-P.

  HEAT KERNEL exp(-tL)    Difusion sobre el grafo. A t pequeno la difusion solo
      ve vecinos inmediatos; a t grande ve la estructura global. Barrer t da la
      escala a la que el grafo tiene estructura, y la traza del kernel es una
      firma que no depende del etiquetado.

  EMBEDDING vs BANDAS     Los autovectores del laplaciano dan coordenadas de
      cada indice. La pregunta directa: esa geometria coincide con la del
      espacio de bandas Sentinel-2 (que banda entra en que formula)? Se mide
      con Procrustes. Si coincide, el grafo esta recuperando la construccion
      algebraica de los indices; si no, esta viendo otra cosa.

  CURVATURA DE OLLIVIER-RICCI   Curvatura por arista via transporte optimo
      entre las distribuciones de vecinos. Negativa marca cuellos de botella
      -aristas puente entre regiones-, positiva marca zonas densas. Es
      geometria local que ni el espectro ni las comunidades capturan.

  RESOLVENTE (I - zA)^-1  La transformada que pediste. Es la funcion
      generatriz de caminos: su expansion suma z^k A^k, o sea todos los caminos
      de longitud k pesados por z^k. Los polos estan en 1/lambda_i, asi que la
      resolvente codifica el espectro completo, y evaluarla a distintos z dice
      a que longitud de camino vive la estructura.

  ENTROPIA DE VON NEUMANN Con rho = L/tr(L) como matriz de densidad, mide la
      complejidad estructural del grafo en un solo numero comparable entre
      fuentes.

  MATRICES ALEATORIAS     El espectro de una matriz sin estructura sigue una
      ley de semicirculo. Los autovalores que se salen de ese bulk son la
      senal. Cuantos hay y cuanto se salen es una medida de estructura que no
      necesita verdad externa de ningun tipo.

USO
    .venv/bin/python analisis_profundo.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.optimize import linprog

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from certificar_todo import (  # noqa: E402
    BANDAS_INDICE, CANDIDATOS_MATRICES, CANDIDATOS_PARADIGMAS,
    CANDIDATOS_STACK, INDEX_NAMES, N_IDX, PARES, _buscar, familia_de_indice,
    pc_y_r, recolectar_fuentes,
)
from analisis_grafo import _simetrica_positiva  # noqa: E402


# ---------------------------------------------------------------------------
# Homologia persistente
# ---------------------------------------------------------------------------

def persistencia(A: np.ndarray) -> Dict:
    """Diagramas H0 y H1 sobre la filtracion por peso decreciente.

    Se recorren las 66 aristas de mayor a menor peso anadiendolas una a una.
    Cada arista o une dos componentes (muerte de una clase H0) o cierra un
    ciclo (nacimiento de una clase H1). La persistencia de cada clase es el
    rango de umbrales en que existe.

    Con 12 nodos H0 y H1 se calculan exactamente y sin librerias: H0 con
    union-find, y el numero de Betti b1 por la formula de Euler para grafos,
    b1 = aristas - nodos + componentes. No hace falta el complejo de Vietoris-
    Rips completo porque el objeto es un grafo, no una nube de puntos.
    """
    S = _simetrica_positiva(A)
    aristas = sorted(((S[i, j], i, j) for i, j in PARES), reverse=True)
    padre = list(range(N_IDX))

    def raiz(x):
        while padre[x] != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    w_max = aristas[0][0] if aristas else 1.0
    nace_h0 = [w_max] * N_IDX          # todas las componentes nacen al inicio
    barras_h0, barras_h1 = [], []
    n_comp, n_ar = N_IDX, 0
    curva_b1 = []
    for w, i, j in aristas:
        ri, rj = raiz(i), raiz(j)
        n_ar += 1
        if ri != rj:
            padre[ri] = rj
            n_comp -= 1
            # Muere la componente mas joven de las dos.
            nacimiento = max(nace_h0[ri], nace_h0[rj])
            barras_h0.append((float(nacimiento), float(w)))
            nace_h0[rj] = min(nace_h0[ri], nace_h0[rj])
        else:
            barras_h1.append([float(w), 0.0])
        curva_b1.append((float(w), int(n_ar - N_IDX + n_comp)))

    # H1 SOBRE EL COMPLEJO DE CLIQUES, no sobre el grafo pelado. La primera
    # version cerraba los ciclos con la formula de Euler b1 = E - V + C, que
    # sobre 12 nodos y 66 aristas da 55 SIEMPRE, para cualquier matriz: es una
    # constante del grafo completo y no distingue nada. Un ciclo solo esta
    # realmente vivo mientras no se rellene con triangulos, asi que hay que
    # descontar las 2-celdas: b1 = E - V + C - (rango de las caras).
    # Con 12 nodos el complejo de Rips de dimension 2 se calcula exacto.
    orden_w = [w for w, _i, _j in aristas]
    umbrales = sorted(set(orden_w), reverse=True)
    if len(umbrales) > 40:
        umbrales = [umbrales[k] for k in
                    np.linspace(0, len(umbrales) - 1, 40).astype(int)]
    curva_b1 = []
    for u in umbrales:
        act = [(i, j) for w, i, j in aristas if w >= u]
        if not act:
            continue
        adj = np.zeros((N_IDX, N_IDX), dtype=bool)
        for i, j in act:
            adj[i, j] = adj[j, i] = True
        # componentes
        vis, comp = np.zeros(N_IDX, dtype=bool), 0
        for v in range(N_IDX):
            if vis[v]:
                continue
            comp += 1
            pila = [v]
            vis[v] = True
            while pila:
                x = pila.pop()
                for y in np.where(adj[x] & ~vis)[0]:
                    vis[y] = True
                    pila.append(y)
        tri = sum(1 for a_ in range(N_IDX) for b_ in range(a_ + 1, N_IDX)
                  for c_ in range(b_ + 1, N_IDX)
                  if adj[a_, b_] and adj[b_, c_] and adj[a_, c_])
        b1 = max(0, len(act) - N_IDX + comp - tri)
        curva_b1.append((float(u), int(b1)))
    # Persistencia de H1: cuanto rango de umbrales hay con ciclos vivos.
    vivos = [u for u, b in curva_b1 if b > 0]
    rango_h1 = (float(max(vivos) - min(vivos)) if len(vivos) > 1 else 0.0)
    w_min = aristas[-1][0] if aristas else 0.0
    for b in barras_h1:
        b[1] = float(w_min)

    pers_h0 = sorted((n - m for n, m in barras_h0), reverse=True)
    pers_h1 = sorted((n - m for n, m in barras_h1), reverse=True)
    return dict(
        n_ciclos_h1=len(barras_h1),
        b1_en_umbral_conexion=0,
        persistencia_h0_top=[float(x) for x in pers_h0[:6]],
        persistencia_h1_top=[float(x) for x in pers_h1[:6]],
        persistencia_h0_total=float(sum(pers_h0)),
        persistencia_h1_total=float(sum(pers_h1)),
        b1_maximo=int(max((b for _w, b in curva_b1), default=0)),
        rango_umbrales_con_ciclos=rango_h1,
        curva_b1=[[float(u), int(b)] for u, b in curva_b1],
        # Umbral donde el grafo queda conexo: es el corte natural, y sale de la
        # filtracion en vez de fijarse a mano como Top-P.
        umbral_conexion=float(barras_h0[-1][1]) if barras_h0 else float("nan"))


# ---------------------------------------------------------------------------
# Heat kernel y difusion
# ---------------------------------------------------------------------------

def laplaciano(A: np.ndarray) -> np.ndarray:
    S = _simetrica_positiva(A)
    d = np.maximum(S.sum(axis=1), 1e-12)
    return np.eye(N_IDX) - S / np.sqrt(np.outer(d, d))


def laplaciano_combinatorio(A: np.ndarray) -> np.ndarray:
    """L = D - W, sin normalizar.

    El laplaciano normalizado de un grafo denso tiene casi todos los
    autovalores en 1, asi que su traza de calor y su entropia de von Neumann
    salen identicas para cualquier matriz de este proyecto (medido: 2.398 =
    log(11) en las ocho fuentes). El combinatorio conserva la escala de los
    pesos y si distingue.
    """
    S = _simetrica_positiva(A)
    return np.diag(S.sum(axis=1)) - S


def heat_kernel(A: np.ndarray, ts: Tuple[float, ...]) -> Dict:
    """Traza y firma del nucleo del calor a varias escalas.

    exp(-tL) describe como se difunde calor por el grafo en tiempo t. La traza
    sum_i exp(-t lambda_i) es un invariante que no depende del etiquetado, y su
    caida con t dice a que escala el grafo tiene estructura: un grafo con
    comunidades marcadas retiene calor dentro de ellas y su traza cae despacio.
    """
    Lc = laplaciano_combinatorio(A)
    w, V = np.linalg.eigh(Lc)
    # Tiempo adimensional: se escala por el autovalor mayor para que t signifique
    # lo mismo en fuentes con rangos de peso distintos.
    w = w / max(w.max(), 1e-12)
    out = {}
    for t in ts:
        e = np.exp(-t * w)
        K = (V * e) @ V.T
        out[f"traza_t{t:g}"] = float(e.sum())
        out[f"hks_t{t:g}"] = {INDEX_NAMES[i]: float(K[i, i])
                              for i in range(N_IDX)}
    # Escala caracteristica: donde la traza cae a la mitad de su rango.
    ts_fino = np.logspace(-2, 2.5, 80)
    trazas = np.array([np.exp(-t * w).sum() for t in ts_fino])
    objetivo = (trazas[0] + trazas[-1]) / 2
    out["escala_caracteristica"] = float(
        ts_fino[int(np.argmin(np.abs(trazas - objetivo)))])
    return out


def distancia_difusion(A: np.ndarray, t: float = 1.0) -> np.ndarray:
    """Distancia entre nodos segun como se difunde el calor desde cada uno.

    Dos indices estan cerca si el calor que sale de ellos se reparte igual, o
    sea si juegan el mismo papel en el grafo. Es una nocion de equivalencia
    estructural mucho mas fina que compartir una arista fuerte.
    """
    w, V = np.linalg.eigh(laplaciano(A))
    Phi = V * np.exp(-t * w)
    D = np.zeros((N_IDX, N_IDX))
    for i in range(N_IDX):
        for j in range(i + 1, N_IDX):
            D[i, j] = D[j, i] = np.linalg.norm(Phi[i] - Phi[j])
    return D


# ---------------------------------------------------------------------------
# Embedding contra el espacio de bandas
# ---------------------------------------------------------------------------

def matriz_bandas() -> Tuple[np.ndarray, List[str]]:
    bandas = sorted({b for v in BANDAS_INDICE.values() for b in v})
    M = np.zeros((N_IDX, len(bandas)))
    for i, nombre in enumerate(INDEX_NAMES):
        for b in BANDAS_INDICE.get(nombre, ()):
            M[i, bandas.index(b)] = 1.0
    return M, bandas


def procrustes_bandas(A: np.ndarray, k: int = 3) -> Dict:
    """Compara el embedding espectral del grafo con el espacio de bandas.

    LA PREGUNTA QUE DECIDE SI EL GRAFO APORTA. Si la geometria que induce el
    grafo coincide con la de "que banda entra en que formula", el modelo esta
    recuperando la construccion algebraica de los indices y no el fenomeno. Si
    no coincide, esta viendo otra cosa, y esa otra cosa es lo interesante.

    Se alinean ambas nubes con Procrustes ortogonal (que absorbe rotacion,
    reflexion y escala, porque ninguna de las dos tiene ejes con significado) y
    se reporta la disparidad residual: 0 = geometrias identicas, 1 = sin
    relacion.
    """
    w, V = np.linalg.eigh(laplaciano(A))
    X = V[:, 1:k + 1]
    Y, _ = matriz_bandas()
    Yc = Y - Y.mean(0)
    u, s, vt = np.linalg.svd(Yc, full_matrices=False)
    Y3 = (u[:, :k] * s[:k])
    def norm(Z):
        Z = Z - Z.mean(0)
        return Z / max(np.linalg.norm(Z), 1e-12)
    Xn, Yn = norm(X), norm(Y3)
    u2, s2, _ = np.linalg.svd(Xn.T @ Yn)
    disparidad = float(max(0.0, 1 - (s2.sum() ** 2)))
    return dict(disparidad_procrustes=disparidad,
                similitud_con_bandas=float(s2.sum() ** 2))


# ---------------------------------------------------------------------------
# Curvatura de Ollivier-Ricci
# ---------------------------------------------------------------------------

def _w1(mu: np.ndarray, nu: np.ndarray, D: np.ndarray) -> float:
    """Distancia de Wasserstein-1 exacta por programacion lineal."""
    n = len(mu)
    c = D.reshape(-1)
    A_eq, b_eq = [], []
    for i in range(n):
        fila = np.zeros((n, n))
        fila[i, :] = 1
        A_eq.append(fila.reshape(-1))
        b_eq.append(mu[i])
    for j in range(n):
        fila = np.zeros((n, n))
        fila[:, j] = 1
        A_eq.append(fila.reshape(-1))
        b_eq.append(nu[j])
    r = linprog(c, A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                bounds=(0, None), method="highs")
    return float(r.fun) if r.success else float("nan")


def curvatura_ollivier(A: np.ndarray, alpha: float = 0.5) -> Dict:
    """Curvatura de Ricci por arista, via transporte optimo.

    kappa(i,j) = 1 - W1(mu_i, mu_j) / d(i,j)

    Negativa significa que la arista es un puente: mover la masa de los vecinos
    de i a los de j cuesta mas que la propia arista, lo que pasa cuando i y j
    estan en regiones distintas. Positiva significa vecindario compartido. Es
    informacion local que el espectro promedia y las comunidades binarizan.
    """
    S = _simetrica_positiva(A)
    # Distancia de camino minimo con coste inverso al peso.
    W = np.where(S > 0, 1.0 / np.maximum(S, 1e-12), np.inf)
    np.fill_diagonal(W, 0.0)
    D = W.copy()
    for k in range(N_IDX):                       # Floyd-Warshall
        D = np.minimum(D, D[:, [k]] + D[[k], :])
    finito = D[np.isfinite(D)]
    if finito.size:
        D = np.where(np.isfinite(D), D, finito.max() * 10)

    mus = []
    for i in range(N_IDX):
        m = S[i].copy()
        tot = m.sum()
        m = m / tot if tot > 0 else np.full(N_IDX, 1.0 / N_IDX)
        m = (1 - alpha) * m
        m[i] += alpha
        mus.append(m)

    curv = {}
    for i, j in PARES:
        if S[i, j] <= 0 or D[i, j] <= 0:
            continue
        curv[(i, j)] = 1.0 - _w1(mus[i], mus[j], D) / D[i, j]
    vals = np.array(list(curv.values()))
    orden = sorted(curv.items(), key=lambda kv: kv[1])
    return dict(
        curvatura_media=float(vals.mean()) if vals.size else float("nan"),
        curvatura_min=float(vals.min()) if vals.size else float("nan"),
        n_aristas_negativas=int((vals < 0).sum()),
        puentes=[dict(par=f"{INDEX_NAMES[i]}-{INDEX_NAMES[j]}",
                      kappa=float(k)) for (i, j), k in orden[:6]],
        nucleos=[dict(par=f"{INDEX_NAMES[i]}-{INDEX_NAMES[j]}",
                      kappa=float(k)) for (i, j), k in orden[-4:]])


# ---------------------------------------------------------------------------
# Resolvente, entropia y matrices aleatorias
# ---------------------------------------------------------------------------

def resolvente(A: np.ndarray) -> Dict:
    """(I - zA)^-1: funcion generatriz de caminos, evaluada a varias escalas.

    Su expansion en serie es sum_k z^k A^k, o sea todos los caminos de longitud
    k pesados por z^k. Con z pequeno dominan los caminos cortos; al acercarse a
    1/lambda_max la serie diverge y dominan los largos. Comparar la traza a
    distintos z dice a que longitud de camino vive la estructura, que es la
    pregunta que la lectura arista a arista no puede formular.
    """
    S = _simetrica_positiva(A)
    lmax = float(np.abs(np.linalg.eigvalsh(S)).max())
    out = {"radio_espectral": lmax}
    for frac in (0.1, 0.3, 0.5, 0.7, 0.9):
        z = frac / max(lmax, 1e-12)
        try:
            R = np.linalg.inv(np.eye(N_IDX) - z * S)
            out[f"traza_resolvente_z{frac:g}"] = float(np.trace(R))
            # Comunicabilidad: cuanto se comunican los nodos por TODOS los
            # caminos, no solo por la arista directa.
            out[f"comunicabilidad_z{frac:g}"] = float(
                (R.sum() - np.trace(R)) / (N_IDX * (N_IDX - 1)))
        except np.linalg.LinAlgError:
            pass
    # Indice de Estrada: suma de exp(lambda_i), cuenta subgrafos cerrados.
    ev = np.linalg.eigvalsh(S)
    out["indice_estrada"] = float(np.exp(ev).sum())
    return out


def entropia_von_neumann(A: np.ndarray) -> float:
    """S = -tr(rho log rho) con rho = L / tr(L). Complejidad estructural."""
    L = laplaciano_combinatorio(A)
    t = np.trace(L)
    if t <= 0:
        return 0.0
    ev = np.linalg.eigvalsh(L / t)
    ev = ev[ev > 1e-12]
    return float(-(ev * np.log(ev)).sum())


def fuera_del_bulk(A: np.ndarray, n_nulos: int = 200,
                   rng: Optional[np.random.Generator] = None) -> Dict:
    """Autovalores que se salen de lo que produce el azar con las mismas fuerzas.

    El espectro de una matriz simetrica sin estructura se concentra en un bulk
    (ley de semicirculo). Los autovalores que quedan fuera son senal. El nulo
    aqui no es una gaussiana teorica sino una permutacion de las entradas de la
    propia matriz, que conserva su distribucion de pesos exacta y solo destruye
    donde esta cada uno: asi el bulk se calibra con los mismos numeros.
    """
    rng = rng or np.random.default_rng(0)
    S = _simetrica_positiva(A)
    ev = np.sort(np.linalg.eigvalsh(S))[::-1]
    vals = np.array([S[i, j] for i, j in PARES])
    maxs = []
    for _ in range(n_nulos):
        v = rng.permutation(vals)
        M = np.zeros((N_IDX, N_IDX))
        for k, (i, j) in enumerate(PARES):
            M[i, j] = M[j, i] = v[k]
        e = np.linalg.eigvalsh(M)
        maxs.append([e[-1], e[-2], e[0]])
    maxs = np.array(maxs)
    lim_sup = float(np.percentile(maxs[:, 1], 99))   # borde del bulk
    return dict(
        autovalores_top=[float(x) for x in ev[:4]],
        borde_bulk=lim_sup,
        n_fuera_del_bulk=int((ev[1:] > lim_sup).sum()),
        separacion_primero=float(ev[0] - float(np.percentile(maxs[:, 0], 99))),
        p_primero=float((1 + int((maxs[:, 0] >= ev[0]).sum()))
                        / (1 + len(maxs))))


# ---------------------------------------------------------------------------

def analizar(etiqueta: str, mats: List[np.ndarray],
             rng: np.random.Generator) -> Dict:
    A = np.mean(mats, axis=0)
    TS = (0.1, 0.5, 1.0, 3.0, 10.0)
    hk = heat_kernel(A, TS)
    per = persistencia(A)
    per_sem = [persistencia(m)["n_ciclos_h1"] for m in mats]
    Dd = distancia_difusion(A, t=1.0)
    orden_hks = sorted(hk["hks_t1"].items(), key=lambda kv: -kv[1])
    return dict(
        etiqueta=etiqueta, n_semillas=len(mats),
        **{k: v for k, v in per.items()},
        ciclos_h1_semilla_media=float(np.mean(per_sem)),
        ciclos_h1_semilla_std=float(np.std(per_sem)),
        escala_caracteristica=hk["escala_caracteristica"],
        trazas_heat={f"t={t:g}": hk[f"traza_t{t:g}"] for t in TS},
        hks_top=[dict(indice=k, valor=v) for k, v in orden_hks[:5]],
        pares_mas_cercanos_difusion=[
            dict(par=f"{INDEX_NAMES[i]}-{INDEX_NAMES[j]}",
                 d=float(Dd[i, j]))
            for i, j in sorted(PARES, key=lambda p: Dd[p[0], p[1]])[:6]],
        **procrustes_bandas(A),
        **curvatura_ollivier(A),
        **resolvente(A),
        entropia_von_neumann=entropia_von_neumann(A),
        **fuera_del_bulk(A, rng=rng))


def texto(filas: List[Dict]) -> str:
    W = 78
    L: List[str] = []
    L.append("=" * W)
    L.append(" ANALISIS GEOMETRICO Y TOPOLOGICO PROFUNDO")
    L.append("=" * W)
    L.append("")
    L.append(f"  {'fuente':<30}{'b1':>4}{'rango':>7}{'escala':>8}"
             f"{'bandas':>8}{'curv':>8}{'vonNeu':>8}{'fuera':>7}{'p1':>7}")
    for r in filas:
        L.append(f"  {r['etiqueta']:<30}{r['b1_maximo']:>4}"
                 f"{r['rango_umbrales_con_ciclos']:>7.3f}"
                 f"{r['escala_caracteristica']:>8.2f}"
                 f"{r['similitud_con_bandas']:>8.3f}"
                 f"{r['curvatura_media']:>8.3f}"
                 f"{r['entropia_von_neumann']:>8.3f}"
                 f"{r['n_fuera_del_bulk']:>7}{r['p_primero']:>7.3f}")
    L.append("")
    L.append("  b1 = ciclos maximos del complejo de cliques a lo largo de la")
    L.append("     filtracion. La version sobre el grafo pelado daba 55 en las")
    L.append("     ocho fuentes: es E-V+C del grafo completo, una constante.")
    L.append("  rango = ancho de umbrales en que hay ciclos vivos.")
    L.append("  escala = tiempo de difusion donde el grafo revela su")
    L.append("     estructura; pequena = estructura local, grande = global.")
    L.append("  bandas = similitud de Procrustes entre el embedding espectral")
    L.append("     del grafo y el espacio de bandas Sentinel-2. Alta significa")
    L.append("     que el grafo reproduce la construccion algebraica de los")
    L.append("     indices y no el fenomeno.")
    L.append("  curv = curvatura de Ollivier-Ricci media. Negativa = grafo con")
    L.append("     cuellos de botella entre regiones.")
    L.append("  vonNeu = entropia de von Neumann, complejidad estructural.")
    L.append("  fuera = autovalores fuera del bulk del nulo (senal espectral).")
    L.append("  p1 = p del primer autovalor contra el nulo por permutacion.")
    for r in filas:
        L.append("")
        L.append("-" * W)
        L.append(f" {r['etiqueta']}  ({r['n_semillas']} semillas)")
        L.append(f"  topologia: {r['n_ciclos_h1']} ciclos H1 "
                 f"(por semilla {r['ciclos_h1_semilla_media']:.1f}"
                 f" +-{r['ciclos_h1_semilla_std']:.1f}), b1 maximo "
                 f"{r['b1_maximo']}, umbral de conexion "
                 f"{r['umbral_conexion']:.4f}")
        L.append(f"    persistencia H0 mas larga: "
                 + ", ".join(f"{x:.4f}" for x in r['persistencia_h0_top'][:4]))
        L.append(f"  difusion: escala caracteristica "
                 f"{r['escala_caracteristica']:.3f}; trazas "
                 + ", ".join(f"{k} {v:.2f}"
                             for k, v in r["trazas_heat"].items()))
        L.append("    nodos que mas retienen calor (t=1): "
                 + ", ".join(f"{d['indice']} {d['valor']:.3f}"
                             for d in r["hks_top"][:4]))
        L.append("    pares mas cercanos en difusion (papel estructural igual):")
        L.append("      " + ", ".join(f"{d['par']}"
                                      for d in r["pares_mas_cercanos_difusion"][:5]))
        L.append(f"  geometria vs bandas: similitud "
                 f"{r['similitud_con_bandas']:.3f} "
                 f"(disparidad {r['disparidad_procrustes']:.3f})")
        L.append(f"  curvatura media {r['curvatura_media']:+.3f}, minima "
                 f"{r['curvatura_min']:+.3f}, "
                 f"{r['n_aristas_negativas']} aristas negativas")
        L.append("    puentes (curvatura mas negativa): "
                 + ", ".join(f"{d['par']} ({d['kappa']:+.2f})"
                             for d in r["puentes"][:4]))
        L.append(f"  espectro: radio {r['radio_espectral']:.4f}, Estrada "
                 f"{r['indice_estrada']:.2f}, autovalores top "
                 + ", ".join(f"{x:.3f}" for x in r["autovalores_top"]))
        L.append(f"    fuera del bulk: {r['n_fuera_del_bulk']}, borde "
                 f"{r['borde_bulk']:.4f}, separacion del primero "
                 f"{r['separacion_primero']:+.4f}, p={r['p_primero']:.4f}")
        com = {k: v for k, v in r.items() if k.startswith("comunicabilidad_z")}
        L.append("    comunicabilidad por caminos: "
                 + ", ".join(f"{k.split('_z')[1]}={v:.3f}"
                             for k, v in com.items()))
    L.append("=" * W)
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="Analisis geometrico profundo.")
    ap.add_argument("--stack", default=None)
    ap.add_argument("--matrices", default=None)
    ap.add_argument("--paradigmas", default=None)
    ap.add_argument("--salida", default="resultados")
    ap.add_argument("--semilla", type=int, default=0)
    a = ap.parse_args()

    stack_p = a.stack or _buscar(CANDIDATOS_STACK, False)
    if not stack_p:
        print("ERROR: no encuentro el stack.")
        return 1
    PC, RR = pc_y_r(np.load(stack_p))
    dirm = a.matrices or _buscar(CANDIDATOS_MATRICES, True)
    dirp = a.paradigmas or _buscar(CANDIDATOS_PARADIGMAS, True)
    fuentes = [(e, p) for e, _n, _A, _s, p in
               recolectar_fuentes(dirm, dirp, PC, RR) if p and len(p) >= 5]
    print(f"  fuentes: {len(fuentes)}")
    rng = np.random.default_rng(a.semilla)
    filas = []
    for etiqueta, mats in fuentes:
        print(f"    analizando: {etiqueta}")
        filas.append(analizar(etiqueta, mats, rng))
    txt = texto(filas)
    print("\n" + txt)
    os.makedirs(a.salida, exist_ok=True)
    with open(os.path.join(a.salida, "ANALISIS_PROFUNDO.json"), "w",
              encoding="utf-8") as f:
        json.dump(filas, f, indent=2, ensure_ascii=False)
    with open(os.path.join(a.salida, "ANALISIS_PROFUNDO.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt)
    print(f"\n  Guardado: {os.path.join(a.salida, 'ANALISIS_PROFUNDO.txt')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
