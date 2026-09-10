#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
De que ARIDAD es la senal que reproducen los 50 mapas de atencion.

EL PROBLEMA QUE RESUELVE ESTE MODULO
------------------------------------
Todos los analisis anteriores le preguntaron a la atencion lo mismo: que par
de indices se relaciona con cual. Y en todos fallo: no supera el liston de
|r|, su asimetria esta dominada en un 85% por efecto de nodo, su topologia es
trivial (b1 = 0), su curvatura de Ollivier-Ricci es casi constante y sus
aristas se acoplan con NMI = 1.

Esa lista de fracasos tiene una lectura mas barata que "la atencion es ruido":
que la atencion SI lleva senal reproducible, pero de aridad UNO y no de aridad
DOS. O sea que contesta "a que indice se mira", no "que indice depende de
cual". Si eso es cierto, cada uno de los fracasos anteriores deja de ser un
resultado negativo suelto y pasa a ser la consecuencia obligada de un solo
hecho medible.

COMO SE MIDE
------------
Sobre las entradas fuera de la diagonal de cada matriz se hace la
descomposicion aditiva de dos vias

    A[i,j] = mu + r_i + c_j + R[i,j]

    mu    nivel global                    (aridad 0)
    r_i   cuanto EMITE la fila i          (aridad 1)
    c_j   cuanto RECIBE la columna j      (aridad 1, el efecto sumidero)
    R     lo que queda, y solo eso es una relacion de PAR (aridad 2)

Los efectos se obtienen por centrado iterativo por filas y columnas, que con
la diagonal ausente converge a la solucion de minimos cuadrados y deja los
tres terminos ortogonales, de modo que las fracciones de suma de cuadrados
suman uno.

Con eso se contestan dos cosas distintas que hasta ahora iban mezcladas:

  1. CUANTA varianza de la matriz vive en cada aridad. Es descriptivo y
     depende de la escala de la fuente.
  2. CUANTA de esa varianza REPRODUCEN las 50 semillas, por aridad, via el
     ICC de cada componente sobre la tabla (semillas x componente). Esta es
     la que importa: una fuente puede tener mucha varianza de par y no
     reproducir ninguna, y al reves.

LA PRUEBA CONSTRUCTIVA
----------------------
Medir la aridad no basta, porque alguien puede objetar que la parte de par
existe aunque sea pequena. Asi que ademas se fabrica el SUSTITUTO UNARIO de
cada semilla,

    A_sur[i,j] = mu + r_i + c_j        (parte de par exactamente cero)

y se le pasan los mismos descriptores estructurales que se le pasaron a la
matriz real: comunidades, numero de Betti b1, curvatura de Ollivier-Ricci,
grafo Top-P, ranking de pares. El sustituto no puede contener ninguna
relacion binaria porque se construyo sin ella. Si aun asi reproduce los
descriptores de la atencion real, entonces esos descriptores nunca midieron
una relacion binaria en la atencion. Es una refutacion por construccion, no
una correlacion mas.

La ablacion entra como control: ahi el sustituto unario TIENE que fallar. Si
fallara en las dos, la prueba no distinguiria nada y habria que descartarla.

QUE ES EL VECTOR c
------------------
Si la senal reproducible de la atencion es unaria, entonces c es un objeto
interpretable por si mismo -un ranking de saliencia sobre los 12 indices- y
se puede preguntar contra que covariable fisica o estadistica se alinea:
varianza del dato, redundancia respecto de los otros once, conectividad
marginal, numero y resolucion de las bandas Sentinel-2, tamano de familia
espectral. Y sobre todo, si coincide con el ranking unario de la ABLACION:
que dos operadores de lectura independientes coincidan en QUE indices pesan,
aunque discrepen en QUE PAR, es un resultado positivo y no un consuelo.

USO
    .venv/bin/python analisis_atencion.py --permutaciones 500
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
    BANDAS_INDICE, CANDIDATOS_MATRICES, CANDIDATOS_PARADIGMAS,
    CANDIDATOS_STACK, INDEX_NAMES, N_IDX, PARES, RESOLUCION_BANDA,
    _buscar, familia_de_indice, fuerza_por_par, pc_y_r, recolectar_fuentes,
)
from analisis_grafo import (  # noqa: E402
    _indice_rand_ajustado, comunidades, discretizar_top_p, jaccard_grafos,
)
from analisis_profundo import curvatura_ollivier, persistencia  # noqa: E402

OFF = ~np.eye(N_IDX, dtype=bool)


# ---------------------------------------------------------------------------
# Descomposicion por aridad
# ---------------------------------------------------------------------------

def descomponer_aridad(A: np.ndarray, n_iter: int = 300
                       ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """mu, r, c, R por centrado iterativo sobre las entradas fuera de diagonal.

    La diagonal se excluye porque no es una relacion: en la atencion es
    autoatencion mas el 0.25*I que mete el rollout por construccion, y en la
    ablacion no esta definida. Al faltar una celda por fila el diseno deja de
    ser balanceado y restar la media de fila y la de columna una sola vez no
    ortogonaliza; iterar si converge a la solucion de minimos cuadrados.
    """
    M = np.asarray(A, dtype=np.float64).copy()
    M[~OFF] = np.nan
    mu = float(np.nanmean(M))
    R = M - mu
    r = np.zeros(N_IDX)
    c = np.zeros(N_IDX)
    for _ in range(n_iter):
        dr = np.nanmean(R, axis=1)
        R -= dr[:, None]
        r += dr
        dc = np.nanmean(R, axis=0)
        R -= dc[None, :]
        c += dc
        if max(np.abs(dr).max(), np.abs(dc).max()) < 1e-14:
            break
    R = np.nan_to_num(R, nan=0.0)
    return mu, r, c, R


def fracciones_aridad(A: np.ndarray) -> Dict:
    """Reparto de la suma de cuadrados fuera de diagonal entre las aridades."""
    _mu, r, c, R = descomponer_aridad(A)
    ss_fila = float((N_IDX - 1) * np.sum(r ** 2))
    ss_col = float((N_IDX - 1) * np.sum(c ** 2))
    ss_par = float(np.sum(R[OFF] ** 2))
    tot = ss_fila + ss_col + ss_par
    if tot <= 0:
        return dict(frac_fila=float("nan"), frac_col=float("nan"),
                    frac_par=float("nan"), frac_unaria=float("nan"))
    return dict(frac_fila=ss_fila / tot, frac_col=ss_col / tot,
                frac_par=ss_par / tot,
                frac_unaria=(ss_fila + ss_col) / tot)


def _icc(T: np.ndarray) -> float:
    """ICC de una tabla (semillas x componentes).

    Varianza entre componentes sobre varianza total. Cero significa que las
    componentes son indistinguibles y todo lo que se ve es la semilla.
    """
    if T.shape[0] < 2 or T.shape[1] < 2:
        return float("nan")
    entre = float(T.mean(axis=0).var(ddof=1))
    dentro = float(T.var(axis=0, ddof=1).mean())
    return entre / (entre + dentro) if (entre + dentro) > 0 else 0.0


def icc_por_aridad(mats: List[np.ndarray], n_perm: int = 500,
                   semilla: int = 0) -> Dict:
    """ICC de r, de c y de R a lo largo de las semillas, con nulo.

    NULO. La primera version barajaba las semillas DENTRO de cada componente.
    Eso no sirve: permutar las filas de una columna no cambia ni su media ni su
    varianza, asi que el ICC del nulo es identico al observado por construccion
    y el p sale pegado a 1 pase lo que pase. El nulo correcto baraja al reves,
    ACROSS componentes dentro de cada semilla: eso conserva la distribucion de
    valores de cada semilla y destruye que componente es cual, que es
    exactamente la hipotesis nula "las semillas no coinciden en que componente
    vale cuanto".
    """
    rs, cs, Rs = [], [], []
    for A in mats:
        _mu, r, c, R = descomponer_aridad(A)
        rs.append(r)
        cs.append(c)
        Rs.append(R[OFF])
    T = dict(fila=np.stack(rs), col=np.stack(cs), par=np.stack(Rs))
    rng = np.random.default_rng(semilla)
    out = {}
    for k, tabla in T.items():
        obs = _icc(tabla)
        nulos = np.empty(n_perm)
        for b in range(n_perm):
            Z = np.empty_like(tabla)
            for i in range(tabla.shape[0]):
                Z[i, :] = rng.permutation(tabla[i, :])
            nulos[b] = _icc(Z)
        out["icc_" + k] = float(obs)
        out["icc_" + k + "_nulo"] = float(np.mean(nulos))
        out["icc_" + k + "_p"] = float((np.sum(nulos >= obs) + 1)
                                       / (n_perm + 1))
    # En las fuentes normalizadas por fila el efecto emisor NO es libre: si
    # cada fila suma una constante, la media de fila fuera de diagonal queda
    # fijada por la diagonal, y r sale igual a c dividido por N-1. Medido sobre
    # la ablacion normalizada da corr(r,c) = +1.000 exacta con sd(r)/sd(c) =
    # 0.0909 = 1/11. O sea que en esas fuentes no hay "quien emite" separado de
    # "quien recibe", y toda la direccion que pueda haber vive en R.
    out["corr_r_c"] = float(np.corrcoef(T["fila"].mean(axis=0),
                                        T["col"].mean(axis=0))[0, 1])
    # Consenso: el vector medio de cada efecto unario.
    out["c_consenso"] = T["col"].mean(axis=0).tolist()
    out["r_consenso"] = T["fila"].mean(axis=0).tolist()
    # Concordancia del ranking unario entre semillas, por pares de semillas.
    for k in ("fila", "col"):
        tabla = T[k]
        vals = [stats.spearmanr(tabla[a], tabla[b]).statistic
                for a in range(len(tabla)) for b in range(a + 1, len(tabla))]
        out["spearman_medio_" + k] = float(np.nanmean(vals))
    return out


# ---------------------------------------------------------------------------
# Sustituto unario y descriptores estructurales
# ---------------------------------------------------------------------------

def sustituto_unario(A: np.ndarray) -> np.ndarray:
    """mu + r_i + c_j fuera de la diagonal: cero relacion de par, por diseno.

    Se desplaza el minimo a un valor positivo pequeno porque todos los
    descriptores estructurales (comunidades, curvatura, Top-P) piden pesos no
    negativos. El desplazamiento es una constante aditiva, o sea aridad cero,
    y por tanto no puede introducir estructura de par.
    """
    mu, r, c, _R = descomponer_aridad(A)
    S = mu + r[:, None] + c[None, :]
    S[~OFF] = 0.0
    m = S[OFF].min()
    if m <= 0:
        S[OFF] = S[OFF] - m + 1e-6 * max(abs(mu), 1e-12)
    return S


def descriptores(A: np.ndarray) -> Dict:
    """Los mismos descriptores estructurales de los modulos anteriores."""
    pers = persistencia(A)
    curv = curvatura_ollivier(A)
    return dict(
        b1_maximo=int(pers["b1_maximo"]),
        n_ciclos_h1=int(pers["n_ciclos_h1"]),
        curvatura_media=float(curv["curvatura_media"]),
        curvatura_min=float(curv["curvatura_min"]),
        n_aristas_negativas=int(curv["n_aristas_negativas"]),
        puente_top=curv["puentes"][0]["par"] if curv["puentes"] else "",
        com3=comunidades(A, k=3).tolist(),
        fuerza=fuerza_por_par(A).tolist(),
        topp=discretizar_top_p(A, 0.15).tolist())


def comparar_con_sustituto(mats: List[np.ndarray]) -> Dict:
    """Real contra sustituto unario, semilla a semilla y en consenso.

    Si el sustituto reproduce los descriptores, esos descriptores no estaban
    midiendo ninguna relacion binaria.
    """
    A_real = np.mean(mats, axis=0)
    A_sur = np.mean([sustituto_unario(m) for m in mats], axis=0)
    d_real, d_sur = descriptores(A_real), descriptores(A_sur)
    rho = stats.spearmanr(d_real["fuerza"], d_sur["fuerza"]).statistic
    ari = _indice_rand_ajustado(np.array(d_real["com3"]),
                                np.array(d_sur["com3"]))
    jac = jaccard_grafos(np.array(d_real["topp"]), np.array(d_sur["topp"]))
    # Por semilla, para que el resultado no dependa del promedio.
    rhos, aris, jacs = [], [], []
    for m in mats:
        s = sustituto_unario(m)
        rhos.append(stats.spearmanr(fuerza_por_par(m),
                                    fuerza_por_par(s)).statistic)
        aris.append(_indice_rand_ajustado(comunidades(m, 3), comunidades(s, 3)))
        jacs.append(jaccard_grafos(discretizar_top_p(m, 0.15),
                                   discretizar_top_p(s, 0.15)))
    return dict(
        real={k: v for k, v in d_real.items()
              if k not in ("com3", "fuerza", "topp")},
        sustituto={k: v for k, v in d_sur.items()
                   if k not in ("com3", "fuerza", "topp")},
        spearman_ranking=float(rho), ari_comunidades=float(ari),
        jaccard_topp=float(jac),
        spearman_por_semilla=float(np.nanmean(rhos)),
        ari_por_semilla=float(np.nanmean(aris)),
        jaccard_por_semilla=float(np.nanmean(jacs)))


# ---------------------------------------------------------------------------
# La parte de par, comparada contra la verdad y contra el otro operador
# ---------------------------------------------------------------------------

def par_simetrico(R: np.ndarray) -> np.ndarray:
    """Los 66 valores de par de una matriz ya descompuesta.

    Se simetriza por el mismo motivo que fuerza_por_par: la correlacion parcial
    contra la que se compara no tiene direccion, y penalizar a la fuente por
    expresar una que el patron de referencia no puede expresar seria injusto.
    Aqui se promedia en vez de tomar el maximo porque R ya esta centrado y el
    maximo de dos numeros centrados en cero es un estadistico sesgado.
    """
    S = 0.5 * (R + R.T)
    return np.array([S[i, j] for i, j in PARES])


def asimetria_relativa(A: np.ndarray) -> float:
    """||A - A^T|| / ||A||, para saber si la fuente es simetrica en disco."""
    A = np.asarray(A, dtype=np.float64)
    den = np.linalg.norm(A[OFF])
    return float(np.linalg.norm((A - A.T)[OFF]) / den) if den > 0 else 0.0


def par_contra_verdad(mats: List[np.ndarray], pc: np.ndarray, rr: np.ndarray,
                      n_perm: int = 2000, semilla: int = 0) -> Dict:
    """El efecto de nodo estaba tapando la senal de par?

    Esta es la pregunta que el resto del informe no pudo hacer. Hasta ahora la
    atencion se comparo contra |pc| con fuerza_por_par, que mezcla las tres
    aridades: si el 75% de la varianza de la matriz es efecto sumidero, ese 75%
    entra en la comparacion como ruido correlacionado y hunde cualquier senal
    de par que hubiera debajo. Comparar R en lugar de A separa las dos cosas.

    Se reporta tambien el liston |r| con el mismo tratamiento, y un test de
    signos sobre las 50 semillas: que la parte de par gane a la matriz cruda en
    la mayoria de las semillas es mas fuerte que ganarle en el promedio, porque
    el promedio lo podria decidir una sola semilla rara.
    """
    rng = np.random.default_rng(semilla)
    A_med = np.mean(mats, axis=0)
    _mu, _r, _c, R_med = descomponer_aridad(A_med)
    crudo = fuerza_por_par(A_med)
    par = par_simetrico(R_med)

    def _rho(x, y):
        return float(stats.spearmanr(x, y).statistic)

    # Nulo: permutar las 12 etiquetas de indice y rehacer los 66 pares. Es el
    # nulo que respeta que los 66 pares no son independientes -comparten
    # nodos-, cosa que permutar los 66 valores sueltos no respeta.
    def _p_perm(vec, ref):
        obs = abs(_rho(vec, ref))
        M = np.zeros((N_IDX, N_IDX))
        for k, (i, j) in enumerate(PARES):
            M[i, j] = M[j, i] = vec[k]
        cuenta = 0
        for _ in range(n_perm):
            o = rng.permutation(N_IDX)
            P = M[np.ix_(o, o)]
            v = np.array([P[i, j] for i, j in PARES])
            if abs(_rho(v, ref)) >= obs:
                cuenta += 1
        return float((cuenta + 1) / (n_perm + 1))

    por_semilla = []
    for m in mats:
        _m, _rr2, _cc2, Rm = descomponer_aridad(m)
        por_semilla.append((abs(_rho(par_simetrico(Rm), pc)),
                            abs(_rho(fuerza_por_par(m), pc))))
    gana = sum(1 for a, b in por_semilla if a > b)
    return dict(
        rho_crudo_pc=_rho(crudo, pc), rho_par_pc=_rho(par, pc),
        rho_crudo_r=_rho(crudo, rr), rho_par_r=_rho(par, rr),
        p_perm_crudo=_p_perm(crudo, pc), p_perm_par=_p_perm(par, pc),
        semillas_donde_el_par_gana=int(gana), n_semillas=len(mats),
        p_signos=float(stats.binomtest(gana, len(mats), 0.5,
                                       alternative="greater").pvalue),
        asimetria_relativa=asimetria_relativa(A_med))


def acuerdo_entre_operadores(m_at: List[np.ndarray], m_ab: List[np.ndarray],
                             n_perm: int = 2000, semilla: int = 0) -> Dict:
    """Atencion contra ablacion, antes y despues de quitar el efecto de nodo.

    Con el nulo de permutar etiquetas de indice, que es el unico que respeta
    que los 66 pares comparten nodos.
    """
    rng = np.random.default_rng(semilla)
    A_at, A_ab = np.mean(m_at, axis=0), np.mean(m_ab, axis=0)
    _a, _b, _c, R_at = descomponer_aridad(A_at)
    _d, _e, _f, R_ab = descomponer_aridad(A_ab)
    par_at, par_ab = par_simetrico(R_at), par_simetrico(R_ab)
    crudo_at, crudo_ab = fuerza_por_par(A_at), fuerza_por_par(A_ab)

    def _perm_p(x, y):
        obs = abs(float(stats.spearmanr(x, y).statistic))
        M = np.zeros((N_IDX, N_IDX))
        for k, (i, j) in enumerate(PARES):
            M[i, j] = M[j, i] = x[k]
        cuenta = 0
        for _ in range(n_perm):
            o = rng.permutation(N_IDX)
            P = M[np.ix_(o, o)]
            v = np.array([P[i, j] for i, j in PARES])
            if abs(float(stats.spearmanr(v, y).statistic)) >= obs:
                cuenta += 1
        return float((cuenta + 1) / (n_perm + 1))

    # Por semilla de cada lado, para no depender del promedio.
    pares_at = [par_simetrico(descomponer_aridad(m)[3]) for m in m_at]
    pares_ab = [par_simetrico(descomponer_aridad(m)[3]) for m in m_ab]
    cruz = [float(stats.spearmanr(a, b).statistic)
            for a, b in zip(pares_at, pares_ab)]
    n_pos = int(sum(1 for v in cruz if v > 0))
    return dict(
        rho_crudo=float(stats.spearmanr(crudo_at, crudo_ab).statistic),
        rho_par=float(stats.spearmanr(par_at, par_ab).statistic),
        p_perm_par=_perm_p(par_at, par_ab),
        p_perm_crudo=_perm_p(crudo_at, crudo_ab),
        rho_par_por_semilla=float(np.mean(cruz)),
        semillas_con_acuerdo_positivo=n_pos, n_semillas=len(cruz),
        p_signos=float(stats.binomtest(n_pos, len(cruz), 0.5,
                                       alternative="greater").pvalue))


# ---------------------------------------------------------------------------
# Contra que se alinea el vector de saliencia
# ---------------------------------------------------------------------------

def covariables_por_indice(stack: np.ndarray, PC: np.ndarray,
                           RR: np.ndarray) -> Dict[str, np.ndarray]:
    """Cantidades por indice, candidatas a explicar el ranking unario.

    Ninguna sale de los modelos: son propiedades del dato y de la definicion
    de cada indice, asi que si alguna predice c es explicacion y no circulo.
    """
    X = stack.reshape(-1, stack.shape[-1]).astype(np.float64)
    X = X[np.isfinite(X).all(axis=1)]
    if len(X) > 200000:
        X = X[np.linspace(0, len(X) - 1, 200000).astype(int)]
    Z = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-12)

    # Redundancia: R2 de predecir el indice i con los otros once.
    red = np.empty(N_IDX)
    for i in range(N_IDX):
        otros = [j for j in range(N_IDX) if j != i]
        B, *_ = np.linalg.lstsq(Z[:, otros], Z[:, i], rcond=None)
        res = Z[:, i] - Z[:, otros] @ B
        red[i] = 1.0 - res.var() / max(Z[:, i].var(), 1e-12)

    fam = familia_de_indice()
    n_bandas = np.array([len(BANDAS_INDICE[n]) for n in INDEX_NAMES],
                        dtype=float)
    frac20 = np.array([np.mean([RESOLUCION_BANDA[b] == 20
                                for b in BANDAS_INDICE[n]])
                       for n in INDEX_NAMES])
    return dict(
        redundancia=red,
        varianza=X.var(axis=0),
        asimetria=np.abs(stats.skew(Z, axis=0)),
        curtosis=stats.kurtosis(Z, axis=0),
        r_marginal_medio=np.array([np.mean([RR[i, j] for j in range(N_IDX)
                                            if j != i])
                                   for i in range(N_IDX)]),
        pc_medio=np.array([np.mean([PC[i, j] for j in range(N_IDX) if j != i])
                           for i in range(N_IDX)]),
        n_bandas=n_bandas, frac_bandas_20m=frac20,
        tam_familia=np.array([float((fam == fam[i]).sum())
                              for i in range(N_IDX)]))


def explicar_vector(v: np.ndarray, cov: Dict[str, np.ndarray]) -> List[Dict]:
    out = []
    for k, x in cov.items():
        if np.std(x) < 1e-12:
            continue
        rho = stats.spearmanr(v, x)
        out.append(dict(covariable=k, rho=float(rho.statistic),
                        p=float(rho.pvalue)))
    return sorted(out, key=lambda d: -abs(d["rho"]))


# ---------------------------------------------------------------------------
# Informe
# ---------------------------------------------------------------------------

def texto(filas: List[Dict], expl: Dict, cruce: Dict, n_perm: int) -> str:
    W = 78
    L = ["=" * W, " ARIDAD DE LA SENAL: QUE REPRODUCEN LAS 50 SEMILLAS", "",
         " A[i,j] = mu + r_i + c_j + R[i,j]  sobre las entradas fuera de la",
         " diagonal. r y c son efectos de NODO (aridad 1): cuanto emite la fila",
         " y cuanto recibe la columna. Solo R es una relacion de PAR (aridad 2).",
         "", " La columna que decide no es cuanta varianza tiene cada aridad",
         " sino cuanta REPRODUCEN las semillas, o sea el ICC de cada componente.",
         "=" * W, ""]
    L.append("  REPARTO DE VARIANZA (descriptivo, dentro de cada matriz)")
    L.append("  " + "fuente".ljust(30) + "M".rjust(4) + "fila".rjust(9)
             + "columna".rjust(9) + "PAR".rjust(9))
    for f in filas:
        L.append("  " + f["etiqueta"][:29].ljust(30) + str(f["M"]).rjust(4)
                 + ("%.1f%%" % (100 * f["frac_fila"])).rjust(9)
                 + ("%.1f%%" % (100 * f["frac_col"])).rjust(9)
                 + ("%.1f%%" % (100 * f["frac_par"])).rjust(9))
    L.append("")
    L.append("  QUE SE REPRODUCE ENTRE SEMILLAS (ICC por aridad, nulo con "
             + str(n_perm) + " permutaciones)")
    L.append("  " + "fuente".ljust(30) + "ICC fila".rjust(10)
             + "ICC col".rjust(10) + "ICC PAR".rjust(10) + "p par".rjust(9))
    for f in filas:
        L.append("  " + f["etiqueta"][:29].ljust(30)
                 + ("%.3f" % f["icc_fila"]).rjust(10)
                 + ("%.3f" % f["icc_col"]).rjust(10)
                 + ("%.3f" % f["icc_par"]).rjust(10)
                 + ("%.4f" % f["icc_par_p"]).rjust(9))
    L.append("")
    L.append("  Un ICC de columna alto con ICC de par bajo significa que las")
    L.append("  semillas coinciden en A QUE INDICE se mira y no en QUE PAR se")
    L.append("  relaciona. La fuente lleva senal, pero de aridad uno, y")
    L.append("  cualquier lectura relacional que se le pida va a fallar por")
    L.append("  construccion, no por falta de potencia estadistica.")
    L.append("")
    L.append("  Spearman medio del ranking unario entre pares de semillas,")
    L.append("  y si el efecto emisor es libre o esta atado al receptor")
    L.append("  " + "fuente".ljust(30) + "r (emite)".rjust(12)
             + "c (recibe)".rjust(12) + "corr(r,c)".rjust(12))
    for f in filas:
        L.append("  " + f["etiqueta"][:29].ljust(30)
                 + ("%+.3f" % f["spearman_medio_fila"]).rjust(12)
                 + ("%+.3f" % f["spearman_medio_col"]).rjust(12)
                 + ("%+.3f" % f["corr_r_c"]).rjust(12))
    L.append("")
    L.append("  corr(r,c) = +1 significa que el efecto emisor no es una")
    L.append("  cantidad propia: la normalizacion por fila lo deja fijado por")
    L.append("  el receptor, r = c/(N-1). En esas fuentes preguntar quien")
    L.append("  emite y quien recibe no tiene respuesta, y toda la direccion")
    L.append("  que exista tiene que vivir en la parte de par.")
    L.append("")
    L.append("=" * W)
    L.append("  PRUEBA CONSTRUCTIVA: SUSTITUTO SIN NINGUNA RELACION DE PAR")
    L.append("")
    L.append("  A_sur[i,j] = mu + r_i + c_j. Parte de par exactamente cero.")
    L.append("  Si el sustituto reproduce los descriptores de la matriz real,")
    L.append("  esos descriptores nunca midieron una relacion binaria.")
    L.append("")
    L.append("  " + "fuente".ljust(26) + "rho rank".rjust(10)
             + "ARI com3".rjust(10) + "Jaccard".rjust(9) + "b1 real".rjust(9)
             + "b1 sur".rjust(8))
    for f in filas:
        s = f["sustituto"]
        L.append("  " + f["etiqueta"][:25].ljust(26)
                 + ("%.3f" % s["spearman_por_semilla"]).rjust(10)
                 + ("%.3f" % s["ari_por_semilla"]).rjust(10)
                 + ("%.3f" % s["jaccard_por_semilla"]).rjust(9)
                 + str(s["real"]["b1_maximo"]).rjust(9)
                 + str(s["sustituto"]["b1_maximo"]).rjust(8))
    L.append("")
    L.append("  " + "fuente".ljust(26) + "curv real".rjust(11)
             + "curv sur".rjust(11) + "neg real".rjust(10)
             + "neg sur".rjust(9))
    for f in filas:
        s = f["sustituto"]
        L.append("  " + f["etiqueta"][:25].ljust(26)
                 + ("%+.3f" % s["real"]["curvatura_media"]).rjust(11)
                 + ("%+.3f" % s["sustituto"]["curvatura_media"]).rjust(11)
                 + str(s["real"]["n_aristas_negativas"]).rjust(10)
                 + str(s["sustituto"]["n_aristas_negativas"]).rjust(9))
    L.append("")
    L.append("=" * W)
    L.append("  EL EFECTO DE NODO ESTABA TAPANDO LA SENAL DE PAR?")
    L.append("")
    L.append("  Hasta ahora la comparacion contra |pc| usaba la matriz cruda,")
    L.append("  que mezcla las tres aridades. Si el 75% de su varianza es")
    L.append("  efecto sumidero, ese 75% entra en la comparacion como ruido")
    L.append("  correlacionado. Aqui se compara la parte de PAR sola.")
    L.append("")
    L.append("  " + "fuente".ljust(26) + "crudo".rjust(8) + "PAR".rjust(8)
             + "p perm".rjust(9) + "gana".rjust(7) + "p signos".rjust(10)
             + "|A-At|".rjust(9))
    for f in filas:
        v = f["verdad"]
        L.append("  " + f["etiqueta"][:25].ljust(26)
                 + ("%+.3f" % v["rho_crudo_pc"]).rjust(8)
                 + ("%+.3f" % v["rho_par_pc"]).rjust(8)
                 + ("%.4f" % v["p_perm_par"]).rjust(9)
                 + (str(v["semillas_donde_el_par_gana"]) + "/"
                    + str(v["n_semillas"])).rjust(7)
                 + ("%.4g" % v["p_signos"]).rjust(10)
                 + ("%.3f" % v["asimetria_relativa"]).rjust(9))
    L.append("")
    L.append("  crudo y PAR son Spearman contra |pc|. gana = en cuantas de las")
    L.append("  50 semillas la parte de par le gana a la matriz cruda. La")
    L.append("  ultima columna es ||A - A^T|| / ||A||: si sale 0 la fuente esta")
    L.append("  guardada simetrizada en disco y su direccion no es analizable.")
    L.append("")
    L.append("=" * W)
    L.append("  CONTRA QUE SE ALINEA EL RANKING UNARIO")
    L.append("")
    L.append("  Covariables del DATO y de la definicion de cada indice, ninguna")
    L.append("  sacada de los modelos.")
    for et, tabla in expl.items():
        L.append("")
        L.append("  " + et)
        for d in tabla[:5]:
            L.append("    " + d["covariable"].ljust(22)
                     + "rho " + ("%+.3f" % d["rho"]).rjust(7)
                     + "   p " + ("%.4f" % d["p"]).rjust(7))
    L.append("")
    L.append("=" * W)
    L.append("  COINCIDEN LOS DOS OPERADORES DE LECTURA EN LA PARTE UNARIA?")
    L.append("")
    for k, v in cruce.items():
        L.append("  " + k.ljust(52) + ("%+.3f" % v).rjust(8))
    L.append("")
    L.append("  Si el acuerdo de la parte de PAR es mayor que el de la matriz")
    L.append("  cruda, el efecto de nodo no era solo ruido: estaba tapando")
    L.append("  activamente la coincidencia entre los dos operadores, y todas")
    L.append("  las comparaciones anteriores entre atencion y ablacion")
    L.append("  subestimaban cuanto se parecen donde de verdad se comparan.")
    L.append("=" * W)
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", default=None)
    ap.add_argument("--matrices", default=None)
    ap.add_argument("--paradigmas", default=None)
    ap.add_argument("--permutaciones", type=int, default=500)
    ap.add_argument("--salida", default="resultados")
    a = ap.parse_args()

    stack = np.load(a.stack or _buscar(CANDIDATOS_STACK, False))
    PC, RR = pc_y_r(stack)
    pc_pares = np.array([PC[i, j] for i, j in PARES])
    rr_pares = np.array([RR[i, j] for i, j in PARES])
    dirm = a.matrices or _buscar(CANDIDATOS_MATRICES, True)
    dirp = a.paradigmas or _buscar(CANDIDATOS_PARADIGMAS, True)
    fuentes = [(e, p) for e, _n, _A, _s, p
               in recolectar_fuentes(dirm, dirp, PC, RR)
               if p and len(p) >= 5]
    print("  fuentes con matrices por semilla: " + str(len(fuentes)))

    filas = []
    for etiqueta, mats in fuentes:
        print("  " + etiqueta + " (" + str(len(mats)) + " semillas)")
        f = dict(etiqueta=etiqueta, M=len(mats))
        f.update(fracciones_aridad(np.mean(mats, axis=0)))
        f.update(icc_por_aridad(mats, n_perm=a.permutaciones))
        f["sustituto"] = comparar_con_sustituto(mats)
        f["verdad"] = par_contra_verdad(mats, pc_pares, rr_pares,
                                        n_perm=a.permutaciones)
        filas.append(f)

    cov = covariables_por_indice(stack, PC, RR)
    expl, vectores = {}, {}
    for f in filas:
        c = np.array(f["c_consenso"])
        vectores[f["etiqueta"]] = c
        expl[f["etiqueta"] + "  (c, lo que RECIBE)"] = explicar_vector(c, cov)

    def _busca(sub):
        for e, m in fuentes:
            if sub in e:
                return e, m
        return None, None

    cruce, acuerdo = {}, {}
    e_at, m_at = _busca("atencion rollout")
    e_ab, m_ab = _busca("dependencia por ablacion")
    if m_at is not None and m_ab is not None:
        c_at, c_ab = vectores[e_at], vectores[e_ab]
        r_at = np.array(next(f for f in filas
                             if f["etiqueta"] == e_at)["r_consenso"])
        r_ab = np.array(next(f for f in filas
                             if f["etiqueta"] == e_ab)["r_consenso"])
        cruce["Spearman c(atencion) contra c(ablacion)"] = float(
            stats.spearmanr(c_at, c_ab).statistic)
        cruce["Spearman r(atencion) contra r(ablacion)"] = float(
            stats.spearmanr(r_at, r_ab).statistic)
        ac = acuerdo_entre_operadores(m_at, m_ab, n_perm=a.permutaciones)
        cruce["Spearman de pares SIN descomponer (3 aridades juntas)"] =             ac["rho_crudo"]
        cruce["   p de permutacion de etiquetas de indice"] =             ac["p_perm_crudo"]
        cruce["Spearman de la parte de PAR (efecto de nodo fuera)"] =             ac["rho_par"]
        cruce["   p de permutacion de etiquetas de indice "] =             ac["p_perm_par"]
        cruce["Spearman de par medio, semilla a semilla"] =             ac["rho_par_por_semilla"]
        cruce["Semillas con acuerdo de par positivo"] = float(
            ac["semillas_con_acuerdo_positivo"])
        cruce["   p del test de signos"] = ac["p_signos"]
        acuerdo = ac

    txt = texto(filas, expl, cruce, a.permutaciones)
    print("\n" + txt)
    os.makedirs(a.salida, exist_ok=True)
    with open(os.path.join(a.salida, "ANALISIS_ATENCION.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(txt)
    with open(os.path.join(a.salida, "ANALISIS_ATENCION.json"), "w",
              encoding="utf-8") as fh:
        json.dump(dict(fuentes=filas, explicacion=expl, cruce=cruce,
                       acuerdo=acuerdo),
                  fh, indent=2, ensure_ascii=False)
    print("\n  Guardado: "
          + os.path.join(a.salida, "ANALISIS_ATENCION.txt"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
