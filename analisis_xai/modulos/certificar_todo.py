#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Certificacion GLOBAL: cada fuente de aristas del framework, contra verdad
conocida, con el mismo test y una nulidad por permutacion.

INDEPENDIENTE del pipeline: no lo importa, no entrena, no toca la GPU. Lee lo
que la corrida ya dejo escrito, asi que funciona con cualquier version del
pipeline y tarda segundos.

QUE RESPONDE Y POR QUE ASI
--------------------------
La tesis afirma una cosa: que del modelo se pueden extraer RELACIONES ENTRE
LAS VARIABLES. Esa afirmacion no se sostiene ni se cae con un experimento
suelto, asi que aqui se somete a TODAS las fuentes de aristas que la corrida
produce -atencion, atencion por capa, dependencia por ablacion, frecuencia
probabilistica (P1), degradacion interventiva (P2), percentil Rashomon- a la
misma pregunta y al mismo test. Si ninguna supera el azar, la afirmacion no se
sostiene, y da igual cual de los cinco paradigmas se mire. Si alguna lo supera,
queda dicho cual, cuanto, y con que limitacion.

Se incluyen dos fuentes que NO son del framework y estan a proposito:

  |pc| (verdad)      es el propio ground truth. Tiene que salir perfecto. Si no
                     sale perfecto, el bug esta en este script, no en el
                     framework, y todo lo demas de esta salida es basura.
  |r| marginal       es lo que ya da np.corrcoef sin modelo ninguno. Marca el
                     liston: una fuente del framework que no le gane no aporta
                     nada por encima de dos lineas de numpy.

EL TEST
-------
Estadistico T = puesto medio de los K pares realmente dependientes dentro del
ranking de la fuente, promediado sobre K = 3..12. Bajo es bueno; el azar da
(n_pares+1)/2 = 33.5. Se barre K en vez de fijar uno porque la distribucion de
|pc| no tiene ningun salto que justifique un corte concreto.

La significacion NO sale de una tabla: sale de permutar las 12 etiquetas de
indice y recalcular T. Esto corrige el barrido de K de forma exacta -la
nulidad se construye con el mismo barrido- y ademas respeta la estructura de
dependencia entre los 66 pares, que comparten indices y no son independientes.
Un p de tabla sobre 66 pares correlacionados esta mal calibrado; este no.

ENTRADA (se busca sola, o se pasa por argumento)
    salidas/multi_indices/indices_12.npy          stack de 12 indices
    resultados/sensitivity_K_experiment/matrices/ attention_seed_*.npy
                                                  attention_capas_seed_*.npy
                                                  dependencia_seed_*.npy
                                                  rashomon.json
    resultados/paradigms_experiment/01_probabilistic/paradigm1_results.json
    resultados/paradigms_experiment/02_interventional/paradigm2_results.json

SALIDA
    resultados/CERTIFICACION_GLOBAL.json
    resultados/CERTIFICACION_GLOBAL.txt

USO
    .venv/bin/python certificar_todo.py
    .venv/bin/python certificar_todo.py --permutaciones 5000
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

# Orden fijo de los 12 indices, igual que INDEX_NAMES en el pipeline. Se
# replica para no depender de poder importarlo.
INDEX_NAMES: List[str] = [
    "NDVI", "MARI", "ARI", "EVI", "EVI2", "NDWI",
    "NDMI", "CHL_REDEDGE", "NDII", "SAVI", "PSRI", "KNDVI",
]
N_IDX = len(INDEX_NAMES)

# Familias espectrales: que indices comparten bandas. Copiado de
# INDEX_FAMILIES del pipeline. Se usa para el diagnostico de sustitutos.
INDEX_FAMILIES: Dict[str, List[str]] = {
    "G1_NIR_ROJO":  ["NDVI", "EVI", "EVI2", "SAVI", "KNDVI"],
    "G2_RED_EDGE":  ["MARI", "ARI", "CHL_REDEDGE"],
    "G3_NIR_SWIR":  ["NDMI", "NDII"],
    "G4_VERDE_NIR": ["NDWI"],
    "G5_SENESC":    ["PSRI"],
}

# Bandas Sentinel-2 que entra en la formula de cada indice. Sirve para separar
# los pares que comparten informacion espectral de origen de los que no: una
# asimetria entre dos indices que comparten bandas puede ser un artefacto de la
# formula, mientras que entre indices sin ninguna banda comun no puede serlo.
BANDAS_INDICE: Dict[str, Tuple[str, ...]] = {
    "NDVI":        ("B4", "B8"),
    "MARI":        ("B3", "B5", "B7"),
    "ARI":         ("B3", "B5"),
    "EVI":         ("B2", "B4", "B8"),
    "EVI2":        ("B4", "B8"),
    "NDWI":        ("B3", "B8"),
    # B8A y no B8: la doc de sentinel-hub escribe la formula con B08 pero su
    # script usa B8A, y compute_12_indices() en el pipeline hace lo mismo
    # (b8a.subtract(b11)). Importa porque B8A es de 20 m y B8 de 10 m, asi que
    # el par NDMI-NDII pasa de compartir dos bandas a compartir solo B11, y
    # NDMI pasa de "mixto" a "ambos con 20 m" en el contraste de resolucion.
    "NDMI":        ("B8A", "B11"),
    "CHL_REDEDGE": ("B5", "B7"),
    "NDII":        ("B8", "B11"),
    "SAVI":        ("B4", "B8"),
    "PSRI":        ("B2", "B4", "B6"),
    "KNDVI":       ("B4", "B8"),
}

# Resolucion nativa antes del remuestreo a la malla comun. En Sentinel-2 son de
# 10 m B2, B3, B4 y B8; de 20 m B5, B6, B7, B8A, B11 y B12. Las de 20 m llegan
# a la malla de 10 m remuestreadas, lo que introduce correlacion espacial
# artificial entre pixeles vecinos: si la "variabilidad anunciada" saliera solo
# de indices que usan bandas de 20 m, seria ruido de remuestreo y no biologia.
RESOLUCION_BANDA: Dict[str, int] = {
    "B2": 10, "B3": 10, "B4": 10, "B8": 10,
    "B5": 20, "B6": 20, "B7": 20, "B8A": 20, "B11": 20, "B12": 20,
}

VAL_RATIO = 0.2

# Identidades algebraicas EXACTAS sobre este stack: KNDVI = tanh(NDVI^2) y
# MARI = ARI*B7. No son hipotesis, son verdad por construccion.
CONTROLES_POSITIVOS: List[Tuple[str, str]] = [("NDVI", "KNDVI"), ("MARI", "ARI")]

K_MIN, K_MAX = 3, 12

CANDIDATOS_STACK = [
    "salidas/multi_indices/indices_12.npy",
    "multi_indices_extraido/indices_12.npy",
    "resultados/indices_12.npy",
    "indices_12.npy",
]
CANDIDATOS_MATRICES = [
    "resultados/sensitivity_K_experiment/matrices",
    "resultados/paradigms_experiment/matrices",
    "resultados/stability_experiment/matrices",
    "resultados/stability_50_experiment/matrices",
]
CANDIDATOS_PARADIGMAS = [
    "resultados/paradigms_experiment",
    "resultados/paradigmas_experiment",
]

PARES: List[Tuple[int, int]] = [(i, j) for i in range(N_IDX)
                                for j in range(i + 1, N_IDX)]
N_PARES = len(PARES)
PUESTO_AZAR = (N_PARES + 1) / 2.0


# ---------------------------------------------------------------------------
# Verdad conocida
# ---------------------------------------------------------------------------

def pc_y_r(stack: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """|correlacion parcial| y |correlacion marginal|, sobre el tramo de train.

    La parcial es la matriz de precision normalizada: dependencia de i con j
    controlando por los otros 10. Es la que el framework dice medir. Se calcula
    solo sobre train porque darle el bloque de validacion seria regalarle
    informacion que el modelo no vio.
    """
    a = np.asarray(stack, dtype=np.float64)
    a = a[:int((1.0 - VAL_RATIO) * len(a))]
    X = a.reshape(-1, a.shape[-1])
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    C = np.corrcoef(X.T)
    Pm = np.linalg.inv(C + 1e-8 * np.eye(C.shape[0]))
    d = np.sqrt(np.diag(Pm))
    return np.abs(-Pm / np.outer(d, d)), np.abs(C)


def _dcor(x: np.ndarray, y: np.ndarray) -> float:
    """Correlacion de distancia (Szekely, Rizzo y Bakirov 2007).

    Vale 0 si y solo si las dos variables son independientes, para CUALQUIER
    forma de dependencia. La correlacion de Pearson, en cambio, vale 0 para
    dependencias no lineales perfectas (por ejemplo y = x^2 sobre datos
    centrados). Esa diferencia es todo el punto de esta seccion.
    """
    n = len(x)
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    A = a - a.mean(0)[None, :] - a.mean(1)[:, None] + a.mean()
    B = b - b.mean(0)[None, :] - b.mean(1)[:, None] + b.mean()
    dcov2 = (A * B).mean()
    dvx, dvy = (A * A).mean(), (B * B).mean()
    den = np.sqrt(dvx * dvy)
    return float(np.sqrt(max(dcov2, 0.0) / den)) if den > 0 else 0.0


def _dcor_gauss_teorico(rho: float) -> float:
    """dcor de una normal bivariada con correlacion rho (Szekely-Rizzo 2009).

    Es el valor que alcanza la correlacion de distancia cuando la dependencia
    es EXACTAMENTE lineal-gaussiana. Sirve de referencia: un dcor observado por
    encima de esto es dependencia que la correlacion no captura.
    """
    r = abs(float(rho))
    if r >= 1.0:
        return 1.0
    num = (r * np.arcsin(r) + np.sqrt(1 - r * r)
           - r * np.arcsin(r / 2) - np.sqrt(4 - r * r) + 1)
    den = 1 + np.pi / 3 - np.sqrt(3.0)
    return float(np.sqrt(max(num / den, 0.0)))


def _centrar_por_escena(a: np.ndarray) -> np.ndarray:
    """Estandariza cada canal DENTRO de cada fecha: quita nivel y escala.

    Es el blindaje contra el efecto de escena. Medido sobre estos datos, la
    heterocedasticidad entre fechas (0.37 a 0.74) es dos o tres veces la de
    dentro de fecha (0.12 a 0.31), y las diez escenas de mayor residuo caen
    siete de diez entre mayo y septiembre: invierno austral en -35.45, o sea
    nubosidad residual. Sin esto, cualquier dependencia que se mida en las
    varianzas es sobre todo el paso de nubes.

    Restar la media por fecha no basta: el efecto de escena esta en la ESCALA,
    que es justo lo que mide la heterocedasticidad. Hay que dividir por la
    desviacion de la escena tambien, y entonces lo que queda es solo la
    covariacion de variabilidad entre pixeles de una misma imagen.
    """
    b = a.reshape(len(a), -1, a.shape[-1]).astype(np.float64)
    mu = b.mean(axis=1, keepdims=True)
    sd = b.std(axis=1, keepdims=True) + 1e-12
    return ((b - mu) / sd).reshape(a.shape)


def dependencia_no_lineal(stack: np.ndarray, n_muestra: int = 1500,
                          n_repeticiones: int = 4,
                          semilla: int = 0,
                          centrar_escena: bool = False) -> np.ndarray:
    """Dependencia condicional NO LINEAL de cada par, sobre el tramo de train.

    QUE MIDE Y POR QUE IMPORTA
    --------------------------
    |pc| es la correlacion parcial gaussiana: mide la parte LINEAL de la
    dependencia entre i y j una vez descontados linealmente los otros diez.
    Es ciega a todo lo demas. Si dos indices se relacionan por una curva, un
    umbral o un producto, |pc| puede dar casi cero mientras la dependencia
    existe y es fuerte.

    Aqui se mide lo mismo sin suponer linealidad:
      1. se quita de i y de j, por regresion lineal, el efecto de los otros 10
      2. sobre esos residuos se calcula la correlacion de distancia

    El resultado captura la dependencia condicional TOTAL. Restarle la parte
    lineal deja lo que |pc| no puede ver, y eso es lo unico que un modelo
    profundo puede aportar por encima de dos lineas de numpy: si el framework
    solo reordenara |pc|, seria un rodeo caro para llegar a algo que ya se
    tiene. Sobre estos pares la pregunta deja de ser circular.

    Se submuestrea porque la correlacion de distancia es O(n^2) en memoria y
    el stack tiene ~500k filas; se promedia sobre varias submuestras para que
    el resultado no dependa de cual toco.
    """
    a = np.asarray(stack, dtype=np.float64)
    a = a[:int((1.0 - VAL_RATIO) * len(a))]
    if centrar_escena:
        a = _centrar_por_escena(a)
    X = a.reshape(-1, a.shape[-1])
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    rng = np.random.default_rng(semilla)
    obs = np.zeros(N_PARES, dtype=np.float64)
    nulo = np.zeros(N_PARES, dtype=np.float64)
    for _ in range(n_repeticiones):
        idx = rng.choice(len(X), size=min(n_muestra, len(X)), replace=False)
        M = X[idx]
        n = len(M)
        for k, (i, j) in enumerate(PARES):
            otros = [t for t in range(N_IDX) if t not in (i, j)]
            Z = np.column_stack([M[:, otros], np.ones(n)])
            coef, *_ = np.linalg.lstsq(Z, M[:, [i, j]], rcond=None)
            res = M[:, [i, j]] - Z @ coef
            # Normal scores (van der Waerden): se sustituye cada residuo por
            # el cuantil normal de su rango. Deja las dos marginales EXACTAMENTE
            # normales sin tocar el orden, o sea sin tocar la dependencia.
            # Hace falta porque los pares mas colineales dejan residuos
            # diminutos y de colas raras, y sin esto un exceso podia venir de
            # la forma de esas colas en vez de la relacion entre las dos
            # variables. Con las marginales fijadas a normal en el observado y
            # en la referencia, lo unico que puede producir exceso es la copula.
            res = np.column_stack([
                stats.norm.ppf(stats.rankdata(res[:, c]) / (n + 1))
                for c in (0, 1)])
            obs[k] += _dcor(res[:, 0], res[:, 1])
            # Referencia: una gaussiana con la MISMA correlacion parcial y el
            # MISMO n. Se simula en vez de usar solo la formula cerrada porque
            # el dcor muestral esta sesgado hacia arriba con n finito, y ese
            # sesgo hay que cancelarlo restando algo que lo tenga igual.
            rho = float(np.corrcoef(res[:, 0], res[:, 1])[0, 1])
            rho = 0.0 if not np.isfinite(rho) else np.clip(rho, -0.999, 0.999)
            g = rng.standard_normal((n, 2))
            g[:, 1] = rho * g[:, 0] + np.sqrt(1 - rho * rho) * g[:, 1]
            nulo[k] += _dcor(g[:, 0], g[:, 1])
    return np.column_stack([obs / n_repeticiones, nulo / n_repeticiones])


def pares_no_lineales(dcor_obs: np.ndarray, dcor_nulo: np.ndarray,
                      cuantos: int = 8,
                      dcor_minimo: float = 0.10) -> np.ndarray:
    """Pares donde la dependencia EXCEDE lo que explica la parte lineal.

    Criterio: exceso = dcor observado menos el dcor de una gaussiana con la
    misma correlacion parcial y el mismo n. Positivo y grande significa que
    entre los residuos hay estructura que ninguna correlacion puede expresar.

    El primer intento ordenaba por la brecha entre el puesto en |pc| y el
    puesto en dcor, y eligio pares con |r|=0.998 y dcor=0.14: la brecha de
    RANGOS premia a los pares donde ambas medidas son bajas pero una lo es un
    poco menos, que es ruido de cola, y a los muy colineales cuyos residuos
    aun comparten varianza. Con el exceso contra la referencia gaussiana esas
    dos trampas desaparecen: un par perfectamente lineal da exceso cero por
    grande que sea su correlacion.

    `dcor_minimo` descarta los pares cuya dependencia total es despreciable:
    un exceso de 0.02 sobre un dcor de 0.05 no es un hallazgo.
    """
    exceso = dcor_obs - dcor_nulo
    vivos = np.where(dcor_obs >= dcor_minimo)[0]
    if vivos.size == 0:
        return np.array([], dtype=int)
    return vivos[np.argsort(-exceso[vivos])[:cuantos]].astype(int)


def _normal_scores(v: np.ndarray) -> np.ndarray:
    return stats.norm.ppf(stats.rankdata(v) / (len(v) + 1))


def _residuos_par(X: np.ndarray, i: int, j: int) -> np.ndarray:
    """Residuos de i y j tras quitar linealmente los otros 10, en normal scores."""
    otros = [t for t in range(N_IDX) if t not in (i, j)]
    Z = np.column_stack([X[:, otros], np.ones(len(X))])
    coef, *_ = np.linalg.lstsq(Z, X[:, [i, j]], rcond=None)
    res = X[:, [i, j]] - Z @ coef
    return np.column_stack([_normal_scores(res[:, 0]),
                            _normal_scores(res[:, 1])])


def diagnostico_forma(stack: np.ndarray, sel: np.ndarray,
                      n_muestra: int = 4000, n_perm: int = 400,
                      semilla: int = 0) -> List[Dict]:
    """QUE FORMA tiene la dependencia no lineal de cada par seleccionado.

    Decir "hay dependencia no lineal" no es un hallazgo utilizable: no dice
    que hacer con ella ni permite contrastarla con el dominio. Esto la separa
    en las tres formas que puede tomar entre dos residuos con marginales ya
    normalizadas, y da un p por permutacion para cada una:

      curvatura        la MEDIA de j cambia con i de forma no lineal. Se mide
                       como el R2 que gana un ajuste cuadratico y cubico sobre
                       el lineal. Es la forma que un modelo lineal no puede
                       representar de ninguna manera.
      heterocedastici. la media es lineal pero la VARIANZA de j depende de i.
                       Se mide como la correlacion de Spearman entre |residuo
                       de i| y |residuo del ajuste de j sobre i|. Un modelo de
                       media condicional no la captura aunque acierte la media.
      dependencia de cola  la relacion es mas fuerte en los extremos que en el
                       centro. Se mide comparando la correlacion en el 20% mas
                       extremo de i contra la del 60% central.

    El p sale de permutar uno de los dos residuos, que rompe cualquier
    dependencia dejando intactas las dos marginales. Asi el p no depende de
    suponer normalidad ni de la n.
    """
    a = np.asarray(stack, dtype=np.float64)
    a = a[:int((1.0 - VAL_RATIO) * len(a))]
    X = a.reshape(-1, a.shape[-1])
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    rng = np.random.default_rng(semilla)
    idx = rng.choice(len(X), size=min(n_muestra, len(X)), replace=False)
    X = X[idx]

    def _stats(u: np.ndarray, v: np.ndarray) -> Tuple[float, float, float]:
        # curvatura: R2 extra de los terminos cuadratico y cubico
        lin = np.column_stack([u, np.ones(len(u))])
        pol = np.column_stack([u, u ** 2, u ** 3, np.ones(len(u))])
        r_lin = v - lin @ np.linalg.lstsq(lin, v, rcond=None)[0]
        r_pol = v - pol @ np.linalg.lstsq(pol, v, rcond=None)[0]
        vt = np.var(v)
        curv = float((np.var(r_lin) - np.var(r_pol)) / vt) if vt > 0 else 0.0
        # heterocedasticidad: |u| contra |residuo lineal de v|
        het = float(stats.spearmanr(np.abs(u), np.abs(r_lin)).statistic)
        # cola: correlacion en el 20% extremo de u menos la del 60% central
        q = np.quantile(np.abs(u), [0.8])
        ext, cen = np.abs(u) >= q, np.abs(u) < q
        if ext.sum() > 30 and cen.sum() > 30:
            cola = float(abs(stats.spearmanr(u[ext], v[ext]).statistic)
                         - abs(stats.spearmanr(u[cen], v[cen]).statistic))
        else:
            cola = float("nan")
        return curv, het, cola

    salida = []
    for k in sel:
        i, j = PARES[int(k)]
        res = _residuos_par(X, i, j)
        u, v = res[:, 0], res[:, 1]
        obs = _stats(u, v)
        nulo = np.empty((n_perm, 3), dtype=np.float64)
        for t in range(n_perm):
            nulo[t] = _stats(u, v[rng.permutation(len(v))])
        ps = [float((1 + int((np.abs(nulo[:, c]) >= abs(obs[c])).sum()))
                    / (1 + n_perm)) if np.isfinite(obs[c]) else float("nan")
              for c in range(3)]
        formas = [("curvatura", obs[0], ps[0]),
                  ("heterocedasticidad", obs[1], ps[1]),
                  ("dependencia de cola", obs[2], ps[2])]
        sig = [f for f in formas if np.isfinite(f[2]) and f[2] < 0.05]
        dominante = (max(sig, key=lambda f: abs(f[1]))[0] if sig
                     else "ninguna forma sale del ruido")
        salida.append(dict(
            par=f"{INDEX_NAMES[i]}-{INDEX_NAMES[j]}",
            rho_parcial=float(stats.pearsonr(u, v).statistic),
            curvatura=obs[0], p_curvatura=ps[0],
            heterocedasticidad=obs[1], p_heterocedasticidad=ps[1],
            cola=obs[2], p_cola=ps[2], forma_dominante=dominante))
    return salida


def descomponer_por_fecha(stack: np.ndarray, sel: np.ndarray,
                          fechas_millis: Optional[np.ndarray] = None,
                          por_fecha: int = 300, n_perm: int = 400,
                          semilla: int = 0) -> Tuple[List[Dict], List[Dict]]:
    """Separa la heterocedasticidad en la parte ENTRE fechas y la de DENTRO.

    LA PREGUNTA QUE DECIDE SI EL HALLAZGO VALE. Que |residuo de i| y |residuo
    de j| covaríen admite dos explicaciones muy distintas:

      entre fechas   hay escenas enteras malas -nube fina, angulo solar, humo-
                     en las que TODOS los indices se desvian a la vez. Eso es
                     un factor global de adquisicion, no una relacion entre
                     las variables, y no se puede reportar como hallazgo del
                     dominio: se reportaria un artefacto del sensor.
      dentro de fecha  dentro de una misma escena, los pixeles donde i se
                     desvia son los pixeles donde j se desvia. Eso si es
                     estructura espacial real y sobrevive a controlar la fecha.

    Se calculan las dos por separado, cada una con su nulo:
      entre  se permutan las 186 medias por fecha de una de las dos series
      dentro se permuta una serie DENTRO de cada fecha, lo que rompe la
             relacion espacial y deja intacto el efecto de escena

    El muestreo es estratificado -el mismo numero de pixeles por fecha- para
    que ninguna escena pese mas que otra en el resultado.
    """
    a = np.asarray(stack, dtype=np.float64)
    n_train = int((1.0 - VAL_RATIO) * len(a))
    a = a[:n_train]
    n_f, H, W, _ = a.shape
    X = a.reshape(-1, a.shape[-1])
    mu, sd = X.mean(0), X.std(0) + 1e-12
    rng = np.random.default_rng(semilla)
    tomar = min(por_fecha, H * W)
    idx = np.concatenate([f * H * W + rng.choice(H * W, tomar, replace=False)
                          for f in range(n_f)])
    fecha = np.repeat(np.arange(n_f), tomar)
    M = (X[idx] - mu) / sd

    def _medias(v: np.ndarray) -> np.ndarray:
        return np.array([v[fecha == f].mean() for f in range(n_f)])

    filas = []
    for k in sel:
        i, j = PARES[int(k)]
        res = _residuos_par(M, i, j)
        u, v = np.abs(res[:, 0]), np.abs(res[:, 1])
        het = float(stats.spearmanr(u, v).statistic)

        mu_u, mu_v = _medias(u), _medias(v)
        entre = float(stats.spearmanr(mu_u, mu_v).statistic)
        nulo_e = np.array([stats.spearmanr(mu_u, rng.permutation(mu_v)).statistic
                           for _ in range(n_perm)])
        p_entre = float((1 + int((np.abs(nulo_e) >= abs(entre)).sum()))
                        / (1 + n_perm))

        # Centrar por fecha quita el efecto de escena entero.
        cu = u - np.repeat(mu_u, tomar)
        cv = v - np.repeat(mu_v, tomar)
        dentro = float(stats.spearmanr(cu, cv).statistic)
        nulo_d = np.empty(n_perm)
        for t in range(n_perm):
            perm = np.concatenate([rng.permutation(tomar) + f * tomar
                                   for f in range(n_f)])
            nulo_d[t] = stats.spearmanr(cu, cv[perm]).statistic
        p_dentro = float((1 + int((np.abs(nulo_d) >= abs(dentro)).sum()))
                         / (1 + n_perm))

        if p_dentro < 0.05 and abs(dentro) >= abs(entre) * 0.5:
            veredicto = "estructura espacial real"
        elif p_entre < 0.05 and p_dentro >= 0.05:
            veredicto = "SOLO efecto de escena (artefacto de adquisicion)"
        elif p_entre < 0.05:
            veredicto = "mezcla, domina el efecto de escena"
        else:
            veredicto = "ninguna de las dos sale del ruido"
        filas.append(dict(par=f"{INDEX_NAMES[i]}-{INDEX_NAMES[j]}",
                          het_total=het, het_entre=entre, p_entre=p_entre,
                          het_dentro=dentro, p_dentro=p_dentro,
                          veredicto=veredicto))

    # Las escenas mas ruidosas, para poder mirarlas: si son cuatro fechas
    # sueltas es nubosidad; si siguen el calendario es fenologia.
    ruido = np.zeros(n_f)
    for k in sel:
        i, j = PARES[int(k)]
        res = _residuos_par(M, i, j)
        ruido += _medias(np.abs(res[:, 0])) + _medias(np.abs(res[:, 1]))
    ruido /= max(len(sel), 1)
    peores = np.argsort(-ruido)[:10]
    etiquetas = []
    for f in peores:
        if fechas_millis is not None and f < len(fechas_millis):
            import datetime as _dt
            d = _dt.datetime.utcfromtimestamp(
                int(fechas_millis[f]) / 1000).strftime("%Y-%m-%d")
        else:
            d = f"escena {int(f)}"
        etiquetas.append(dict(fecha=d, indice=int(f),
                              residuo_medio=float(ruido[f]),
                              z=float((ruido[f] - ruido.mean())
                                      / (ruido.std() + 1e-12))))
    return filas, etiquetas


def seleccion_estable(stack: np.ndarray, n_muestra: int, n_repeticiones: int,
                      n_replicas: int = 8, cuantos: int = 8,
                      frecuencia_minima: float = 0.5,
                      centrar_escena: bool = False
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Pares no lineales que sobreviven a repetir la seleccion con otra muestra.

    POR QUE HACE FALTA. `dependencia_no_lineal` estima la correlacion de
    distancia sobre una submuestra aleatoria, porque es O(n^2) y el stack tiene
    ~500k filas. La primera version elegia el top-8 de una sola estimacion, y
    al cambiar los parametros de muestreo la lista cambiaba: entre dos corridas
    de este mismo script, la fuente "atencion capa A1" paso de estar en el 14%
    de las semillas al 94%, no porque cambiara nada del modelo sino porque
    cambiaron los pares contra los que se la medía. Un subconjunto que se mueve
    con la semilla del muestreo no puede sostener ninguna conclusion.

    Aqui la seleccion se repite `n_replicas` veces con semillas distintas y se
    conservan los pares elegidos en al menos `frecuencia_minima` de ellas. Se
    devuelve tambien la frecuencia de cada par, que es el dato honesto: si
    ninguno pasa del 50%, el hallazgo no existe y hay que decirlo.
    """
    votos = np.zeros(N_PARES, dtype=float)
    for b in range(n_replicas):
        dcm = dependencia_no_lineal(stack, n_muestra=n_muestra,
                                    n_repeticiones=n_repeticiones,
                                    semilla=1000 + b,
                                    centrar_escena=centrar_escena)
        sel = pares_no_lineales(dcm[:, 0], dcm[:, 1], cuantos=cuantos)
        votos[sel] += 1
    frec = votos / n_replicas
    return np.where(frec >= frecuencia_minima)[0].astype(int), frec


def _r2_no_parametrico(x: np.ndarray, y: np.ndarray, n_bins: int = 20) -> float:
    """Fraccion de la varianza de y que explica una funcion cualquiera de x.

    Se estima por bins de cuantiles, sin suponer ninguna forma funcional. El
    sesgo por numero de bins existe pero es identico en las dos direcciones
    que se comparan (mismo n, mismos bins), asi que se cancela en la resta.
    """
    q = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    b = np.clip(np.searchsorted(q, x, side="right") - 1, 0, n_bins - 1)
    med = np.array([y[b == t].mean() if (b == t).sum() > 5 else y.mean()
                    for t in range(n_bins)])
    vy = np.var(y)
    return float(1 - np.var(y - med[b]) / vy) if vy > 0 else 0.0


def direccion_con_estabilidad(stack: np.ndarray, n_muestra: int = 4000,
                              n_replicas: int = 12, semilla: int = 0,
                              centrar_escena: bool = True,
                              mascara: Optional[np.ndarray] = None,
                              usar_dcor: bool = False
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """D media, error estandar y consistencia de signo, sobre replicas.

    POR QUE. `direccion_heterocedastica` estima d[i,j] sobre una submuestra de
    4000 pixeles y los valores salen pequeños: el mayor medido fue 0.0288, o
    sea 2.9% de varianza diferencial. Ordenar por |d| y escribir "MARI gobierna
    la variabilidad de CHL_REDEDGE" sin comprobar que ese numero se distingue
    de cero es el mismo fallo que ya aparecio en la seleccion de pares no
    lineales, donde tres de ocho resultaron ruido de muestreo.

    Aqui se repite la estimacion con `n_replicas` submuestras independientes y
    se devuelve, por par: la media, el error estandar de la media y en que
    fraccion de las replicas el signo coincide con el de la media. Una
    direccion solo se puede nombrar si el signo aguanta -consistencia alta- y
    la media supera varias veces su propio error estandar.
    """
    reps = np.stack([
        direccion_heterocedastica(stack, n_muestra=n_muestra,
                                  semilla=2000 + b,
                                  centrar_escena=centrar_escena,
                                  mascara=mascara, usar_dcor=usar_dcor)
        for b in range(n_replicas)])
    media = reps.mean(axis=0)
    se = reps.std(axis=0, ddof=1) / np.sqrt(n_replicas)
    consistencia = (np.sign(reps) == np.sign(media)[None]).mean(axis=0)
    return media, se, consistencia


def direccion_heterocedastica(stack: np.ndarray, n_muestra: int = 4000,
                              semilla: int = 0,
                              centrar_escena: bool = True,
                              mascara: Optional[np.ndarray] = None,
                              usar_dcor: bool = False) -> np.ndarray:
    """Asimetria de la dependencia en varianza, para los 66 pares.

    POR QUE ESTO NO LO DA NINGUNA CORRELACION. La correlacion marginal, la
    parcial y la de distancia son SIMETRICAS por construccion: r(i,j)=r(j,i)
    siempre. Un grafo hecho con cualquiera de ellas no puede tener direccion,
    ni siquiera en principio. Y bajo el modelo lineal-gaussiano tampoco la hay
    que encontrar: con medias lineales, la varianza explicada de i por j es
    igual a la de j por i, asi que la direccion no esta identificada.

    Pero con heterocedasticidad si lo esta. Que el NIVEL de j gobierne la
    VARIABILIDAD de i es una afirmacion distinta de la reciproca, y las dos se
    pueden medir por separado:

        d[i,j] = R2( |residuo de i|  explicado por  residuo de j )
               - R2( |residuo de j|  explicado por  residuo de i )

    Positivo significa que j predice cuanto se desvia i mejor que al reves.
    Como los pares que salieron no lineales son justamente heterocedasticos,
    aqui hay una direccion real que medir, y una matriz de atencion -que si es
    dirigida- puede acertarla o no. Ninguna correlacion puede ni intentarlo.

    Se mide con la escena ya estandarizada por fecha: sin eso la direccion
    saldria de la nubosidad, que afecta a todos los canales a la vez.
    """
    a = np.asarray(stack, dtype=np.float64)
    a = a[:int((1.0 - VAL_RATIO) * len(a))]
    # La mascara se evalua sobre los valores ORIGINALES, antes de estandarizar
    # por escena: "NDVI > 0.3" es un umbral fisico y deja de serlo en unidades
    # normalizadas por fecha.
    sel_pix = None
    if mascara is not None:
        sel_pix = mascara[:len(a)].reshape(-1)
    if centrar_escena:
        a = _centrar_por_escena(a)
    X = a.reshape(-1, a.shape[-1])
    if sel_pix is not None:
        X = X[sel_pix]
        if len(X) < 500:
            return np.zeros((N_IDX, N_IDX), dtype=np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    rng = np.random.default_rng(semilla)
    X = X[rng.choice(len(X), size=min(n_muestra, len(X)), replace=False)]
    D = np.zeros((N_IDX, N_IDX), dtype=np.float64)
    for i, j in PARES:
        res = _residuos_par(X, i, j)
        u, v = res[:, 0], res[:, 1]
        if usar_dcor:
            # Version sin suponer que la dependencia viva en la MEDIA de
            # |residuo|: la correlacion de distancia detecta cualquier forma.
            # Mas general que el R2 por bins, y bastante mas cara.
            d = _dcor(v, np.abs(u)) - _dcor(u, np.abs(v))
        else:
            d = (_r2_no_parametrico(v, np.abs(u))
                 - _r2_no_parametrico(u, np.abs(v)))
        D[i, j], D[j, i] = d, -d
    return D


def mascaras_cobertura(stack: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """Clases de cobertura, para ver en cual de ellas vive la asimetria.

    El control binario vegetacion/no-vegetacion dijo que el efecto NO es
    biologico -aguanta igual fuera de la vegetacion, |d| 0.0057 contra 0.0044-
    pero deja sin contestar de donde sale entonces. Aqui se parte el bloque de
    no-vegetacion para distinguir dos explicaciones muy distintas:

      firma espectral del suelo   el efecto vive sobre suelo desnudo, o sea es
                                  una propiedad de como se acoplan las
                                  regiones del espectro sobre material mineral
      perdida de señal            el efecto vive en agua y zonas oscuras, donde
                                  la reflectancia colapsa y los indices son
                                  cocientes de numeros pequeños: ahi cualquier
                                  asimetria puede salir de la propagacion de
                                  error, no de la superficie

    LIMITACION QUE HAY QUE DECLARAR. En disco solo estan los 12 indices, no las
    bandas crudas, asi que no se puede calcular albedo y la clase de sombra no
    se separa de la de roca o suelo oscuro. La cuarta clase se llama "oscuro o
    mezcla" a proposito: agrupa lo que queda cuando no hay ni vegetacion, ni
    agua, ni suelo con algo de verdor, y no admite una lectura fisica limpia.
    """
    i_ndvi, i_ndwi = INDEX_NAMES.index("NDVI"), INDEX_NAMES.index("NDWI")
    a = np.asarray(stack, dtype=np.float64)
    a = a[:int((1.0 - VAL_RATIO) * len(a))]
    ndvi = a[..., i_ndvi].reshape(len(a), -1)
    ndwi = a[..., i_ndwi].reshape(len(a), -1)
    agua = ndwi > 0
    veg = (ndvi > 0.3) & ~agua
    suelo = (ndvi >= 0.1) & (ndvi <= 0.3) & ~agua
    resto = ~(veg | suelo | agua)
    return [("vegetacion NDVI>0.3", veg),
            ("suelo desnudo 0.1-0.3", suelo),
            ("agua NDWI>0", agua),
            ("oscuro o mezcla", resto)]


def mascara_vegetacion(stack: np.ndarray, umbral: float = 0.3) -> np.ndarray:
    """Pixeles con vegetacion activa (NDVI > umbral), forma (n_fechas, H*W).

    Separa el fenomeno biologico del geometrico. Si el desacople entre los
    indices de antocianinas y los de verdor es estres vegetal, tiene que
    reforzarse donde hay vegetacion y desaparecer sobre suelo desnudo, agua y
    sombra, donde MARI y ARI no miden pigmento sino reflectancia de fondo.
    """
    i_ndvi = INDEX_NAMES.index("NDVI")
    a = np.asarray(stack, dtype=np.float64)
    a = a[:int((1.0 - VAL_RATIO) * len(a))]
    return (a[..., i_ndvi].reshape(len(a), -1) > umbral)


def contrastes_direccion(D: np.ndarray, se: np.ndarray, cons: np.ndarray,
                         z_min: float = 3.0,
                         cons_min: float = 0.90) -> List[Dict]:
    """Separa los 66 pares por origen espectral y comprueba si d sobrevive.

    TRES HIPOTESIS ALTERNATIVAS QUE HAY QUE DESCARTAR ANTES DE HABLAR DE
    BIOLOGIA, cada una con su particion de los pares:

    solapamiento de bandas
        Si dos indices comparten bandas (NDVI, EVI2, SAVI y KNDVI usan todos
        B4 y B8), su relacion puede venir de la formula y no del terreno. La
        pregunta que decide es si la asimetria aparece TAMBIEN en pares sin
        ninguna banda comun, como MARI (B3,B5,B7) contra NDVI (B4,B8): ahi no
        hay algebra compartida que pueda fabricarla.

    resolucion nativa
        Las bandas de 20 m llegan a la malla de 10 m remuestreadas, y eso
        correlaciona espacialmente los pixeles vecinos. Si la variabilidad
        anunciada saliera solo de indices con bandas de 20 m, seria ruido de
        remuestreo. Se comparan los pares 10-10, los mixtos y los 20-20.

    direccion del remuestreo
        Version mas fina de lo anterior: si el efecto fuera de remuestreo, el
        emisor -el que anuncia- deberia ser sistematicamente el indice con mas
        bandas de 20 m. Se cuenta cuantas veces ocurre y se compara con el 50%
        que daria el azar.

    Devuelve una fila por grupo con el numero de pares sostenibles y el |d|
    medio, para poder decir en cual de los grupos vive el efecto.
    """
    def _bandas(nombre: str) -> set:
        return set(BANDAS_INDICE.get(nombre, ()))

    def _n20(nombre: str) -> int:
        return sum(1 for b in BANDAS_INDICE.get(nombre, ())
                   if RESOLUCION_BANDA.get(b, 10) == 20)

    grupos: Dict[str, List[int]] = {
        "comparten bandas": [], "sin bandas comunes": [],
        "ambos solo 10 m": [], "mixto 10/20 m": [], "ambos con 20 m": [],
    }
    for k, (i, j) in enumerate(PARES):
        ni, nj = INDEX_NAMES[i], INDEX_NAMES[j]
        comunes = _bandas(ni) & _bandas(nj)
        grupos["comparten bandas" if comunes
               else "sin bandas comunes"].append(k)
        a20, b20 = _n20(ni) > 0, _n20(nj) > 0
        if not a20 and not b20:
            grupos["ambos solo 10 m"].append(k)
        elif a20 and b20:
            grupos["ambos con 20 m"].append(k)
        else:
            grupos["mixto 10/20 m"].append(k)

    dv = np.array([D[i, j] for i, j in PARES])
    sev = np.array([se[i, j] for i, j in PARES])
    cov = np.array([cons[i, j] for i, j in PARES])
    z = np.abs(dv) / np.maximum(sev, 1e-12)
    sostenible = (cov >= cons_min) & (z >= z_min)

    filas = []
    for nombre, ks in grupos.items():
        ks = np.array(ks, dtype=int)
        if ks.size == 0:
            continue
        filas.append(dict(grupo=nombre, n_pares=int(ks.size),
                          n_sostenibles=int(sostenible[ks].sum()),
                          frac_sostenibles=float(sostenible[ks].mean()),
                          d_medio=float(np.abs(dv[ks]).mean()),
                          z_medio=float(z[ks].mean())))

    # Direccion del remuestreo: entre los pares sostenibles, cuantas veces el
    # que anuncia tiene MAS bandas de 20 m que el anunciado.
    emisor_mas_20 = 0
    total = 0
    for k in np.where(sostenible)[0]:
        i, j = PARES[int(k)]
        # d[i,j] > 0 significa que j anuncia la variabilidad de i.
        anunciado, anuncia = ((i, j) if dv[k] > 0 else (j, i))
        da = _n20(INDEX_NAMES[anuncia]) - _n20(INDEX_NAMES[anunciado])
        if da != 0:
            total += 1
            emisor_mas_20 += int(da > 0)
    p_res = (float(stats.binomtest(emisor_mas_20, total, 0.5).pvalue)
             if total else float("nan"))
    filas.append(dict(grupo="__resampleo__", n_pares=total,
                      n_sostenibles=emisor_mas_20,
                      frac_sostenibles=(emisor_mas_20 / total if total
                                        else float("nan")),
                      d_medio=float("nan"), z_medio=p_res))
    return filas


def _parte_de_par(S: np.ndarray) -> np.ndarray:
    """Quita de una matriz antisimetrica el efecto aditivo de nodo.

    POR QUE ES OBLIGATORIO AQUI. Las matrices de atencion salen del pipeline
    renormalizadas por fila (`rollout / rollout.sum(axis=1)`), asi que cada
    fila suma 1. Con esa restriccion, si un indice j es un sumidero -muchas
    filas le dan peso alto- entonces A[i,j] es alto para casi todo i mientras
    A[j,i] es bajo, porque la fila de j reparte su unidad entre once. La
    asimetria A[i,j]-A[j,i] queda dominada por cuan sumidero es cada NODO, que
    es un ranking de indices, no una relacion entre pares.

    La misma trampa afecta al lado de la verdad: un indice intrinsecamente mas
    ruidoso tiene |residuo| mas explicable por cualquier otro.

    Toda matriz antisimetrica se descompone en S[i,j] = (s_i - s_j) + R[i,j],
    con s_i la media de la fila i y R sin efecto de nodo. Correlacionar las R
    de dos matrices mide si coinciden en QUE PAR va en que sentido, y no si
    coinciden en el ranking de nodos, que es una afirmacion mucho mas debil y
    la que se obtendria gratis de cualquier medida de centralidad.
    """
    s = S.mean(axis=1)
    return S - (s[:, None] - s[None, :])


def analisis_direccion(D_verdad: np.ndarray,
                       por_fuente: Dict[str, List[np.ndarray]],
                       n_perm: int = 5000, semilla: int = 0) -> List[Dict]:
    """Compara la asimetria de cada fuente dirigida contra la de la verdad.

    La asimetria de una fuente es A[i,j]-A[j,i]. Se correlaciona con la
    asimetria heterocedastica sobre los 66 pares, con la misma nulidad de
    permutar las 12 etiquetas que usa el resto del informe. Se reporta ademas
    en cuantas de las 50 semillas la correlacion sale positiva: si la
    direccion fuera ruido de inicializacion, saldria en la mitad.
    """
    def _vec(S):
        return np.array([S[i, j] for i, j in PARES])

    S_ver = D_verdad - D_verdad.T          # antisimetrica por construccion
    d_bruto = _vec(S_ver)
    d_par = _vec(_parte_de_par(S_ver))
    rng = np.random.default_rng(semilla)
    salida = []
    for etiqueta, mats in por_fuente.items():
        if not mats:
            continue
        A = np.mean(mats, axis=0)
        S = A - A.T
        if np.allclose(S, 0):
            continue
        a_bruto, a_par = _vec(S), _vec(_parte_de_par(S))
        rho = float(stats.spearmanr(a_bruto, d_bruto).statistic)
        rho_par = float(stats.spearmanr(a_par, d_par).statistic)
        nulo = np.empty(n_perm)
        nulo_par = np.empty(n_perm)
        for t in range(n_perm):
            pi = rng.permutation(N_IDX)
            Sp = S[np.ix_(pi, pi)]
            nulo[t] = stats.spearmanr(_vec(Sp), d_bruto).statistic
            nulo_par[t] = stats.spearmanr(_vec(_parte_de_par(Sp)),
                                          d_par).statistic
        p = float((1 + int((np.abs(nulo) >= abs(rho)).sum())) / (1 + n_perm))
        p_par = float((1 + int((np.abs(nulo_par) >= abs(rho_par)).sum()))
                      / (1 + n_perm))
        por_sem = np.array([stats.spearmanr(_vec(m - m.T),
                                            d_bruto).statistic for m in mats])
        por_sem_par = np.array([
            stats.spearmanr(_vec(_parte_de_par(m - m.T)),
                            d_par).statistic for m in mats])
        n_pos = int((por_sem > 0).sum())
        n_pos_par = int((por_sem_par > 0).sum())
        salida.append(dict(
            etiqueta=etiqueta, rho=rho, p=p,
            rho_semilla_mediana=float(np.median(por_sem)),
            frac_semillas_positivo=float(n_pos / len(por_sem)),
            p_signos=float(stats.binomtest(n_pos, len(por_sem), 0.5,
                                           alternative="greater").pvalue),
            rho_par=rho_par, p_par=p_par,
            rho_par_semilla_mediana=float(np.median(por_sem_par)),
            frac_semillas_par_positivo=float(n_pos_par / len(por_sem_par)),
            p_signos_par=float(stats.binomtest(
                n_pos_par, len(por_sem_par), 0.5,
                alternative="greater").pvalue),
            asimetria_media=float(np.mean(np.abs(a_bruto)))))
    return sorted(salida, key=lambda r: -abs(r["rho_par"]))


def fuerza_por_par(A: np.ndarray) -> np.ndarray:
    """max(A[i,j], A[j,i]) sobre los 66 pares no dirigidos.

    Se simetriza porque la correlacion parcial no tiene direccion: comparar un
    grafo dirigido contra un baseline simetrico sin declararlo lo penalizaria
    por algo que el baseline ni siquiera puede expresar.
    """
    return np.array([max(A[i, j], A[j, i]) for i, j in PARES])


def puestos(fr: np.ndarray) -> np.ndarray:
    """Puesto 1..66 de cada par (1 = el mas fuerte segun la fuente).

    Los empates reciben el puesto MEDIO del bloque, no el orden en que
    aparezcan en la lista de pares. Varias fuentes puntuan con valores muy
    repetidos -jaccard_drop da el mismo numero a los 66 pares, la frecuencia
    de P1 solo toma unos pocos valores enteros- y con desempate por posicion
    el ranking lo acabaria decidiendo el orden de INDEX_NAMES, que es
    arbitrario y ademas esta correlacionado con las familias espectrales. Eso
    fabricaba tanto puestos como sesgos que no estaban en la medida.
    """
    return stats.rankdata(-fr, method="average")


def familia_de_indice() -> np.ndarray:
    fam = np.full(N_IDX, -1, dtype=int)
    for f, (_n, miembros) in enumerate(INDEX_FAMILIES.items()):
        for nm in miembros:
            if nm in INDEX_NAMES:
                fam[INDEX_NAMES.index(nm)] = f
    sig = len(INDEX_FAMILIES)
    for i in range(N_IDX):
        if fam[i] < 0:
            fam[i] = sig
            sig += 1
    return fam


def n_sustitutos_por_par(fam: np.ndarray) -> np.ndarray:
    """Cuantos indices distintos de i y j pueden sustituirlos en su familia.

    Es la cantidad que hace fallar a la ablacion: si al ocultar j quedan tres
    hermanos suyos visibles, el modelo reconstruye i igual de bien y el delta
    de MSE que mide la ablacion sale casi cero aunque la relacion i-j exista.
    """
    tam = np.array([(fam == f).sum() for f in fam])
    out = np.empty(N_PARES, dtype=np.float64)
    for k, (i, j) in enumerate(PARES):
        if fam[i] == fam[j]:
            out[k] = tam[i] - 2          # hermanos que quedan aparte de i y j
        else:
            out[k] = (tam[i] - 1) + (tam[j] - 1)
    return out


# ---------------------------------------------------------------------------
# Estadistico y nulidad por permutacion
# ---------------------------------------------------------------------------

def estadistico_T(fr: np.ndarray, orden_pc: np.ndarray) -> float:
    """Puesto medio de los K pares mas dependientes, promediado sobre K.

    Bajo = la fuente pone arriba lo que de verdad existe. El azar da 33.5.
    No devuelve ningun p: la significacion la da la permutacion, que repite
    este mismo barrido de K sobre etiquetas barajadas y por tanto lo corrige
    de forma exacta.
    """
    p = puestos(fr)
    return float(np.mean([p[orden_pc[:K]].mean()
                          for K in range(K_MIN, K_MAX + 1)]))


def pares_solo_condicionales(puesto_pc: np.ndarray, puesto_rr: np.ndarray,
                             tope_pc: int = 12,
                             dif_min: int = 15) -> np.ndarray:
    """Pares fuertes en |pc| que la correlacion marginal NO encuentra.

    POR QUE ESTE SUBCONJUNTO. De los seis pares mas dependientes, dos
    (EVI2-SAVI, NDVI-KNDVI) tambien estan en el top-5 de |r|: cualquier metodo
    que copie np.corrcoef los "encuentra" sin aportar nada. La capacidad que la
    tesis reclama solo se puede medir donde las dos verdades DISCREPAN. Aqui se
    seleccionan los pares con |pc| entre los `tope_pc` primeros cuyo puesto en
    |r| es al menos `dif_min` peor. Sobre estos 12 indices salen ocho, y entre
    ellos estan MARI-ARI (verdad 3, marginal 50) y NDWI-PSRI (6 contra 45).

    Los cortes son arbitrarios, pero no eligen el resultado: la nulidad por
    permutacion se calcula sobre EL MISMO subconjunto, asi que un corte que
    favoreciera a una fuente favoreceria igual a sus permutaciones.
    """
    return np.array([k for k in range(len(puesto_pc))
                     if puesto_pc[k] <= tope_pc
                     and puesto_rr[k] - puesto_pc[k] >= dif_min], dtype=int)


def estadistico_sub(fr: np.ndarray, sel: np.ndarray) -> float:
    """Puesto medio de un subconjunto fijo de pares. Azar = (n_pares+1)/2."""
    if sel.size == 0:
        return float("nan")
    return float(puestos(fr)[sel].mean())


def nulidad_permutando(A: np.ndarray, orden_pc: np.ndarray,
                       n_perm: int, rng: np.random.Generator,
                       subconjuntos: Optional[Dict[str, np.ndarray]] = None
                       ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """T bajo permutacion de las 12 etiquetas de indice.

    Se permuta la matriz de la fuente, no el ground truth: es equivalente y
    deja fija la verdad para todas las fuentes. La hipotesis nula es que la
    fuente no sabe cual indice es cual; conserva la distribucion de fuerzas y
    la estructura de solapamiento entre los 66 pares, que es justo lo que un
    p de tabla ignora.
    """
    subs = subconjuntos or {}
    out = np.empty(n_perm, dtype=np.float64)
    out_sub = {k: np.empty(n_perm, dtype=np.float64) for k in subs}
    for t in range(n_perm):
        pi = rng.permutation(N_IDX)
        fr = fuerza_por_par(A[np.ix_(pi, pi)])
        out[t] = estadistico_T(fr, orden_pc)
        for k, sel in subs.items():
            out_sub[k][t] = estadistico_sub(fr, sel)
    return out, out_sub


def _parcial_spearman(x: np.ndarray, y: np.ndarray,
                      z: np.ndarray) -> Tuple[float, float]:
    """Spearman parcial de x con y controlando z, por residuos de rangos."""
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))

    def resid(a, b):
        b1 = np.column_stack([b, np.ones_like(b)])
        coef, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ coef

    ex, ey = resid(rx, rz), resid(ry, rz)
    r = stats.pearsonr(ex, ey)
    return float(r.statistic), float(r.pvalue)


# ---------------------------------------------------------------------------
# Certificacion de una fuente
# ---------------------------------------------------------------------------

def certificar_fuente(A: np.ndarray, etiqueta: str, nota: str,
                      pc: np.ndarray, rr: np.ndarray, orden_pc: np.ndarray,
                      puesto_pc: np.ndarray, n_sust: np.ndarray,
                      n_perm: int, rng: np.random.Generator,
                      n_semillas: int = 1,
                      por_semilla: Optional[List[np.ndarray]] = None,
                      T_liston: float = float("nan"),
                      subconjuntos: Optional[Dict[str, np.ndarray]] = None
                      ) -> Dict:
    fr = fuerza_por_par(A)
    pu = puestos(fr)
    subs = {k: v for k, v in (subconjuntos or {}).items() if v.size}

    T = estadistico_T(fr, orden_pc)
    nulo, nulo_sub = nulidad_permutando(A, orden_pc, n_perm, rng, subs)

    # Cada subconjunto es una pregunta distinta sobre la misma fuente, y todas
    # se contestan igual: puesto medio de esos pares contra la nulidad de
    # permutar las etiquetas, y -si hay matrices por semilla- en cuantos
    # modelos queda por debajo del azar. Ver los constructores de cada uno
    # (pares_solo_condicionales, pares_no_lineales) para que mide cada cual.
    sub_res: Dict[str, Dict] = {}
    for k, sel in subs.items():
        Tk = estadistico_sub(fr, sel)
        sub_res[k] = dict(
            n=int(sel.size), T=Tk,
            p=float((1 + int((nulo_sub[k] <= Tk).sum())) / (1 + n_perm)),
            T_semilla_mediana=float("nan"),
            frac_semillas_mejor_azar=float("nan"),
            p_signos=float("nan"), q_fdr=float("nan"))

    if por_semilla:
        for k, sel in subs.items():
            vals = np.array([estadistico_sub(fuerza_por_par(m), sel)
                             for m in por_semilla])
            n_mej = int((vals < PUESTO_AZAR).sum())
            sub_res[k]["T_semilla_mediana"] = float(np.median(vals))
            sub_res[k]["frac_semillas_mejor_azar"] = float(n_mej / len(vals))
            sub_res[k]["p_signos"] = float(stats.binomtest(
                n_mej, len(vals), 0.5, alternative="greater").pvalue)

    # +1 en numerador y denominador: con un numero finito de permutaciones el p
    # no puede ser 0, y sin la correccion se reportaria un imposible.
    p_perm = float((1 + int((nulo <= T).sum())) / (1 + n_perm))

    s_cond = stats.spearmanr(fr, pc)
    s_marg = stats.spearmanr(fr, rr)

    # Descriptivo, no inferencial: el rank-sum por K se deja porque es legible,
    # pero el p que vale es el de permutacion.
    enriq = []
    for K in range(K_MIN, K_MAX + 1):
        sel = orden_pc[:K]
        r_sel = pu[sel]
        r_res = np.delete(pu, sel)
        try:
            p_rs = float(stats.mannwhitneyu(r_sel, r_res,
                                            alternative="less").pvalue)
        except Exception:
            p_rs = float("nan")
        enriq.append(dict(K=int(K), puesto_medio=float(r_sel.mean()),
                          p_rank_sum=p_rs,
                          en_top_K=int((r_sel <= K).sum()),
                          esperado=float(K * K / N_PARES)))

    controles = []
    idx_par = {p: k for k, p in enumerate(PARES)}
    umbral = max(1, N_PARES // 3)
    for na, nb in CONTROLES_POSITIVOS:
        i, j = INDEX_NAMES.index(na), INDEX_NAMES.index(nb)
        k = idx_par[(min(i, j), max(i, j))]
        controles.append(dict(par=f"{na}-{nb}", pc=float(pc[k]),
                              puesto_verdad=int(puesto_pc[k]),
                              puesto_framework=int(pu[k]),
                              ok=bool(pu[k] <= umbral)))
    n_ok = sum(1 for c in controles if c["ok"])

    top_reales = [dict(par=f"{INDEX_NAMES[PARES[k][0]]}-{INDEX_NAMES[PARES[k][1]]}",
                       pc=float(pc[k]), puesto=int(pu[k]),
                       n_sustitutos=int(n_sust[k]))
                  for k in orden_pc[:6]]

    # Diagnostico de sustitutos: el puesto que da la fuente, contra cuantos
    # hermanos de familia tiene el par, CONTROLANDO por |pc|. Aisla el sesgo
    # que NO se explica por la dependencia real. Las dos direcciones importan
    # y significan cosas distintas:
    #   rho > 0  la fuente manda ABAJO los pares con sustitutos: se le tapa la
    #            relacion cuando el modelo tiene con que reemplazar el canal
    #            (enmascaramiento por redundancia, el fallo esperado de la
    #            ablacion).
    #   rho < 0  la fuente manda ARRIBA los pares con sustitutos: prefiere el
    #            racimo colineal grande por ser colineal, no por depender.
    #            Es la firma del atajo, medida sin mirar |r|.
    rho_sust, p_sust = _parcial_spearman(pu, n_sust, pc)
    if p_sust < 0.05 and rho_sust > 0:
        sesgo = ("pierde los pares que tienen sustitutos "
                 "(enmascaramiento por redundancia)")
    elif p_sust < 0.05 and rho_sust < 0:
        sesgo = ("favorece los pares del racimo colineal grande "
                 "(firma del atajo)")
    else:
        sesgo = "sin sesgo detectable por numero de sustitutos"

    # Empates: un ranking con muchos valores repetidos se ordena por indice de
    # par, no por la fuente. Conviene saberlo antes de creerse un puesto.
    _v, _c = np.unique(fr, return_counts=True)
    frac_empates = float(_c[_c > 1].sum() / len(fr))

    # Comparacion contra el liston, semilla a semilla. La pregunta que de
    # verdad importa no es "supera el azar" -np.corrcoef tambien lo supera-
    # sino "supera a np.corrcoef". Con las matrices por semilla eso se puede
    # contestar: se calcula T en cada modelo y se cuenta en cuantos queda por
    # debajo del liston. Un test de signos da el p. Sin las matrices por
    # semilla (fuentes que solo dejan un agregado) queda en nan, y entonces la
    # diferencia contra el liston es un numero sin barra de error.
    T_semillas: List[float] = []
    if por_semilla:
        for m in por_semilla:
            T_semillas.append(estadistico_T(fuerza_por_par(m), orden_pc))
    if T_semillas and np.isfinite(T_liston):
        arr = np.array(T_semillas)
        n_gana = int((arr < T_liston).sum())
        frac_gana_liston = float(n_gana / len(arr))
        p_signos = float(stats.binomtest(n_gana, len(arr), 0.5,
                                         alternative="greater").pvalue)
        T_semilla_mediana = float(np.median(arr))
    else:
        frac_gana_liston = float("nan")
        p_signos = float("nan")
        T_semilla_mediana = float("nan")

    if p_perm < 0.05 and n_ok == len(controles):
        veredicto = "ENCUENTRA RELACIONES REALES"
    elif p_perm < 0.05:
        veredicto = (f"ENCUENTRA RELACIONES REALES pero falla "
                     f"{len(controles) - n_ok}/{len(controles)} controles")
    else:
        veredicto = "NO SE DISTINGUE DEL AZAR"

    if s_marg.statistic > s_cond.statistic and s_marg.pvalue < 0.05:
        forma = "el ranking entero sigue la correlacion marginal"
    elif s_cond.statistic > s_marg.statistic and s_cond.pvalue < 0.05:
        forma = "el ranking entero sigue la dependencia condicional"
    else:
        forma = "el ranking entero no sigue a ninguna de las dos"

    return dict(etiqueta=etiqueta, nota=nota, n_semillas=int(n_semillas),
                T=T, T_azar=PUESTO_AZAR,
                T_nulo_media=float(nulo.mean()),
                T_nulo_p05=float(np.percentile(nulo, 5)),
                p_permutacion=p_perm, n_permutaciones=int(n_perm),
                rho_condicional=float(s_cond.statistic),
                p_condicional=float(s_cond.pvalue),
                rho_marginal=float(s_marg.statistic),
                p_marginal=float(s_marg.pvalue),
                forma=forma, veredicto=veredicto,
                controles=controles, n_controles_ok=n_ok,
                n_controles=len(controles), umbral_control=int(umbral),
                top_pares_reales=top_reales, enriquecimiento=enriq,
                rho_sustitutos=rho_sust, p_sustitutos=p_sust, sesgo=sesgo,
                frac_empates=frac_empates,
                T_liston=float(T_liston),
                T_semilla_mediana=T_semilla_mediana,
                frac_semillas_gana_liston=frac_gana_liston,
                p_signos_vs_liston=p_signos,
                sub=sub_res,
                q_fdr=float("nan"))


# ---------------------------------------------------------------------------
# Carga de fuentes
# ---------------------------------------------------------------------------

def _buscar(cands: Sequence[str], es_dir: bool) -> Optional[str]:
    for c in cands:
        if (os.path.isdir(c) if es_dir else os.path.exists(c)):
            return c
    return None


def _cargar_npy(dirm: str, patron: str) -> List[np.ndarray]:
    def _n(p):
        d = re.findall(r"(\d+)", os.path.basename(p))
        return int(d[-1]) if d else 0
    out = []
    for f in sorted(glob.glob(os.path.join(dirm, patron)), key=_n):
        try:
            m = np.load(f)
        except Exception:
            continue
        if m.ndim >= 2 and m.shape[-1] == N_IDX and m.shape[-2] == N_IDX:
            out.append(np.asarray(m, dtype=np.float64))
    return out


def _matriz_de_aristas(edges: Sequence[dict], campo: str,
                       signo: float = 1.0) -> Optional[np.ndarray]:
    """Lista de aristas dirigidas {i, j, <campo>} a matriz 12x12."""
    A = np.zeros((N_IDX, N_IDX), dtype=np.float64)
    n = 0
    for e in edges:
        try:
            v = float(e[campo])
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(v):
            continue
        i, j = int(e["i"]), int(e["j"])
        if 0 <= i < N_IDX and 0 <= j < N_IDX:
            A[i, j] = signo * v
            n += 1
    # Con menos de la mitad de las aristas puntuadas el ranking seria casi todo
    # empates a cero y el puesto lo decidiria el orden de la lista, no la
    # fuente. Mejor descartarla que certificar ruido de ordenacion.
    return A if n >= N_PARES else None


def _json(path: str) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def recolectar_fuentes(dirm: Optional[str], dirp: Optional[str],
                       PC: np.ndarray, RR: np.ndarray
                       ) -> List[Tuple[str, str, np.ndarray, int,
                                       Optional[List[np.ndarray]]]]:
    """(etiqueta, nota, matriz agregada, n_semillas, matrices por semilla).

    El ultimo campo es None en las fuentes que solo dejan un agregado en disco
    (P1, P2, Rashomon): de esas no se puede medir variabilidad entre modelos, y
    la comparacion contra el liston se queda sin barra de error.
    """
    F: List[Tuple[str, str, np.ndarray, int, Optional[List[np.ndarray]]]] = []

    # Referencias. Van primero a proposito: la primera valida este script y la
    # segunda marca el liston que hay que superar para aportar algo.
    F.append(("|pc| (VERDAD, control del script)",
              "es el ground truth; si no sale perfecto el bug esta aqui",
              PC.copy(), 0, None))
    F.append(("|r| marginal (SIN MODELO, liston)",
              "dos lineas de numpy; el framework tiene que ganarle",
              RR.copy(), 0, None))

    if dirm:
        at = _cargar_npy(dirm, "attention_seed_*.npy")
        if len(at) >= 2:
            F.append(("atencion rollout", "pesos de atencion agregados",
                      np.mean(at, axis=0), len(at), at))
        capas = _cargar_npy(dirm, "attention_capas_seed_*.npy")
        if len(capas) >= 2 and capas[0].ndim == 3:
            for c in range(capas[0].shape[0]):
                por_c = [m[c] for m in capas]
                F.append((f"atencion capa A{c + 1}",
                          "pesos de una sola capa, sin rollout",
                          np.mean(por_c, axis=0), len(capas), por_c))
            if capas[0].shape[0] == 2:
                # DESCOMPOSICION DEL ROLLOUT. Con residual_fraction=0.5 y L=2,
                # attention_rollout da 0.25(I + A1 + A2 + A2A1). Las capas
                # sueltas salieron en el azar o por debajo, asi que la ventaja
                # que muestra el rollout tiene que venir de uno de los otros
                # dos terminos. Estas dos fuentes lo separan:
                #   A2A1  el unico camino de dos saltos, sin identidad
                #   A1+A2 la parte aditiva, sin identidad ni composicion
                # Si ninguna conserva la ventaja, la pone la I: el rollout
                # estaria puntuando la diagonal, no una relacion aprendida.
                prod = [m[1] @ m[0] for m in capas]
                F.append(("atencion A2*A1 (sin residual)",
                          "composicion de las dos capas, sin el termino identidad",
                          np.mean(prod, axis=0), len(capas), prod))
                suma = [m[0] + m[1] for m in capas]
                F.append(("atencion A1+A2 (aditiva)",
                          "parte aditiva del rollout, sin identidad",
                          np.mean(suma, axis=0), len(capas), suma))
        cab = _cargar_npy(dirm, "attention_cabezas_seed_*.npy")
        if len(cab) >= 2 and cab[0].ndim == 4:
            # Promediar sobre cabezas puede borrar una relacion que vive en UNA
            # cabeza especializada. El maximo sobre cabezas es la lectura mas
            # generosa posible de la atencion: si tampoco encuentra nada, el
            # problema no es el promedio.
            mx = [m.max(axis=(0, 1)) for m in cab]
            F.append(("atencion max sobre cabezas",
                      "la cabeza mas fuerte de cada arista, sin promediar",
                      np.mean(mx, axis=0), len(cab), mx))
        dep = _cargar_npy(dirm, "dependencia_seed_*.npy")
        if len(dep) >= 2:
            F.append(("dependencia por ablacion",
                      "interviene la entrada, no lee pesos",
                      np.mean(dep, axis=0), len(dep), dep))
        depc = _cargar_npy(dirm, "dependencia_cruda_seed_*.npy")
        if len(depc) >= 2:
            F.append(("dependencia por ablacion (cruda)",
                      "la misma sin normalizar por fila",
                      np.mean(depc, axis=0), len(depc), depc))
        rash = _json(os.path.join(dirm, "rashomon.json"))
        if rash and isinstance(rash.get("aristas"), list):
            A = _matriz_de_aristas(rash["aristas"], "pct_media")
            if A is not None:
                F.append(("Rashomon (percentil de fila)",
                          "importancia agregada sobre el conjunto Rashomon",
                          A, int(rash.get("M", 0)), None))

    if dirp:
        p1 = _json(os.path.join(dirp, "01_probabilistic",
                                "paradigm1_results.json"))
        if p1 and isinstance(p1.get("edges"), list):
            A = _matriz_de_aristas(p1["edges"], "phi")
            if A is not None:
                F.append(("P1 probabilistico (frecuencia)",
                          "en cuantas semillas sobrevive la arista",
                          A, int(p1.get("M", 0)), None))
            # p_value bajo = arista mas creible, asi que se invierte el signo
            # para que en todas las fuentes "mas alto" signifique "mas fuerte".
            A = _matriz_de_aristas(p1["edges"], "p_value", signo=-1.0)
            if A is not None:
                F.append(("P1 probabilistico (-p_value)",
                          "significacion contra el modelo nulo",
                          A, int(p1.get("M", 0)), None))
        p2 = _json(os.path.join(dirp, "02_interventional",
                                "paradigm2_results.json"))
        if p2 and isinstance(p2.get("ablation"), list):
            for campo, nota in (
                    ("auc_degradation",
                     "area de la curva de degradacion al debilitar la arista"),
                    ("max_degradation", "peor degradacion alcanzada"),
                    ("jaccard_drop", "cuanto cambia el grafo al intervenir")):
                A = _matriz_de_aristas(p2["ablation"], campo)
                if A is not None:
                    F.append((f"P2 interventivo ({campo})", nota,
                              A, int(p2.get("M", 0)), None))
        p3 = _json(os.path.join(dirp, "03_geometric_topological",
                                "paradigm3_results.json"))
        top = (p3 or {}).get("topologia_persistente", {})
        if isinstance(top.get("edges"), list):
            for campo, nota in (
                    ("brecha", "brecha topologica de la arista"),
                    ("peso", "peso de la arista en el grafo de consenso")):
                A = _matriz_de_aristas(top["edges"], campo)
                if A is not None:
                    F.append((f"P3 topologico ({campo})", nota,
                              A, int(p3.get("M", 0)), None))
        p4 = _json(os.path.join(dirp, "04_triangulation",
                                "triangulation_results.json"))
        if p4 and isinstance(p4.get("profiles"), list):
            A = _matriz_de_aristas(p4["profiles"], "te_brecha")
            if A is not None:
                F.append(("P4 triangulacion (te_brecha)",
                          "brecha del criterio de estabilidad topologica",
                          A, 0, None))
    return F


def fuentes_de_consenso(cert_previa: List[Dict],
                        matrices: Dict[str, np.ndarray],
                        empates_max: float = 0.30
                        ) -> List[Tuple[str, str, np.ndarray, int,
                                        Optional[List[np.ndarray]]]]:
    """Fusiona fuentes por rango medio y devuelve el consenso como una fuente mas.

    POR QUE. Ninguna fuente suelta sobrevive la correccion por multiplicidad,
    pero eso no cierra la pregunta: puede haber senal repartida que ninguna
    capta entera y un consenso si. Es ademas la unica manera de usar los cinco
    paradigmas para lo que se construyeron, en vez de quedarse con el mejor.

    Se fusiona por RANGO y no por valor porque las fuentes viven en escalas
    incomparables (pesos de softmax, delta de MSE, percentiles, brechas
    topologicas); promediar sus valores dejaria que la de rango dinamico mas
    grande decida sola.

    Las agrupaciones se fijan por COMO se obtiene la evidencia, no por como
    puntuaron -elegirlas mirando el resultado seria escoger el ganador y luego
    testearlo-:
      todas         cualquier fuente con ranking real
      intervencion  las que perturban la entrada y miden el efecto
      pesos         las que leen pesos de atencion
    Se excluyen las fuentes con demasiados empates: meter un ranking decidido
    por el desempate solo aporta ruido al promedio.

    La nulidad no necesita nada especial: permutar las etiquetas de la matriz
    de consenso equivale a permutarlas en cada fuente y volver a fusionar,
    porque la fusion es par a par y la permutacion solo remapea pares.
    """
    grupos = {
        "todas": lambda e: True,
        "intervencion": lambda e: ("ablacion" in e or "P2 interventivo" in e),
        "pesos de atencion": lambda e: e.startswith("atencion"),
    }
    usables = [r for r in cert_previa
               if not r["etiqueta"].startswith(("|pc|", "|r|"))
               and r["frac_empates"] <= empates_max
               and r["etiqueta"] in matrices]
    # Fuera los duplicados: P3 (brecha) y P4 (te_brecha) resultaron ser la
    # misma cantidad con dos nombres, y meterla dos veces en el promedio le
    # daria doble voto a un solo experimento.
    vistos: List[Tuple[str, np.ndarray]] = []
    unicas = []
    for r in usables:
        f = fuerza_por_par(matrices[r["etiqueta"]])
        gemela = next((e for e, g in vistos
                       if abs(stats.spearmanr(f, g).statistic) > 0.999), None)
        if gemela is not None:
            print(f"    consenso: {r['etiqueta']} es duplicado de {gemela}, "
                  f"se excluye")
            continue
        vistos.append((r["etiqueta"], f))
        unicas.append(r)
    usables = unicas
    salida = []
    for nombre, pertenece in grupos.items():
        miembros = [r["etiqueta"] for r in usables if pertenece(r["etiqueta"])]
        if len(miembros) < 2:
            continue
        rangos = np.mean([puestos(fuerza_por_par(matrices[m]))
                          for m in miembros], axis=0)
        # De vuelta a matriz simetrica: rango bajo = arista fuerte, y las
        # fuentes se puntuan con "mas alto es mas fuerte".
        A = np.zeros((N_IDX, N_IDX), dtype=np.float64)
        for k, (i, j) in enumerate(PARES):
            A[i, j] = A[j, i] = -rangos[k]
        salida.append((f"CONSENSO {nombre} ({len(miembros)})",
                       "rango medio de " + ", ".join(miembros),
                       A, len(miembros), None))
    return salida


# ---------------------------------------------------------------------------
# Informe
# ---------------------------------------------------------------------------

def texto(cert: List[Dict], n_perm: int,
          info_nolin: Optional[List[Dict]] = None,
          formas: Optional[List[Dict]] = None,
          fechas_filas: Optional[List[Dict]] = None,
          escenas: Optional[List[Dict]] = None,
          direccion: Optional[List[Dict]] = None,
          D_dir: Optional[np.ndarray] = None,
          D_se: Optional[np.ndarray] = None,
          D_cons: Optional[np.ndarray] = None,
          contrastes: Optional[List[Dict]] = None,
          submedidas: Optional[List[Tuple]] = None) -> str:
    L: List[str] = []
    W = 78
    L.append("=" * W)
    L.append(" CERTIFICACION GLOBAL DEL FRAMEWORK")
    L.append("")
    L.append(" Pregunta unica: alguna fuente de aristas del framework encuentra")
    L.append(" relaciones REALES entre las 12 variables, por encima del azar?")
    L.append("")
    L.append(f" T = puesto medio de los pares realmente dependientes "
             f"(K={K_MIN}..{K_MAX}).")
    L.append(f" Azar = {PUESTO_AZAR:.1f} sobre {N_PARES} pares. Mas bajo es mejor.")
    L.append(f" p = permutacion de las 12 etiquetas, {n_perm} repeticiones.")
    L.append("=" * W)
    L.append("")
    nombres_sub = list(cert[0].get("sub", {}).keys()) if cert else []
    cab = f"  {'fuente':<34}{'T':>7}{'p perm':>9}{'q FDR':>8}{'>liston':>9}"
    for k in nombres_sub:
        cab += f"{k[:9]:>11}{'%sem':>7}"
    cab += f"{'ctrl':>7}{'emp':>6}"
    L.append(cab)
    for r in cert:
        q = ("     -" if not np.isfinite(r.get("q_fdr", float("nan")))
             else f"{r['q_fdr']:>8.4f}")
        g = ("        -" if not np.isfinite(r["frac_semillas_gana_liston"])
             else f"{r['frac_semillas_gana_liston'] * 100:>8.0f}%")
        fila = (f"  {r['etiqueta']:<34}{r['T']:>7.1f}"
                f"{r['p_permutacion']:>9.4f}{q}{g}")
        for k in nombres_sub:
            d = r.get("sub", {}).get(k, {})
            _t = d.get("T", float("nan"))
            _f = d.get("frac_semillas_mejor_azar", float("nan"))
            fila += ("          -" if not np.isfinite(_t)
                     else f"{_t:>7.1f}{'*' if d.get('p', 1) < 0.05 else ' '}   ")
            fila += ("      -" if not np.isfinite(_f)
                     else f"{_f * 100:>6.0f}%")
        fila += (f"{r['n_controles_ok']:>5}/{r['n_controles']}"
                 f"{r['frac_empates'] * 100:>5.0f}%")
        L.append(fila)
    L.append("")
    L.append(f"  azar = {PUESTO_AZAR:.1f}   ·   liston |r| sin modelo = "
             f"{cert[0].get('T_liston', float('nan')):.1f}")
    L.append("  '>liston' = en que porcentaje de las 50 semillas la fuente")
    L.append("  bate a np.corrcoef. Vacio = la fuente solo deja un agregado en")
    L.append("  disco y no se puede medir esa variabilidad.")
    for k in nombres_sub:
        _n = next((r["sub"][k]["n"] for r in cert if r.get("sub", {}).get(k)), 0)
        L.append(f"  '{k}' = puesto medio de {_n} pares. * = p<0.05 por")
        L.append("  permutacion. %sem = en que fraccion de las 50 semillas esa")
        L.append("  fuente queda por debajo del azar en ese subconjunto.")
    L.append("")

    reales = [r for r in cert if not r["etiqueta"].startswith(("|pc|", "|r|"))]
    ganan = [r for r in reales if r["p_permutacion"] < 0.05]
    verdad = next((r for r in cert if r["etiqueta"].startswith("|pc|")), None)
    liston = next((r for r in cert if r["etiqueta"].startswith("|r|")), None)

    EMPATES_MAX = 0.30

    if info_nolin:
        L.append("-" * W)
        L.append(" LA PREGUNTA NO CIRCULAR")
        L.append("")
        L.append("  Comparar el framework contra |pc| tiene un techo obvio: si")
        L.append("  lo que se quiere es el ranking de |pc|, se calcula |pc| y")
        L.append("  listo. Bajo ese criterio ningun modelo puede aportar nada.")
        L.append("")
        L.append("  Pero |pc| es la correlacion parcial GAUSSIANA LINEAL. Es")
        L.append("  ciega a la dependencia condicional no lineal. Estos son los")
        L.append("  pares con dependencia condicional alta medida sin suponer")
        L.append("  linealidad (correlacion de distancia sobre los residuos de")
        L.append("  quitar los otros 10) y baja en |pc|: existen, y ninguna")
        L.append("  correlacion los encuentra.")
        L.append("")
        L.append(f"  {'par':<18}{'dcor':>7}{'gauss':>7}{'exceso':>8}"
                 f"{'|pc|':>8}{'puesto':>8}{'|r|':>8}{'puesto':>8}{'frec':>6}")
        for t in info_nolin:
            L.append(f"  {t['par']:<18}{t['dcor']:>7.3f}"
                     f"{t['dcor_gauss']:>7.3f}{t['exceso']:>+8.3f}"
                     f"{t['pc']:>8.3f}{t['puesto_pc']:>6}/{N_PARES}"
                     f"{t['r']:>8.3f}{t['puesto_r']:>6}/{N_PARES}"
                     f"{t.get('frecuencia', float('nan')) * 100:>6.0f}%")
        if formas:
            L.append("")
            L.append("  QUE FORMA TIENE. Permutando uno de los dos residuos")
            L.append("  (rompe la dependencia, deja las marginales intactas):")
            L.append(f"  {'par':<18}{'rho parc':>9}{'curvat':>8}{'p':>7}"
                     f"{'hetero':>8}{'p':>7}{'cola':>7}{'p':>7}  forma")
            for f in formas:
                L.append(f"  {f['par']:<18}{f['rho_parcial']:>+9.3f}"
                         f"{f['curvatura']:>8.4f}{f['p_curvatura']:>7.3f}"
                         f"{f['heterocedasticidad']:>+8.3f}"
                         f"{f['p_heterocedasticidad']:>7.3f}"
                         f"{f['cola']:>+7.3f}{f['p_cola']:>7.3f}"
                         f"  {f['forma_dominante']}")
            if fechas_filas:
                L.append("")
                L.append("  ES RELACION O ES LA ESCENA. La heterocedasticidad")
                L.append("  separada en la parte de ENTRE fechas (escenas malas")
                L.append("  enteras: nube, angulo solar) y la de DENTRO de cada")
                L.append("  fecha (estructura espacial, ya controlada la escena):")
                L.append(f"  {'par':<18}{'total':>7}{'entre':>7}{'p':>7}"
                         f"{'dentro':>8}{'p':>7}  veredicto")
                for f in fechas_filas:
                    L.append(f"  {f['par']:<18}{f['het_total']:>+7.3f}"
                             f"{f['het_entre']:>+7.3f}{f['p_entre']:>7.3f}"
                             f"{f['het_dentro']:>+8.3f}{f['p_dentro']:>7.3f}"
                             f"  {f['veredicto']}")
            if escenas:
                L.append("")
                L.append("  Escenas con mayor residuo medio (z sobre las 148 de")
                L.append("  train). Pocas y sueltas apunta a nubosidad; seguir el")
                L.append("  calendario apunta a fenologia:")
                for e in escenas:
                    L.append(f"    {e['fecha']:<12} residuo {e['residuo_medio']:.3f}"
                             f"   z={e['z']:+.2f}")
            L.append("  curvat = R2 que ganan los terminos cuadratico y cubico")
            L.append("  sobre el lineal. hetero = Spearman entre |residuo de i|")
            L.append("  y |residuo de j sobre i|: si es alto, lo que depende es")
            L.append("  la VARIANZA, no la media. cola = cuanto mas fuerte es la")
            L.append("  relacion en el 20% extremo que en el 60% central.")
            L.append("")
        L.append("  frec = en que fraccion de las replicas de submuestreo")
        L.append("  entra el par en el top-8. Solo se certifican los que pasan")
        L.append("  del 50%: una seleccion que se mueve con la semilla del")
        L.append("  muestreo no puede sostener ninguna conclusion.")
        L.append("  dcor = dependencia condicional total (sin suponer forma).")
        L.append("  gauss = la que tendria una normal con la misma correlacion")
        L.append("  parcial y el mismo n. exceso = lo que no explica ninguna")
        L.append("  correlacion, ni marginal ni parcial.")
        L.append("")
        L.append("  Si una fuente los pone arriba, eso NO se puede obtener con")
        L.append("  np.corrcoef ni con la correlacion parcial, y es lo unico de")
        L.append("  esta salida que justifica haber entrenado 50 modelos.")
        for _clave, _titulo in (
                ("no lineal", "SIN blindar (incluye el efecto de escena)"),
                ("no lin blindado",
                 "BLINDADO: escena estandarizada por fecha antes de medir")):
            _nl = [r for r in cert
                   if np.isfinite(r.get("sub", {}).get(_clave, {})
                                  .get("T", float("nan")))]
            if not _nl:
                continue
            L.append("")
            L.append(f"  {_titulo}")
            L.append(f"  Quien los encuentra (puesto medio, azar "
                     f"{PUESTO_AZAR:.1f}):")
            _fw = [r for r in _nl
                   if not r["etiqueta"].startswith(("|pc|", "|r|"))]
            for r in sorted(_fw, key=lambda x: x["sub"][_clave]["T"])[:8]:
                d = r["sub"][_clave]
                _s = ("" if not np.isfinite(d["frac_semillas_mejor_azar"])
                      else f"  {d['frac_semillas_mejor_azar'] * 100:.0f}% "
                           f"de semillas p={d['p_signos']:.2g}")
                _q = d.get("q_fdr", float("nan"))
                _qs = "" if not np.isfinite(_q) else f"  q={_q:.4f}"
                L.append(f"    {r['etiqueta']:<36}{d['T']:>7.1f}  "
                         f"p={d['p']:.4f}{_qs}{_s}")
            for r in _nl:
                if r["etiqueta"].startswith(("|pc|", "|r|")):
                    d = r["sub"][_clave]
                    L.append(f"    {r['etiqueta']:<36}{d['T']:>7.1f}  "
                             f"p={d['p']:.4f}   (referencia)")
        L.append("-" * W)

    if direccion:
        L.append("-" * W)
        L.append(" DIRECCION: LO QUE NINGUNA CORRELACION PUEDE EXPRESAR")
        L.append("")
        L.append("  La correlacion marginal, la parcial y la de distancia son")
        L.append("  simetricas por construccion. Un grafo hecho con cualquiera")
        L.append("  de ellas no tiene direccion ni puede tenerla. Bajo el modelo")
        L.append("  lineal-gaussiano tampoco hay direccion que encontrar: con")
        L.append("  medias lineales, la varianza de i explicada por j iguala a la")
        L.append("  de j por i, y el problema no esta identificado.")
        L.append("")
        L.append("  Con heterocedasticidad si lo esta. Que el nivel de j gobierne")
        L.append("  la VARIABILIDAD de i no es lo mismo que la reciproca:")
        L.append("    d[i,j] = R2(|res i| ~ res j) - R2(|res j| ~ res i)")
        L.append("  Una matriz de atencion es dirigida y puede acertarla o no.")
        L.append("")
        L.append("  BRUTO incluye el efecto de nodo (que indice es sumidero);")
        L.append("  DE PAR lo descuenta y es el unico que habla de pares. Las")
        L.append("  matrices de atencion vienen renormalizadas por fila, asi que")
        L.append("  el bruto esta contaminado por construccion: leer solo DE PAR.")
        L.append(f"  {'fuente':<32}{'bruto':>8}{'%sem':>6}"
                 f"{'DE PAR':>9}{'p':>8}{'%sem':>6}{'p signos':>10}")
        for r in direccion:
            L.append(f"  {r['etiqueta']:<32}{r['rho']:>+8.3f}"
                     f"{r['frac_semillas_positivo'] * 100:>5.0f}%"
                     f"{r['rho_par']:>+9.3f}{r['p_par']:>8.4f}"
                     f"{r['frac_semillas_par_positivo'] * 100:>5.0f}%"
                     f"{r['p_signos_par']:>10.2g}")
        L.append("  rho = Spearman entre la asimetria de la fuente (A[i,j]-A[j,i])")
        L.append("  y la asimetria heterocedastica de los datos, sobre los 66")
        L.append("  pares. %sem+ = en cuantas de las 50 semillas sale positiva;")
        L.append("  si la direccion fuera ruido de inicializacion, seria 50%.")
        if D_dir is not None and D_se is not None and D_cons is not None:
            dv = np.array([D_dir[i, j] for i, j in PARES])
            sev = np.array([D_se[i, j] for i, j in PARES])
            cov = np.array([D_cons[i, j] for i, j in PARES])
            z = np.abs(dv) / np.maximum(sev, 1e-12)
            # Solo se nombra una direccion si el signo aguanta el remuestreo y
            # la media supera 3 errores estandar. Ordenar por |d| a secas
            # premia al ruido de las submuestras pequeñas.
            ok = np.where((cov >= 0.90) & (z >= 3))[0]
            L.append("")
            if ok.size == 0:
                L.append("  NINGUN par tiene una direccion que aguante el")
                L.append("  remuestreo (signo consistente >=90% y |d| >= 3 EE).")
                L.append("  La asimetria existe en agregado pero no se puede")
                L.append("  atribuir a ningun par concreto.")
            else:
                L.append(f"  Pares con direccion sostenible ({ok.size} de "
                         f"{N_PARES}); el resto no aguanta el remuestreo:")
                for k in ok[np.argsort(-np.abs(dv[ok]))][:12]:
                    i, j = PARES[int(k)]
                    a_, b_ = ((i, j) if dv[k] > 0 else (j, i))
                    L.append(f"    {INDEX_NAMES[b_]:>12} anuncia la "
                             f"variabilidad de {INDEX_NAMES[a_]:<12} "
                             f"d={abs(dv[k]):.4f} +-{sev[k]:.4f}  "
                             f"z={z[k]:.1f}  signo {cov[k] * 100:.0f}%")
                L.append("")
                L.append("  'anuncia' y no 'causa': d dice cual de los dos sirve")
                L.append("  de INDICADOR DE REGIMEN -saber su valor te dice si el")
                L.append("  otro va a estar bien predicho por el resto o se va a")
                L.append("  desbandar-, no que uno produzca al otro.")
        if contrastes:
            L.append("")
            L.append("  DE DONDE VIENE LA ASIMETRIA. Tres hipotesis alternativas")
            L.append("  a la biologica, cada una con su particion de los pares:")
            L.append(f"  {'grupo':<22}{'pares':>7}{'sosten.':>9}{'frac':>7}"
                     f"{'|d| medio':>11}{'z medio':>9}")
            for f in contrastes:
                if f["grupo"] == "__resampleo__":
                    continue
                L.append(f"  {f['grupo']:<22}{f['n_pares']:>7}"
                         f"{f['n_sostenibles']:>9}"
                         f"{f['frac_sostenibles'] * 100:>6.0f}%"
                         f"{f['d_medio']:>11.4f}{f['z_medio']:>9.1f}")
            _r = next((f for f in contrastes
                       if f["grupo"] == "__resampleo__"), None)
            if _r and _r["n_pares"]:
                L.append("")
                L.append(f"  Remuestreo: en {_r['n_sostenibles']} de "
                         f"{_r['n_pares']} pares sostenibles con distinta")
                L.append("  composicion de resolucion, el que ANUNCIA es el que "
                         "tiene mas")
                L.append(f"  bandas de 20 m ({_r['frac_sostenibles'] * 100:.0f}%,"
                         f" binomial p={_r['z_medio']:.4f}). Si el efecto fuera")
                L.append("  ruido de remuestreo, esa fraccion tenderia a 100%.")
            _sc = next((f for f in contrastes
                        if f["grupo"] == "sin bandas comunes"), None)
            _cc = next((f for f in contrastes
                        if f["grupo"] == "comparten bandas"), None)
            if _sc and _cc:
                # Test exacto sobre la tabla 2x2 de pares sostenibles: sin el,
                # el contraste 46% contra 18% es un conteo sin incertidumbre.
                tabla = [[_sc["n_sostenibles"],
                          _sc["n_pares"] - _sc["n_sostenibles"]],
                         [_cc["n_sostenibles"],
                          _cc["n_pares"] - _cc["n_sostenibles"]]]
                odds, p_f = stats.fisher_exact(tabla)
                L.append("")
                L.append(f"  Fisher exacto sobre esa tabla 2x2 "
                         f"({tabla[0][0]}/{_sc['n_pares']} sin solapamiento "
                         f"contra {tabla[1][0]}/{_cc['n_pares']} con):")
                L.append(f"    odds ratio {odds:.2f}   p = {p_f:.4f}")
                L.append("  Es evidencia, no prueba: sin corregir por la")
                L.append("  cantidad de contrastes que lleva este informe.")
            L.append("")
            L.append("  Lo que decide: 'sin bandas comunes' son pares que NO")
            L.append("  comparten ninguna banda Sentinel-2, asi que su asimetria")
            L.append("  no puede salir de la formula. Si el efecto vive tambien")
            L.append("  ahi, es acoplamiento entre regiones del espectro.")
        if submedidas:
            L.append("")
            L.append("  ROBUSTEZ DE LA DIRECCION a como y donde se mide:")
            L.append(f"  {'medida':<26}{'sostenibles':>14}{'|d| medio':>11}")
            for nombre, Dm, sem, cm in submedidas:
                dvm = np.array([Dm[i, j] for i, j in PARES])
                sm = np.array([sem[i, j] for i, j in PARES])
                cmv = np.array([cm[i, j] for i, j in PARES])
                zz = np.abs(dvm) / np.maximum(sm, 1e-12)
                ok = int(((cmv >= 0.90) & (zz >= 3)).sum())
                L.append(f"  {nombre:<26}{ok:>8} de {N_PARES}"
                         f"{np.abs(dvm).mean():>11.4f}")
            L.append("")
            L.append("  UN FENOMENO O DOS. Concordancia entre las matrices D de")
            L.append("  cada clase, sobre los 66 pares. Alta: es la misma")
            L.append("  asimetria y la baja reflectancia solo la amplifica.")
            L.append("  Baja: hay dos dinamicas distintas superpuestas, y solo")
            L.append("  la que vive en vegetacion y suelo es reportable como")
            L.append("  propiedad espectral de la superficie.")
            _et = [n for n, *_ in submedidas]
            _vs = [np.array([Dm[i, j] for i, j in PARES])
                   for _, Dm, _s, _c in submedidas]
            L.append("  " + " " * 26 + "".join(f"{n[:11]:>13}" for n in _et))
            for x, nx in zip(_vs, _et):
                fila = f"  {nx:<26}"
                for y in _vs:
                    fila += f"{stats.spearmanr(x, y).statistic:>+13.3f}"
                L.append(fila)
            L.append("")
            L.append("  Aguantar igual dentro y fuera de la vegetacion")
            L.append("  descarta que sea estres vegetal. Que se concentre en")
            L.append("  agua o en 'oscuro o mezcla' apuntaria a propagacion de")
            L.append("  error donde la reflectancia colapsa; que viva en suelo")
            L.append("  desnudo apuntaria a la firma espectral del material.")
            L.append("  Sin las bandas crudas no se puede aislar la sombra: la")
            L.append("  clase 'oscuro o mezcla' es un cajon, no una superficie.")
        L.append("-" * W)

    L.append("-" * W)
    L.append(" CONCLUSION")
    if verdad is not None and verdad["p_permutacion"] >= 0.05:
        L.append("  ATENCION: el control del script (|pc| contra si mismo) NO")
        L.append("  supera el azar. Eso es imposible si el codigo esta bien.")
        L.append("  No leas nada mas de esta salida hasta arreglarlo.")
        L.append("-" * W)
        return "\n".join(L)
    inservibles = [r for r in reales if r["frac_empates"] > EMPATES_MAX]
    if inservibles:
        L.append(f"  {len(inservibles)} fuentes no son certificables: mas del "
                 f"{EMPATES_MAX * 100:.0f}% de sus")
        L.append("  pares empatan, asi que su ranking lo decide el desempate y")
        L.append("  no la medida. Cualquier puesto suyo, y cualquier sesgo que")
        L.append("  se les mida, es artefacto de ordenacion:")
        for r in inservibles:
            L.append(f"    {r['etiqueta']:<36} "
                     f"{r['frac_empates'] * 100:.0f}% empates")
        L.append("")
    sobreviven = [r for r in ganan
                  if np.isfinite(r.get("q_fdr", float("nan")))
                  and r["q_fdr"] < 0.05]
    if not ganan:
        L.append("  Ninguna fuente del framework supera el azar. La afirmacion")
        L.append("  'del modelo se pueden extraer relaciones entre variables' NO")
        L.append("  se sostiene con esta corrida, en ninguno de los paradigmas.")
    elif not sobreviven:
        L.append(f"  {len(ganan)}/{len(reales)} fuentes superan el azar con p")
        L.append("  sin corregir, pero NINGUNA sobrevive a la correccion por")
        L.append("  multiplicidad (Benjamini-Hochberg sobre las "
                 f"{len(reales)} fuentes")
        L.append("  probadas). Se probaron muchos rankings contra la misma")
        L.append("  verdad; con alfa 0.05 se espera "
                 f"{len(reales) * 0.05:.1f} fuentes significativas solo")
        L.append("  por probar. Los p sin corregir mas bajos fueron:")
        for r in sorted(ganan, key=lambda x: x["p_permutacion"]):
            L.append(f"    {r['etiqueta']:<36} T={r['T']:.1f}  "
                     f"p={r['p_permutacion']:.4f}  q={r['q_fdr']:.3f}  "
                     f"ctrl {r['n_controles_ok']}/{r['n_controles']}")
        L.append("")
        L.append("  Ese test tiene un techo de potencia que conviene decir: la")
        L.append("  nulidad baraja 12 etiquetas, o sea 12 unidades")
        L.append("  intercambiables, asi que su p no puede bajar mucho por")
        L.append("  construccion, y encima se le aplica FDR sobre "
                 f"{len(reales)} fuentes.")
        L.append("  No pasarlo no prueba que no haya efecto.")
        _k = "solo condicional"
        _pot = [r for r in reales
                if np.isfinite(r.get("sub", {}).get(_k, {})
                               .get("p_signos", float("nan")))]
        if _pot:
            L.append("")
            L.append("  El test con potencia para esta pregunta es el de")
            L.append("  solo-condicionales POR SEMILLA: 50 modelos dan la")
            L.append("  variabilidad que a 8 pares les falta. Ordenado por")
            L.append("  tamano del efecto (puestos de ventaja sobre el azar):")
            for r in sorted(_pot,
                            key=lambda x: x["sub"][_k]["T_semilla_mediana"]):
                d = r["sub"][_k]
                L.append(f"    {r['etiqueta']:<36} "
                         f"mediana {d['T_semilla_mediana']:.1f} "
                         f"({PUESTO_AZAR - d['T_semilla_mediana']:+.1f} "
                         f"puestos)  {d['frac_semillas_mejor_azar'] * 100:.0f}% "
                         f"de semillas  p={d['p_signos']:.2g}")
            L.append("  Con 50 semillas el test de signos detecta sesgos muy")
            L.append("  pequenos, asi que lo que separa a las fuentes aqui NO es")
            L.append("  el p sino el tamano del efecto: mirar la columna de")
            L.append("  puestos de ventaja, no cual pasa 0.05.")
        L.append("")
        L.append("  Conclusion honesta: esta corrida no prueba la afirmacion")
        L.append("  general de que el framework recupera relaciones entre las")
        L.append("  variables -ningun ranking global sobrevive la correccion-, y")
        L.append("  documenta un resultado negativo bien caracterizado (el sesgo")
        L.append("  por racimo colineal). Lo que si queda medido es cuanta")
        L.append("  ventaja sobre el azar conserva cada fuente donde la")
        L.append("  correlacion marginal falla, y con cuanta reproducibilidad.")
    else:
        L.append(f"  {len(sobreviven)}/{len(reales)} fuentes sobreviven a la "
                 f"correccion por multiplicidad (q<0.05):")
        for r in sorted(sobreviven, key=lambda x: x["T"]):
            L.append(f"    {r['etiqueta']:<36} T={r['T']:.1f}  "
                     f"p={r['p_permutacion']:.4f}  q={r['q_fdr']:.4f}  "
                     f"ctrl {r['n_controles_ok']}/{r['n_controles']}")
        solo_crudo = [r for r in ganan if r not in sobreviven]
        if solo_crudo:
            L.append("  Superan el azar sin corregir pero NO la correccion "
                     "(no reportables):")
            for r in sorted(solo_crudo, key=lambda x: x["p_permutacion"]):
                L.append(f"    {r['etiqueta']:<36} p={r['p_permutacion']:.4f}"
                         f"  q={r['q_fdr']:.4f}")
        if liston is not None:
            mejores = [r for r in ganan if r["T"] < liston["T"]]
            L.append("")
            L.append(f"  El liston sin modelo (|r| marginal) esta en "
                     f"T={liston['T']:.1f}.")
            if mejores:
                L.append("  Le ganan: " + ", ".join(r["etiqueta"]
                                                    for r in mejores))
                L.append("  Esas son las unicas que aportan algo por encima de")
                L.append("  np.corrcoef.")
            else:
                L.append("  NINGUNA fuente del framework le gana. Aunque superen")
                L.append("  el azar, no aportan nada sobre dos lineas de numpy.")
        ctrl_ok = [r for r in ganan if r["n_controles_ok"] == r["n_controles"]]
        L.append("")
        if ctrl_ok:
            L.append("  Con TODOS los controles positivos recuperados: "
                     + ", ".join(r["etiqueta"] for r in ctrl_ok))
        else:
            L.append("  Ninguna de las que supera el azar recupera los dos")
            L.append("  controles positivos. Encuentran relaciones reales, pero")
            L.append("  se pierden identidades que son verdad por construccion:")
            L.append("  el alcance de lo que se puede afirmar queda acotado por")
            L.append("  el diagnostico de sustitutos de cada fuente.")
    # Solo cuentan las fuentes con ranking real: en una con 100% de empates el
    # "sesgo" seria el orden de la lista de pares, no la medida.
    sesgados = [r for r in reales
                if r["p_sustitutos"] < 0.05 and r["rho_sustitutos"] < 0
                and r["frac_empates"] <= EMPATES_MAX]
    if sesgados:
        L.append("")
        L.append(f"  {len(sesgados)}/{len(reales)} fuentes prefieren el racimo")
        L.append("  colineal grande incluso descontando la dependencia real:")
        for r in sesgados:
            L.append(f"    {r['etiqueta']:<36} rho={r['rho_sustitutos']:+.3f}")
        L.append("  Es la firma del atajo medida sin mirar |r|, y es coherente")
        L.append("  con que casi todas sigan la correlacion marginal.")
    L.append("-" * W)

    for r in cert:
        L.append("")
        L.append(f"  --- {r['etiqueta']}  ({r['n_semillas']} semillas) ---")
        L.append(f"  {r['nota']}")
        L.append(f"  VEREDICTO: {r['veredicto']}")
        L.append(f"  T={r['T']:.2f}   nulo {r['T_nulo_media']:.2f} "
                 f"(percentil 5 = {r['T_nulo_p05']:.2f})   "
                 f"p={r['p_permutacion']:.4f}")
        if np.isfinite(r.get("q_fdr", float("nan"))):
            L.append(f"  q FDR sobre las fuentes probadas: {r['q_fdr']:.4f}")
        for k, d in r.get("sub", {}).items():
            if not np.isfinite(d.get("T", float("nan"))):
                continue
            _q = d.get("q_fdr", float("nan"))
            _qs = "" if not np.isfinite(_q) else f"  q={_q:.4f}"
            L.append(f"  [{k}] {d['n']} pares: puesto medio {d['T']:.1f} "
                     f"(azar {PUESTO_AZAR:.1f})  p={d['p']:.4f}{_qs}")
            if np.isfinite(d.get("frac_semillas_mejor_azar", float("nan"))):
                L.append(f"      por semilla: mediana "
                         f"{d['T_semilla_mediana']:.1f}, mejor que el azar en "
                         f"{d['frac_semillas_mejor_azar'] * 100:.0f}% de las "
                         f"semillas (signos p={d['p_signos']:.4g})")
        if np.isfinite(r["frac_semillas_gana_liston"]):
            L.append(f"  contra el liston ({r['T_liston']:.1f}): T mediano por "
                     f"semilla {r['T_semilla_mediana']:.1f}, le gana en "
                     f"{r['frac_semillas_gana_liston'] * 100:.0f}% de las "
                     f"semillas (test de signos p="
                     f"{r['p_signos_vs_liston']:.4f})")
        L.append(f"  Spearman global: condicional {r['rho_condicional']:+.3f} "
                 f"(p={r['p_condicional']:.3f})  ·  marginal "
                 f"{r['rho_marginal']:+.3f} (p={r['p_marginal']:.3f})")
        L.append(f"  {r['forma']}")
        L.append("  controles positivos (identidades exactas):")
        for c in r["controles"]:
            L.append(f"    {c['par']:<14} |pc|={c['pc']:.3f}   verdad "
                     f"{c['puesto_verdad']:>2}/{N_PARES}   fuente "
                     f"{c['puesto_framework']:>2}/{N_PARES}   "
                     f"[{'OK' if c['ok'] else 'NO'}]")
        L.append(f"  {'par real':<16}{'|pc|':>7}{'puesto':>10}{'sustitutos':>12}")
        for t in r["top_pares_reales"]:
            L.append(f"  {t['par']:<16}{t['pc']:>7.3f}"
                     f"{t['puesto']:>7}/{N_PARES}{t['n_sustitutos']:>12}")
        L.append(f"  sustitutos (controlando |pc|): rho="
                 f"{r['rho_sustitutos']:+.3f} p={r['p_sustitutos']:.4f}")
        L.append(f"    {r['sesgo']}")
        if r["frac_empates"] > 0.2:
            L.append(f"    CUIDADO: {r['frac_empates'] * 100:.0f}% de los pares "
                     f"empatan; el puesto lo decide el desempate, no la fuente")
    L.append("")
    L.append("  COMO LEERLO:")
    L.append("   p perm < 0.05  -> la fuente pone arriba los pares que de verdad")
    L.append("          existen, mas de lo que consigue barajar las etiquetas.")
    L.append("          Es el unico p de esta salida que esta calibrado: corrige")
    L.append("          el barrido de K y la dependencia entre los 66 pares.")
    L.append("   q FDR < 0.05 -> ademas sobrevive a haber probado once")
    L.append("          rankings distintos contra la misma verdad. Es el")
    L.append("          criterio para reportar algo como hallazgo.")
    L.append("   >liston -> porcentaje de semillas en que la fuente bate a")
    L.append("          np.corrcoef. Superar el azar no basta: el liston")
    L.append("          tambien lo supera.")
    L.append("   empates -> por encima del 30% el ranking lo decide el")
    L.append("          desempate y la fuente no dice nada.")
    L.append("   T solo -> lo mismo restringido a los pares que la correlacion")
    L.append("          marginal NO encuentra. Es la prueba especifica de la")
    L.append("          tesis: ahi copiar np.corrcoef no sirve de nada.")
    L.append("   T      -> puesto medio de los pares reales. 33.5 es azar puro.")
    L.append("   ctrl   -> identidades exactas (KNDVI=tanh(NDVI^2), MARI=ARI*B7)")
    L.append("          recuperadas en el tercio superior. Una fuente que no las")
    L.append("          encuentre no puede pedir que se le crean las demas.")
    L.append("   sustitutos -> sesgo por numero de hermanos de familia, ya")
    L.append("          descontado el |pc| real. rho>0: la fuente pierde los")
    L.append("          pares que tienen sustitutos (redundancia los tapa).")
    L.append("          rho<0: la fuente prefiere el racimo colineal grande por")
    L.append("          ser colineal. Las dos cosas acotan donde se le cree.")
    L.append("=" * W)
    return "\n".join(L)


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Certificacion global de todas las fuentes de aristas.")
    ap.add_argument("--stack", default=None)
    ap.add_argument("--matrices", default=None)
    ap.add_argument("--paradigmas", default=None)
    ap.add_argument("--salida", default="resultados")
    ap.add_argument("--permutaciones", type=int, default=2000)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--muestra-dcor", type=int, default=1500,
                    dest="muestra_dcor")
    ap.add_argument("--repeticiones-dcor", type=int, default=4,
                    dest="repeticiones_dcor")
    ap.add_argument("--replicas-seleccion", type=int, default=8,
                    dest="replicas_seleccion")
    ap.add_argument("--replicas-direccion", type=int, default=12,
                    dest="replicas_direccion")
    ap.add_argument("--muestra-dcor-direccion", type=int, default=4000,
                    dest="muestra_dcor_direccion")
    a = ap.parse_args()

    stack_p = a.stack or _buscar(CANDIDATOS_STACK, False)
    if not stack_p:
        print("ERROR: no encuentro el stack de indices. Pasa --stack ruta.npy")
        return 1
    stack = np.load(stack_p)
    print(f"  stack: {stack_p}  {stack.shape}")
    if stack.shape[-1] != N_IDX:
        print(f"ERROR: el stack tiene {stack.shape[-1]} canales, "
              f"esperaba {N_IDX}")
        return 1

    PC, RR = pc_y_r(stack)
    pc = np.array([PC[i, j] for i, j in PARES])
    rr = np.array([RR[i, j] for i, j in PARES])
    orden_pc = np.argsort(-pc, kind="stable")
    puesto_pc = puestos(pc)
    n_sust = n_sustitutos_por_par(familia_de_indice())

    dirm = a.matrices or _buscar(CANDIDATOS_MATRICES, True)
    dirp = a.paradigmas or _buscar(CANDIDATOS_PARADIGMAS, True)
    print(f"  matrices:   {dirm}")
    print(f"  paradigmas: {dirp}")

    fuentes = recolectar_fuentes(dirm, dirp, PC, RR)
    print(f"  fuentes certificables: {len(fuentes)}")
    if len(fuentes) <= 2:
        print("  Solo estan las dos referencias: no hay ninguna salida del")
        print("  framework en esas rutas. Revisa --matrices y --paradigmas.")

    # El liston se calcula antes del bucle porque cada fuente lo necesita para
    # la comparacion por semilla.
    T_liston = estadistico_T(fuerza_por_par(RR), orden_pc)

    print("  midiendo dependencia condicional NO LINEAL (correlacion de "
          "distancia sobre residuos)...")
    _dcm = dependencia_no_lineal(stack, n_muestra=a.muestra_dcor,
                                 n_repeticiones=a.repeticiones_dcor)
    dc, dc_nulo = _dcm[:, 0], _dcm[:, 1]
    exceso = dc - dc_nulo
    sel_nolin, frec_nolin = seleccion_estable(
        stack, a.muestra_dcor, a.repeticiones_dcor,
        n_replicas=a.replicas_seleccion)
    print(f"  estabilidad de la seleccion (frecuencia sobre "
          f"{a.replicas_seleccion} replicas):")
    for k in np.argsort(-frec_nolin)[:12]:
        if frec_nolin[k] > 0:
            print(f"    {INDEX_NAMES[PARES[k][0]]}-{INDEX_NAMES[PARES[k][1]]:<14}"
                  f" {frec_nolin[k] * 100:.0f}%")
    puesto_ex = puestos(exceso)
    print(f"  pares no lineales ({sel_nolin.size}): " + ", ".join(
        f"{INDEX_NAMES[PARES[k][0]]}-{INDEX_NAMES[PARES[k][1]]}"
        for k in sel_nolin))
    info_nolin = [dict(
        par=f"{INDEX_NAMES[PARES[k][0]]}-{INDEX_NAMES[PARES[k][1]]}",
        dcor=float(dc[k]), dcor_gauss=float(dc_nulo[k]),
        exceso=float(exceso[k]), puesto_exceso=int(puesto_ex[k]),
        frecuencia=float(frec_nolin[k]),
        pc=float(pc[k]), puesto_pc=int(puesto_pc[k]),
        r=float(rr[k]), puesto_r=int(puestos(rr)[k])) for k in sel_nolin]

    formas: List[Dict] = []
    fechas_filas: List[Dict] = []
    escenas: List[Dict] = []
    if sel_nolin.size:
        print("  diagnosticando la FORMA de esas dependencias...")
        formas = diagnostico_forma(stack, sel_nolin)
        for f, t in zip(formas, info_nolin):
            t["forma_dominante"] = f["forma_dominante"]
        print("  separando el efecto de escena del espacial...")
        fm = None
        for cand in (os.path.join(os.path.dirname(stack_p), "metadata.npz"),):
            try:
                fm = np.load(cand, allow_pickle=True)["dates_millis"]
            except Exception:
                fm = None
        fechas_filas, escenas = descomponer_por_fecha(stack, sel_nolin, fm)

    # BLINDAJE. La misma seleccion, pero midiendo sobre un stack donde cada
    # canal se estandariza DENTRO de cada fecha. Eso borra el efecto de escena
    # -que aqui es dos o tres veces mayor que el espacial- y deja solo la
    # covariacion de variabilidad entre pixeles de una misma imagen. Si una
    # fuente conserva su ventaja sobre este subconjunto, no la estaba sacando
    # de la nubosidad invernal.
    print("  midiendo la version BLINDADA (escena estandarizada por fecha)...")
    _dcb = dependencia_no_lineal(stack, n_muestra=a.muestra_dcor,
                                 n_repeticiones=a.repeticiones_dcor,
                                 centrar_escena=True)
    sel_blind, frec_blind = seleccion_estable(
        stack, a.muestra_dcor, a.repeticiones_dcor,
        n_replicas=a.replicas_seleccion, centrar_escena=True)
    print(f"  pares no lineales blindados ({sel_blind.size}): " + ", ".join(
        f"{INDEX_NAMES[PARES[k][0]]}-{INDEX_NAMES[PARES[k][1]]}"
        for k in sel_blind))
    comunes = sorted(set(sel_nolin.tolist()) & set(sel_blind.tolist()))
    print(f"  comparten {len(comunes)}/{sel_blind.size} pares con la version "
          f"sin blindar")
    info_blind = [dict(
        par=f"{INDEX_NAMES[PARES[k][0]]}-{INDEX_NAMES[PARES[k][1]]}",
        dcor=float(_dcb[k, 0]), dcor_gauss=float(_dcb[k, 1]),
        exceso=float(_dcb[k, 0] - _dcb[k, 1]),
        pc=float(pc[k]), puesto_pc=int(puesto_pc[k]),
        r=float(rr[k]), puesto_r=int(puestos(rr)[k])) for k in sel_blind]
    puesto_rr = puestos(rr)
    sel_cond = pares_solo_condicionales(puesto_pc, puesto_rr)
    if sel_cond.size:
        print(f"  pares solo-condicionales ({sel_cond.size}): " + ", ".join(
            f"{INDEX_NAMES[PARES[k][0]]}-{INDEX_NAMES[PARES[k][1]]}"
            for k in sel_cond))

    # Los tres subconjuntos que se certifican en paralelo. Cada uno responde
    # una pregunta distinta y las tres hacen falta:
    #   solo condicional  el framework recupera lo que |pc| ve y |r| no
    #   no lineal         recupera lo que NINGUNA correlacion ve
    #   no lin blindado   lo mismo con el efecto de escena ya eliminado
    SUBCONJUNTOS = {"solo condicional": sel_cond,
                    "no lineal": sel_nolin,
                    "no lin blindado": sel_blind}

    # DIRECCION. Lo unico de todo el informe que ninguna correlacion puede
    # siquiera intentar, porque todas son simetricas. Ver
    # direccion_heterocedastica.
    print("  midiendo la DIRECCION de la dependencia en varianza...")
    D_dir, D_se, D_cons = direccion_con_estabilidad(
        stack, n_replicas=a.replicas_direccion)
    contrastes = contrastes_direccion(D_dir, D_se, D_cons)

    # Los tres controles que separan biologia de artefacto. Cada uno responde
    # una hipotesis alternativa distinta; ver contrastes_direccion y
    # mascara_vegetacion.
    submedidas = [("R2, todos los pixeles", D_dir, D_se, D_cons)]
    for nombre, msk in mascaras_cobertura(stack):
        frac = float(msk.mean())
        print(f"  cobertura {nombre}: {frac * 100:.1f}% de los pixeles-fecha")
        if frac < 0.02:
            print("    muy pocos pixeles, se salta")
            continue
        Dm, sem, cm = direccion_con_estabilidad(
            stack, n_replicas=a.replicas_direccion, mascara=msk)
        submedidas.append((f"R2, {nombre}", Dm, sem, cm))

    # dcor con la MISMA muestra y las MISMAS replicas que la version R2: si se
    # cae con menos potencia no se sabe si es la medida o el tamaño muestral.
    print(f"  dcor con muestra {a.muestra_dcor_direccion} y "
          f"{a.replicas_direccion} replicas (lento, O(n^2))...")
    D_dc, se_dc, cons_dc = direccion_con_estabilidad(
        stack, n_muestra=a.muestra_dcor_direccion,
        n_replicas=a.replicas_direccion, usar_dcor=True)
    submedidas.append(("dcor, todos los pixeles", D_dc, se_dc, cons_dc))
    dirigidas = {e: p for e, _n, _A, _s, p in fuentes if p}
    direccion = analisis_direccion(D_dir, dirigidas, n_perm=a.permutaciones,
                                   semilla=a.semilla)

    rng = np.random.default_rng(a.semilla)
    cert = []
    for etiqueta, nota, A, n_sem, por_sem in fuentes:
        print(f"    certificando: {etiqueta}")
        cert.append(certificar_fuente(A, etiqueta, nota, pc, rr, orden_pc,
                                      puesto_pc, n_sust, a.permutaciones,
                                      rng, n_sem, por_sem, T_liston,
                                      SUBCONJUNTOS))

    # Consensos entre paradigmas. Se construyen DESPUES de certificar las
    # fuentes sueltas porque necesitan saber cuales tienen ranking real
    # (frac_empates), pero sus miembros se agrupan por como se obtiene la
    # evidencia, no por como puntuo cada una. Ver fuentes_de_consenso.
    por_etiqueta = {e: A for e, _n, A, _s, _p in fuentes}
    for etiqueta, nota, A, n_sem, por_sem in fuentes_de_consenso(cert,
                                                                 por_etiqueta):
        print(f"    certificando: {etiqueta}")
        cert.append(certificar_fuente(A, etiqueta, nota, pc, rr, orden_pc,
                                      puesto_pc, n_sust, a.permutaciones,
                                      rng, n_sem, por_sem, T_liston,
                                      SUBCONJUNTOS))

    # Correccion por multiplicidad sobre las fuentes DEL FRAMEWORK. Se probaron
    # muchos rankings distintos contra la misma verdad; con alfa 0.05 se espera
    # casi una fuente significativa solo por probar. Benjamini-Hochberg controla
    # la tasa de falsos descubrimientos y es lo que decide que se puede
    # reportar como hallazgo y que es ruido de haber mirado muchas veces.
    idx_fw = [i for i, r in enumerate(cert)
              if not r["etiqueta"].startswith(("|pc|", "|r|"))]
    m = len(idx_fw)

    def _bh(campo: str, destino: str) -> None:
        vivos = [i for i in idx_fw if np.isfinite(cert[i][campo])]
        mm = len(vivos)
        if not mm:
            return
        orden = sorted(vivos, key=lambda i: cert[i][campo])
        q_prev = 1.0
        for rango in range(mm, 0, -1):
            i = orden[rango - 1]
            q = min(q_prev, cert[i][campo] * mm / rango)
            cert[i][destino] = float(q)
            q_prev = q

    _bh("p_permutacion", "q_fdr")

    # La misma correccion dentro de cada subconjunto: cada uno se prueba en
    # todas las fuentes por igual, asi que tiene la misma multiplicidad.
    for _k in SUBCONJUNTOS:
        vivos = [i for i in idx_fw
                 if np.isfinite(cert[i].get("sub", {}).get(_k, {})
                                .get("p", float("nan")))]
        mm = len(vivos)
        if not mm:
            continue
        orden = sorted(vivos, key=lambda i: cert[i]["sub"][_k]["p"])
        q_prev = 1.0
        for rango in range(mm, 0, -1):
            i = orden[rango - 1]
            q = min(q_prev, cert[i]["sub"][_k]["p"] * mm / rango)
            cert[i]["sub"][_k]["q_fdr"] = float(q)
            q_prev = q

    txt = texto(cert, a.permutaciones, info_nolin, formas,
                fechas_filas, escenas, direccion, D_dir, D_se, D_cons,
                contrastes, submedidas)
    print("\n" + txt)
    os.makedirs(a.salida, exist_ok=True)
    with open(os.path.join(a.salida, "CERTIFICACION_GLOBAL.json"), "w",
              encoding="utf-8") as f:
        json.dump(cert, f, indent=2, ensure_ascii=False)
    with open(os.path.join(a.salida, "CERTIFICACION_GLOBAL.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt)
    print(f"\n  Guardado: "
          f"{os.path.join(a.salida, 'CERTIFICACION_GLOBAL.json')}")
    print(f"  Guardado: {os.path.join(a.salida, 'CERTIFICACION_GLOBAL.txt')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
