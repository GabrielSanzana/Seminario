#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quien recupera la estructura verdadera: Datt o los metodos clasicos.

EL DISENO
---------
Sobre los datos sinteticos de verdad_sintetica.py, donde el DAG se fijo por
construccion, se corren siete estimadores de estructura y se puntuan todos
contra la misma verdad, con las mismas metricas.

LOS COMPETIDORES, Y POR QUE CADA UNO
------------------------------------
  correlacion         la linea base mas tonta. Si algo no la supera, no vale.
  correlacion parcial el caballo de batalla del caso lineal gaussiano. Es
                      exactamente el metodo cuyos supuestos el regimen lineal
                      cumple, asi que ahi deberia ganar.
  PC (Fisher z)       independencia condicional hasta orden 3, el estandar de
                      descubrimiento causal basado en restricciones.
  informacion mutua   no parametrica: ve dependencia no lineal, pero no
                      orienta nada.
  distancia de corr.  detecta cualquier dependencia, incluida la simetrica que
                      la correlacion de Pearson no ve.
  LOCO con arboles    EL COMPETIDOR QUE IMPORTA. Predecir X_i con todos los
                      demas usando arboles potenciados, y volver a predecir
                      sin X_j; lo que sube el error es la dependencia. Es no
                      lineal, es interventivo, y NO necesita un transformer.
                      Si empata con Datt, la tesis no puede afirmar que haga
                      falta un transformer, solo que hace falta algo no
                      lineal. Es la comparacion que decide el alcance de la
                      contribucion, y por eso esta aqui.
  Datt                el framework.

LAS METRICAS, Y POR QUE TRES
----------------------------
Ningun metodo basado en dependencia condicional puede recuperar el DAG: el
predictor optimo de X_i usa su manto de Markov, que incluye hijos y conyuges.
Puntuar contra el DAG y declarar fracaso general seria puntuar mal. Se separa:

  auc esqueleto    contra las aristas verdaderas sin direccion
  auc moral        contra el grafo moralizado, que es el limite identificable
  orientacion      de las aristas verdaderas recuperadas, que fraccion apunta
                   bien. Azar 0.5.

La tercera es la unica en la que Datt afirma algo que los demas no pueden
afirmar, y la unica que nunca se habia contrastado contra verdad conocida.

USO
    .venv/bin/python benchmark_recuperacion.py --datos datos_sinteticos
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

N_IDX = 12


# ------------------------------------------------------------ estimadores

def s_correlacion(Z):
    C = np.abs(np.corrcoef(Z.T))
    np.fill_diagonal(C, 0.0)
    return C


def s_parcial(Z):
    """Correlacion parcial via la matriz de precision.

    Es el estimador optimo bajo linealidad y gaussianidad, o sea el rival a
    batir en el regimen lineal y el que deberia fallar en el otro.
    """
    S = np.corrcoef(Z.T)
    P = np.linalg.pinv(S)
    d = np.sqrt(np.abs(np.diag(P)))
    R = -P / np.outer(d, d)
    R = np.abs(R)
    np.fill_diagonal(R, 0.0)
    return R


def _fisher_z(r, n, k):
    r = np.clip(r, -0.999999, 0.999999)
    z = 0.5 * np.log((1 + r) / (1 - r)) * np.sqrt(max(n - k - 3, 1))
    return 2 * (1 - stats.norm.cdf(abs(z)))


def _pcor_cond(S, i, j, cond):
    idx = [i, j] + list(cond)
    sub = S[np.ix_(idx, idx)]
    P = np.linalg.pinv(sub)
    return -P[0, 1] / np.sqrt(P[0, 0] * P[1, 1])


def s_pc(Z, orden_max=3, alfa=0.01):
    """Esqueleto del PC: se corta la arista i-j si existe algun conjunto
    condicionante que las vuelva independientes. El puntaje es el p mas alto
    encontrado, invertido, o sea la fuerza con la que la arista resiste."""
    n, p = Z.shape
    S = np.corrcoef(Z.T)
    pmax = np.zeros((p, p))
    for i, j in itertools.combinations(range(p), 2):
        peor = _fisher_z(S[i, j], n, 0)
        vecinos = [k for k in range(p) if k not in (i, j)]
        for orden in range(1, orden_max + 1):
            if peor > alfa:
                break
            for cond in itertools.combinations(vecinos, orden):
                r = _pcor_cond(S, i, j, cond)
                peor = max(peor, _fisher_z(r, n, orden))
                if peor > alfa:
                    break
        pmax[i, j] = pmax[j, i] = peor
    out = -np.log10(np.maximum(pmax, 1e-300))
    np.fill_diagonal(out, 0.0)
    return out


def s_info_mutua(Z, bins=16):
    p = Z.shape[1]
    q = np.array([np.digitize(Z[:, k],
                              np.quantile(Z[:, k], np.linspace(0, 1, bins + 1)[1:-1]))
                  for k in range(p)]).T
    out = np.zeros((p, p))
    for i, j in itertools.combinations(range(p), 2):
        tab = np.histogram2d(q[:, i], q[:, j], bins=bins)[0]
        tab = tab / tab.sum()
        pi, pj = tab.sum(1, keepdims=True), tab.sum(0, keepdims=True)
        nz = tab > 0
        mi = float((tab[nz] * np.log(tab[nz] / (pi @ pj)[nz])).sum())
        out[i, j] = out[j, i] = mi
    return out


def s_dcor(Z, n_sub=2000, semilla=0):
    """Distancia de correlacion: cero si y solo si hay independencia, asi que
    ve la dependencia simetrica que la correlacion de Pearson declara nula."""
    rng = np.random.default_rng(semilla)
    sel = rng.choice(len(Z), min(n_sub, len(Z)), replace=False)
    W = Z[sel]
    p = W.shape[1]
    A = []
    for k in range(p):
        d = np.abs(W[:, k][:, None] - W[:, k][None, :])
        A.append(d - d.mean(0) - d.mean(1)[:, None] + d.mean())
    out = np.zeros((p, p))
    for i, j in itertools.combinations(range(p), 2):
        num = (A[i] * A[j]).mean()
        den = np.sqrt((A[i] ** 2).mean() * (A[j] ** 2).mean())
        out[i, j] = out[j, i] = np.sqrt(max(num, 0)) / np.sqrt(max(den, 1e-12))
    return out


def rasgos_espaciales(X, n_sub=20000, semilla=0):
    """Cada canal con su valor en el pixel y la media de su vecindario 5x5.

    Sin esto la comparacion estaria amanada al reves: en el regimen espacial
    el transformer ve el parche entero y LOCO veria un pixel suelto, asi que
    ganar no probaria nada sobre la arquitectura sino sobre la informacion
    disponible. Con estos rasgos, LOCO tiene acceso al vecindario igual que
    el modelo, y la comparacion vuelve a ser sobre el metodo.
    """
    from scipy.ndimage import uniform_filter
    vec = uniform_filter(X.astype(np.float64), size=(1, 5, 5, 1))
    rng = np.random.default_rng(semilla)
    p = X.shape[-1]
    Z = X.reshape(-1, p).astype(np.float64)
    V = vec.reshape(-1, p)
    sel = rng.choice(len(Z), min(n_sub, len(Z)), replace=False)
    Z, V = Z[sel], V[sel]
    m, s = Z.mean(0), Z.std(0) + 1e-12
    return (Z - m) / s, (V - V.mean(0)) / (V.std(0) + 1e-12)


def s_loco_espacial(Z, V):
    """LOCO donde cada canal aporta pixel y vecindario, y ablacionar el canal
    j le quita los dos. Es el rival mas fuerte posible sin red neuronal."""
    from sklearn.ensemble import HistGradientBoostingRegressor as GB
    n_tr = int(0.7 * len(Z))
    p = Z.shape[1]
    out = np.zeros((p, p))
    F = np.column_stack([Z, V])          # 2p columnas: pixel y vecindario
    for i in range(p):
        cols = [k for k in range(p) if k != i] + [p + k for k in range(p)
                                                  if k != i]
        base = GB(max_iter=120, random_state=0).fit(F[:n_tr][:, cols],
                                                    Z[:n_tr, i])
        e0 = float(np.mean((base.predict(F[n_tr:][:, cols])
                            - Z[n_tr:, i]) ** 2))
        for j in range(p):
            if j == i:
                continue
            resto = [c for c in cols if c not in (j, p + j)]
            m = GB(max_iter=120, random_state=0).fit(F[:n_tr][:, resto],
                                                     Z[:n_tr, i])
            ej = float(np.mean((m.predict(F[n_tr:][:, resto])
                                - Z[n_tr:, i]) ** 2))
            out[i, j] = ej - e0
    return out


def rasgos_retardados(X, n_sub=20000, semilla=0, retardos=(1, 2)):
    """Cada canal con su valor y el de las fechas anteriores.

    En el regimen temporal el mecanismo pasa por el pasado del padre. Un
    metodo que trate cada fecha como una fila independiente esta ciego, asi
    que compararlo contra un modelo con ventana temporal no diria nada sobre
    el metodo. Con estos rasgos, LOCO ve el mismo pasado.
    """
    rng = np.random.default_rng(semilla)
    p = X.shape[-1]
    bloques = [X]
    for r in retardos:
        prev = np.roll(X, r, axis=0)
        prev[:r] = X[:r]
        bloques.append(prev)
    Z = X.reshape(-1, p).astype(np.float64)
    L = [b.reshape(-1, p).astype(np.float64) for b in bloques[1:]]
    sel = rng.choice(len(Z), min(n_sub, len(Z)), replace=False)
    z = Z[sel]
    z = (z - z.mean(0)) / (z.std(0) + 1e-12)
    ls = [(m[sel] - m[sel].mean(0)) / (m[sel].std(0) + 1e-12) for m in L]
    return z, ls


def s_loco_retardado(Z, retardadas):
    """LOCO donde ablacionar el canal j le quita su presente y su pasado."""
    from sklearn.ensemble import HistGradientBoostingRegressor as GB
    n_tr = int(0.7 * len(Z))
    p = Z.shape[1]
    F = np.column_stack([Z] + list(retardadas))
    n_bloques = 1 + len(retardadas)
    out = np.zeros((p, p))
    for i in range(p):
        cols = [b * p + k for b in range(n_bloques) for k in range(p)
                if not (b == 0 and k == i)]
        base = GB(max_iter=120, random_state=0).fit(F[:n_tr][:, cols],
                                                    Z[:n_tr, i])
        e0 = float(np.mean((base.predict(F[n_tr:][:, cols])
                            - Z[n_tr:, i]) ** 2))
        for j in range(p):
            if j == i:
                continue
            quitar = {b * p + j for b in range(n_bloques)}
            resto = [c for c in cols if c not in quitar]
            m = GB(max_iter=120, random_state=0).fit(F[:n_tr][:, resto],
                                                     Z[:n_tr, i])
            ej = float(np.mean((m.predict(F[n_tr:][:, resto])
                                - Z[n_tr:, i]) ** 2))
            out[i, j] = ej - e0
    return out


def s_loco(Z, n_sub=20000, semilla=0):
    """LOCO no lineal con arboles potenciados.

    El rival serio. Para cada canal i se ajusta un predictor con los otros
    once y se vuelve a ajustar sin el canal j; lo que sube el error es cuanto
    aporta j a predecir i. Es la misma logica interventiva de Datt sin ninguna
    red neuronal y sin contexto espacial: cada pixel es una fila.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor as GB
    rng = np.random.default_rng(semilla)
    sel = rng.choice(len(Z), min(n_sub, len(Z)), replace=False)
    W = Z[sel]
    n_tr = int(0.7 * len(W))
    p = W.shape[1]
    out = np.zeros((p, p))
    for i in range(p):
        otros = [k for k in range(p) if k != i]
        base = GB(max_iter=120, random_state=0).fit(W[:n_tr][:, otros],
                                                    W[:n_tr, i])
        e0 = float(np.mean((base.predict(W[n_tr:][:, otros])
                            - W[n_tr:, i]) ** 2))
        for j in otros:
            resto = [k for k in otros if k != j]
            m = GB(max_iter=120, random_state=0).fit(W[:n_tr][:, resto],
                                                     W[:n_tr, i])
            ej = float(np.mean((m.predict(W[n_tr:][:, resto])
                                - W[n_tr:, i]) ** 2))
            out[i, j] = ej - e0
    return out


# ------------------------------------------------------------- puntuacion

def auc(scores, verdad):
    """Area bajo la ROC de las aristas fuera de la diagonal."""
    off = ~np.eye(len(verdad), dtype=bool)
    s, y = scores[off], verdad[off].astype(bool)
    if y.all() or not y.any():
        return float("nan")
    r = stats.rankdata(s)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def precision_en_k(scores, verdad):
    off = ~np.eye(len(verdad), dtype=bool)
    k = int(verdad[off].sum())
    orden = np.argsort(-scores[off])
    return float(verdad[off][orden[:k]].mean()), k


def orientacion(scores, A):
    """De las aristas verdaderas, cuantas se orientan bien.

    Para cada par (i, j) con j padre de i se compara scores[i, j] contra
    scores[j, i]. Un metodo simetrico da 0.5 por construccion y no se puntua.
    Azar 0.5, y se acompana de un test binomial.
    """
    aciertos, total = 0, 0
    for i in range(len(A)):
        for j in range(len(A)):
            if A[i, j]:
                total += 1
                aciertos += int(scores[i, j] > scores[j, i])
    if not total:
        return float("nan"), 0, float("nan")
    p = float(stats.binomtest(aciertos, total, 0.5,
                              alternative="greater").pvalue)
    return aciertos / total, total, p


def es_simetrico(S):
    return bool(np.allclose(S, S.T, atol=1e-9))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", default="datos_sinteticos")
    ap.add_argument("--n-pixeles", type=int, default=200000, dest="n_pix")
    ap.add_argument("--salida", default="resultados")
    a = ap.parse_args()

    with open(os.path.join(a.datos, "verdad.json"), encoding="utf-8") as f:
        v = json.load(f)
    A = np.array(v["dag"])
    M = np.array(v["moral"])
    esq = ((A + A.T) > 0).astype(int)

    W = 88
    L = ["=" * W, " QUIEN RECUPERA LA ESTRUCTURA VERDADERA", "",
         " DAG de %d aristas sobre 12 nodos, grafo moralizado de %d."
         % (int(A.sum()), int(M.sum() // 2)),
         "",
         " Ningun metodo basado en dependencia condicional puede recuperar el",
         " DAG: el predictor optimo de X_i usa su manto de Markov, que incluye",
         " hijos y conyuges. El blanco justo es el grafo moralizado. La",
         " columna que decide es la ultima: orientacion de las aristas",
         " verdaderas, con azar en 0.5.",
         "=" * W]

    todo = {}
    for regimen in ("lineal", "no_lineal", "espacial", "temporal"):
        ruta = os.path.join(a.datos, "stack_" + regimen + ".npy")
        if not os.path.exists(ruta):
            continue
        X = np.load(ruta)
        Z = X.reshape(-1, X.shape[-1]).astype(np.float64)
        rng = np.random.default_rng(0)
        if len(Z) > a.n_pix:
            Z = Z[rng.choice(len(Z), a.n_pix, replace=False)]
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
        print("  %s: %d pixeles, %d canales" % (regimen, *Z.shape))

        met = {}
        met["correlacion"] = s_correlacion(Z)
        met["correlacion parcial"] = s_parcial(Z)
        met["PC (Fisher z)"] = s_pc(Z)
        met["informacion mutua"] = s_info_mutua(Z)
        met["distancia de corr."] = s_dcor(Z)
        met["LOCO con arboles"] = s_loco(Z)
        Zs, Vs = rasgos_espaciales(X)
        met["LOCO con vecindario"] = s_loco_espacial(Zs, Vs)
        Zr, Ls = rasgos_retardados(X)
        met["LOCO con retardos"] = s_loco_retardado(Zr, Ls)

        # Se cargan todas las variantes de Datt que haya para este regimen.
        # El sufijo _largo es el mismo experimento con mas epocas: la primera
        # tanda dejo el modelo sin converger -el log mostraba el regimen
        # lineal mejorando todavia en la epoca 48 de 50- y sin arreglar eso,
        # un Datt bajo no distingue "el estimador no sirve" de "el modelo no
        # habia aprendido nada que auditar".
        hubo = False
        for etiq in ("", "_largo", "_neutro", "_tokens", "_p26"):
            dj = os.path.join(a.datos, "DATT_" + regimen + etiq + ".json")
            if not os.path.exists(dj):
                continue
            hubo = True
            with open(dj, encoding="utf-8") as f:
                d = json.load(f)
            for nombre, blo in d.items():
                clave = "Datt " + nombre + " " + (etiq.lstrip("_")
                                                          or "corto")
                met[clave] = np.array(blo["Datt"])
        if not hubo:
            print("    aun no hay Datt para " + regimen)

        L += ["", " REGIMEN " + regimen.upper().replace("_", " "),
              " " + "-" * (W - 2),
              "  " + "metodo".ljust(24) + "auc esq".rjust(9)
              + "auc moral".rjust(11) + "prec@k".rjust(9)
              + "orientacion".rjust(13) + "p".rjust(9)]
        filas = {}
        for nombre, S in met.items():
            a_e, a_m = auc(S, esq), auc(S, M)
            pk, k = precision_en_k(S, esq)
            if es_simetrico(S):
                ori, tot, po = float("nan"), 0, float("nan")
                txt_ori, txt_p = "simetrico", "-"
            else:
                ori, tot, po = orientacion(S, A)
                txt_ori, txt_p = "%.2f" % ori, "%.4f" % po
            filas[nombre] = dict(auc_esqueleto=a_e, auc_moral=a_m,
                                 precision_en_k=pk, k=k,
                                 orientacion=(None if np.isnan(ori)
                                              else ori),
                                 p_orientacion=(None if np.isnan(po)
                                                else po),
                                 n_aristas_orientadas=tot)
            L.append("  " + nombre[:23].ljust(24)
                     + ("%.3f" % a_e).rjust(9) + ("%.3f" % a_m).rjust(11)
                     + ("%.2f" % pk).rjust(9) + txt_ori.rjust(13)
                     + txt_p.rjust(9))
        todo[regimen] = filas

    L += ["", "=" * W, " COMO LEER ESTO", "",
          " Si en el regimen lineal gana la correlacion parcial, el montaje",
          " es correcto: ahi sus supuestos se cumplen exactamente y tiene que",
          " ganar. Que el framework gane ahi seria motivo de sospecha, no de",
          " celebracion.",
          "",
          " La comparacion que fija el alcance de la contribucion es Datt",
          " contra LOCO con arboles. Los dos son interventivos y no lineales;",
          " LOCO no necesita transformer ni contexto espacial. Si empatan, lo",
          " que hace falta es un metodo no lineal, no esta arquitectura.",
          "=" * W]

    txt = "\n".join(L)
    print("\n" + txt)
    os.makedirs(a.salida, exist_ok=True)
    with open(os.path.join(a.salida, "BENCHMARK_RECUPERACION.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt)
    with open(os.path.join(a.salida, "BENCHMARK_RECUPERACION.json"), "w",
              encoding="utf-8") as f:
        json.dump(todo, f, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
