#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analisis ESTRUCTURAL del grafo, no de aristas sueltas. INDEPENDIENTE del pipeline.

POR QUE ESTE MODULO EXISTE
--------------------------
certificar_todo.py contesta una sola pregunta: el ranking de las 66 aristas
coincide con una verdad externa (|pc|, |r|, la asimetria heterocedastica)?
Esa es la pregunta mas exigente que se le puede hacer al framework, y ademas
no es la que el framework plantea. La propuesta habla de transformar la matriz
de interaccion A en un grafo G y analizarlo como grafo; exigirle que ordene 66
aristas igual que la correlacion parcial es pedirle que reproduzca un estimador
que ya se tiene, arista por arista.

Aqui se contestan las preguntas de nivel de grafo, que son cuatro y ninguna
necesita una verdad externa arista a arista:

  1. REPRODUCIBILIDAD  Las 50 semillas coinciden entre si mas de lo que
     coincidirian 50 matrices con las mismas fuerzas y las etiquetas barajadas?
     Si no, no hay nada que interpretar: cada modelo aprendio otra cosa. Es la
     pregunta que justifica haber entrenado 50 veces y la unica que se contesta
     sin verdad externa de ningun tipo.

  2. MODULARIDAD  El grafo agrupa los indices por familia espectral?

     ADVERTENCIA QUE CAMBIA COMO SE LEE ESTE NUMERO. Las familias NO son
     informacion externa al entrenamiento: MASCARA_MODO="familia_balanceada"
     construye la mascara con _mascara_familias(), de modo que en cada paso se
     oculta la familia completa del indice a reconstruir. El modelo ve el
     patron de familias en cada lote. Recuperarlas NO es descubrimiento: es un
     CONTROL POSITIVO del canal A -> G -> comunidades. Dice que si hay
     estructura en la matriz, el metodo la extrae; no dice que esa estructura
     venga del terreno.

     Con esa lectura sigue siendo util, y es la unica validacion del canal que
     existe sin una verdad externa arista a arista.

  3. CENTRALIDAD  El orden de importancia de los NODOS es estable entre
     semillas? Es una afirmacion mas debil que la de aristas y por eso puede
     sobrevivir donde aquella falla: puede haber acuerdo en que NDVI es central
     sin acuerdo en cual de sus aristas concretas lo hace central.

  4. ESPECTRO  El operador relacional tiene una estructura espectral estable?
     La propuesta menciona el analisis espectral del operador; aqui se mide el
     gap entre el primer y el segundo autovalor, que dice si el grafo tiene una
     direccion dominante o esta reparido, y su variacion entre semillas.

TODAS las pruebas usan la misma nulidad: permutar las 12 etiquetas de indice.
Conserva la distribucion de pesos y la estructura del grafo, y solo destruye
la correspondencia con los nombres. Es el nulo correcto para "el grafo sabe
cual indice es cual", que es lo que hay que demostrar.

USO
    .venv/bin/python analisis_grafo.py
    .venv/bin/python analisis_grafo.py --permutaciones 5000
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
    BANDAS_INDICE, CANDIDATOS_MATRICES, CANDIDATOS_PARADIGMAS,
    CANDIDATOS_STACK, INDEX_FAMILIES, INDEX_NAMES, N_IDX, N_PARES, PARES,
    _buscar, _cargar_npy, familia_de_indice, fuerza_por_par, pc_y_r, puestos,
    recolectar_fuentes,
)


# ---------------------------------------------------------------------------
# Medidas estructurales
# ---------------------------------------------------------------------------

def _simetrica_positiva(A: np.ndarray) -> np.ndarray:
    """Grafo no dirigido con pesos >= 0 y diagonal nula.

    La modularidad y la mayoria de las medidas de comunidad estan definidas
    para pesos no negativos. La matriz de ablacion puede tener entradas
    negativas (j hace que reconstruir i sea MAS facil), asi que se desplaza al
    minimo en vez de recortar: recortar a cero borraria esa informacion, y
    desplazar la conserva como el extremo inferior de la escala.
    """
    S = (np.abs(A) + np.abs(A.T)) / 2.0
    np.fill_diagonal(S, 0.0)
    if S.min() < 0:
        S = S - S.min()
        np.fill_diagonal(S, 0.0)
    return S


def modularidad(A: np.ndarray, particion: np.ndarray) -> float:
    """Modularidad de Newman-Girvan de una particion dada, con pesos.

    Q = (1/2m) * sum_ij [ A_ij - k_i k_j / 2m ] * delta(c_i, c_j)

    Mide cuanto mas peso hay dentro de los grupos del que habria si las mismas
    fuerzas de nodo se repartieran al azar. Aqui la particion NO se busca: es
    la de familias espectrales, que viene del dominio. Se pregunta si el grafo
    aprendido concuerda con ella, no cual es la mejor particion del grafo.
    """
    S = _simetrica_positiva(A)
    k = S.sum(axis=1)
    m2 = k.sum()
    if m2 <= 0:
        return 0.0
    misma = particion[:, None] == particion[None, :]
    esperado = np.outer(k, k) / m2
    return float(((S - esperado) * misma).sum() / m2)


def pagerank(A: np.ndarray, alpha: float = 0.85,
             iteraciones: int = 200) -> np.ndarray:
    """Distribucion estacionaria de la caminata sobre el grafo dirigido.

    Se usa PageRank y no la suma de columna porque el grafo esta renormalizado
    por filas: la suma de columna ya es una medida de "cuanto peso recibe" pero
    ignora de quien lo recibe. PageRank pondera por la importancia del emisor,
    que es lo que distingue un sumidero trivial de un nodo central.
    """
    M = np.abs(A).astype(np.float64)
    np.fill_diagonal(M, 0.0)
    s = M.sum(axis=1, keepdims=True)
    M = np.where(s > 0, M / np.maximum(s, 1e-12), 1.0 / N_IDX)
    v = np.full(N_IDX, 1.0 / N_IDX)
    for _ in range(iteraciones):
        nv = alpha * (M.T @ v) + (1 - alpha) / N_IDX
        if np.abs(nv - v).max() < 1e-12:
            v = nv
            break
        v = nv
    return v


def descomposicion_varianza(mats: List[np.ndarray]) -> Dict:
    """Cuanta variacion es sistematica entre aristas y cuanta es de la semilla.

    LA PREGUNTA CORRECTA DETRAS DE "SI FUERA RUIDO SERIA NORMAL". Esa version
    no sirve por dos motivos: los 50 valores de una arista son la muestra, no
    la media de muestras, asi que el teorema central del limite no dice nada
    sobre su forma; y los pesos estan acotados y renormalizados a suma 1, o sea
    viven en el simplex, de modo que la no-normalidad esta garantizada por
    construccion y no distingue ruido de estructura.

    Lo que si formaliza la intuicion es una descomposicion de varianza. Sobre
    la tabla de 66 aristas x 50 semillas:

        var_entre   varianza de las medias por arista  (senal sistematica)
        var_dentro  varianza dentro de cada arista entre semillas  (ruido)
        ICC = var_entre / (var_entre + var_dentro)

    ICC cerca de 0 significa que las aristas son indistinguibles entre si y
    todo lo que se ve es la semilla: no hay nada que interpretar. Cerca de 1
    significa que cada arista tiene un valor propio que las semillas reproducen.
    Es exactamente "cuanta potencia tiene esa variable", y ademas se puede dar
    arista por arista con la razon senal/ruido |media| / desviacion.
    """
    F = np.stack([fuerza_por_par(m) for m in mats])      # (semillas, pares)
    medias = F.mean(axis=0)
    var_entre = float(medias.var(ddof=1))
    var_dentro = float(F.var(axis=0, ddof=1).mean())
    icc = (var_entre / (var_entre + var_dentro)
           if (var_entre + var_dentro) > 0 else 0.0)
    snr = np.abs(medias) / np.maximum(F.std(axis=0, ddof=1), 1e-12)
    orden = np.argsort(-snr)
    return dict(
        icc=float(icc), var_entre=var_entre, var_dentro=var_dentro,
        snr_mediana=float(np.median(snr)),
        aristas_con_snr_mayor_3=int((snr > 3).sum()),
        top_aristas_snr=[
            dict(par=f"{INDEX_NAMES[PARES[k][0]]}-{INDEX_NAMES[PARES[k][1]]}",
                 snr=float(snr[k]), media=float(medias[k]))
            for k in orden[:8]])


def _indice_rand_ajustado(a: np.ndarray, b: np.ndarray) -> float:
    """ARI entre dos particiones. Corregido por azar: 0 = azar, 1 = identicas."""
    n = len(a)
    ca, cb = np.unique(a, return_inverse=True)[1], np.unique(b, return_inverse=True)[1]
    tabla = np.zeros((ca.max() + 1, cb.max() + 1))
    for x, y in zip(ca, cb):
        tabla[x, y] += 1
    suma_ij = (tabla * (tabla - 1) / 2).sum()
    a_i = tabla.sum(axis=1)
    b_j = tabla.sum(axis=0)
    suma_a = (a_i * (a_i - 1) / 2).sum()
    suma_b = (b_j * (b_j - 1) / 2).sum()
    total = n * (n - 1) / 2
    esperado = suma_a * suma_b / total if total > 0 else 0.0
    maximo = (suma_a + suma_b) / 2
    return float((suma_ij - esperado) / (maximo - esperado)
                 if maximo != esperado else 0.0)


def comunidades(A: np.ndarray, k: int = 5) -> np.ndarray:
    """Particion del grafo en k comunidades, por clustering espectral.

    Se detectan SIN darle la respuesta: a diferencia de la modularidad, que
    evalua una particion conocida, aqui cada semilla propone la suya y luego se
    mide si coinciden entre ellas. Que 50 modelos independientes encuentren la
    misma agrupacion es mas fuerte que reproducir el orden de las aristas,
    porque la agrupacion es una propiedad global del grafo.

    Espectral y no modularity maximization porque no hay networkx en el
    entorno y porque con 12 nodos el laplaciano normalizado es exacto y
    determinista, sin la aleatoriedad de los metodos codiciosos.
    """
    S = _simetrica_positiva(A)
    d = S.sum(axis=1)
    if (d <= 0).any():
        d = np.maximum(d, 1e-12)
    Lsym = np.eye(len(S)) - (S / np.sqrt(np.outer(d, d)))
    w, V = np.linalg.eigh(Lsym)
    U = V[:, 1:k]                                # se salta el trivial
    norma = np.linalg.norm(U, axis=1, keepdims=True)
    U = U / np.maximum(norma, 1e-12)
    # k-means determinista: inicializacion por los puntos mas separados.
    centros = [U[0]]
    for _ in range(k - 1):
        dist = np.min([np.linalg.norm(U - c, axis=1) for c in centros], axis=0)
        centros.append(U[int(np.argmax(dist))])
    centros = np.array(centros)
    etiquetas = np.zeros(len(U), dtype=int)
    for _ in range(100):
        nuevas = np.argmin(
            np.linalg.norm(U[:, None, :] - centros[None], axis=2), axis=1)
        if (nuevas == etiquetas).all():
            break
        etiquetas = nuevas
        for c in range(k):
            if (etiquetas == c).any():
                centros[c] = U[etiquetas == c].mean(axis=0)
    return etiquetas


def discretizar_top_p(A: np.ndarray, p: float = 0.15) -> np.ndarray:
    """Grafo binario con la fraccion p de aristas mas fuertes.

    Es el paso Phi: A -> G de la propuesta, que hasta ahora no se estaba
    analizando: todo el informe corre sobre la matriz continua. Un umbral
    global y no por fila porque por fila fuerza a cada nodo a tener el mismo
    numero de salidas, que es una estructura impuesta por el corte y no
    aprendida.
    """
    S = _simetrica_positiva(A)
    vals = np.array([S[i, j] for i, j in PARES])
    if vals.max() <= 0:
        return np.zeros((N_IDX, N_IDX), dtype=int)
    corte = np.quantile(vals, 1 - p)
    G = np.zeros((N_IDX, N_IDX), dtype=int)
    for i, j in PARES:
        if S[i, j] >= corte:
            G[i, j] = G[j, i] = 1
    return G


def jaccard_grafos(G1: np.ndarray, G2: np.ndarray) -> float:
    a = np.array([G1[i, j] for i, j in PARES], dtype=bool)
    b = np.array([G2[i, j] for i, j in PARES], dtype=bool)
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else float("nan")


def gap_espectral(A: np.ndarray) -> float:
    """|lambda_1| - |lambda_2| del operador relacional, normalizado.

    Alto significa que el grafo tiene un modo dominante -una direccion de flujo
    que explica casi todo-; bajo significa que la estructura esta repartida
    entre varios modos. Es la lectura espectral que menciona la propuesta.
    """
    ev = np.abs(np.linalg.eigvals(np.abs(A)))
    ev.sort()
    if ev[-1] <= 0:
        return 0.0
    return float((ev[-1] - ev[-2]) / ev[-1])


# ---------------------------------------------------------------------------
# Las cuatro preguntas
# ---------------------------------------------------------------------------

def concordancia_entre_semillas(mats: List[np.ndarray],
                                n_pares_max: int = 300,
                                rng: Optional[np.random.Generator] = None
                                ) -> float:
    """Spearman medio entre las fuerzas por par de dos semillas distintas.

    Con 50 semillas hay 1225 parejas; se submuestrean para que el coste no
    crezca al cuadrado, con la misma cantidad en el observado y en el nulo.
    """
    rng = rng or np.random.default_rng(0)
    fr = [fuerza_por_par(m) for m in mats]
    n = len(fr)
    todas = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(todas) > n_pares_max:
        idx = rng.choice(len(todas), n_pares_max, replace=False)
        todas = [todas[t] for t in idx]
    vals = [stats.spearmanr(fr[i], fr[j]).statistic for i, j in todas]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def analizar_fuente(etiqueta: str, mats: List[np.ndarray],
                    fam: np.ndarray, n_perm: int,
                    rng: np.random.Generator) -> Dict:
    """Las cuatro medidas estructurales de una fuente, cada una con su nulo."""
    A = np.mean(mats, axis=0)

    # 1. Reproducibilidad entre semillas. El nulo permuta las etiquetas de cada
    # semilla POR SEPARADO: conserva la forma de cada grafo y destruye solo el
    # acuerdo entre ellos, que es justo lo que se quiere contrastar.
    obs_rep = concordancia_entre_semillas(mats, rng=rng)
    nulo_rep = []
    for _ in range(min(n_perm, 200)):
        barajadas = []
        for m in mats:
            pi = rng.permutation(N_IDX)
            barajadas.append(m[np.ix_(pi, pi)])
        nulo_rep.append(concordancia_entre_semillas(barajadas, n_pares_max=60,
                                                    rng=rng))
    nulo_rep = np.array([v for v in nulo_rep if np.isfinite(v)])
    p_rep = (float((1 + int((nulo_rep >= obs_rep).sum())) / (1 + nulo_rep.size))
             if nulo_rep.size else float("nan"))

    # 2. Modularidad respecto a las familias espectrales.
    obs_mod = modularidad(A, fam)
    nulo_mod = np.empty(n_perm)
    for t in range(n_perm):
        pi = rng.permutation(N_IDX)
        nulo_mod[t] = modularidad(A[np.ix_(pi, pi)], fam)
    # BILATERAL. La version unilateral anterior daba p=0.004 al rollout por
    # tener Q apenas 0.009 por encima de su nulo, y llamaba a eso "agrupa por
    # familia" cuando su Q es negativa: el grafo conecta ENTRE familias, no
    # dentro. Una Q negativa significativa es tan informativa como una
    # positiva, y con el test unilateral quedaba invisible o, peor, invertida.
    desv = np.abs(nulo_mod - nulo_mod.mean())
    p_mod = float((1 + int((desv >= abs(obs_mod - nulo_mod.mean())).sum()))
                  / (1 + n_perm))
    # El sentido lo decide el signo de Q, no la comparacion con el nulo. La
    # version anterior lo decidia contra la media del nulo y etiquetaba como
    # "agrupa DENTRO" a grafos con Q negativa que estaban apenas por encima de
    # un nulo aun mas negativo. El nulo sirve para el p, no para el sentido.
    sentido = ("agrupa DENTRO de familia" if obs_mod > 0
               else "conecta ENTRE familias")
    mod_sem = np.array([modularidad(m, fam) for m in mats])

    # Comunidades detectadas por cada semilla, comparadas entre si. Que 50
    # modelos independientes encuentren la misma agrupacion, sin que nadie se
    # la de, es mas fuerte que reproducir el orden de las aristas.
    parts = [comunidades(m) for m in mats]
    pares_p = [(i, j) for i in range(len(parts)) for j in range(i + 1, len(parts))]
    if len(pares_p) > 300:
        sel = rng.choice(len(pares_p), 300, replace=False)
        pares_p = [pares_p[t] for t in sel]
    ari_obs = float(np.mean([_indice_rand_ajustado(parts[i], parts[j])
                             for i, j in pares_p]))
    ari_nulo = []
    for _ in range(min(n_perm, 200)):
        bar = [p[rng.permutation(N_IDX)] for p in parts]
        ari_nulo.append(np.mean([_indice_rand_ajustado(bar[i], bar[j])
                                 for i, j in pares_p[:60]]))
    ari_nulo = np.array(ari_nulo)
    p_ari = float((1 + int((ari_nulo >= ari_obs).sum())) / (1 + ari_nulo.size))
    ari_familias = float(np.mean([_indice_rand_ajustado(p, fam)
                                  for p in parts]))

    # El grafo DISCRETIZADO, que es el G de Phi: A -> G. Todo lo anterior corre
    # sobre la matriz continua; el framework opera sobre el grafo binario.
    Gs = [discretizar_top_p(m) for m in mats]
    jac = float(np.mean([jaccard_grafos(Gs[i], Gs[j]) for i, j in pares_p]))
    jac_nulo = []
    for _ in range(min(n_perm, 200)):
        pi = rng.permutation(N_IDX)
        jac_nulo.append(np.mean([
            jaccard_grafos(Gs[i], Gs[j][np.ix_(pi, pi)])
            for i, j in pares_p[:60]]))
    jac_nulo = np.array(jac_nulo)
    p_jac = float((1 + int((jac_nulo >= jac).sum())) / (1 + jac_nulo.size))
    G_cons = discretizar_top_p(A)
    aristas_cons = [f"{INDEX_NAMES[i]}-{INDEX_NAMES[j]}"
                    for i, j in PARES if G_cons[i, j]]

    # Las comunidades del grafo de consenso, con nombres: sin esto el ARI es
    # un numero sin contenido y no se puede leer que agrupo el modelo.
    com_cons = comunidades(A)
    grupos_nombrados = [
        [INDEX_NAMES[i] for i in range(N_IDX) if com_cons[i] == c]
        for c in sorted(set(com_cons.tolist()))]
    grupos_nombrados = [g for g in grupos_nombrados if g]

    var = descomposicion_varianza(mats)

    # 3. Estabilidad del orden de los nodos por PageRank.
    pr = np.array([pagerank(m) for m in mats])
    n = len(pr)
    parejas = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(parejas) > 300:
        sel = rng.choice(len(parejas), 300, replace=False)
        parejas = [parejas[t] for t in sel]
    obs_cen = float(np.mean([stats.spearmanr(pr[i], pr[j]).statistic
                             for i, j in parejas]))
    orden_pr = np.argsort(-pr.mean(axis=0))

    # 4. Espectro.
    gaps = np.array([gap_espectral(m) for m in mats])

    return dict(
        etiqueta=etiqueta, n_semillas=len(mats),
        reproducibilidad=obs_rep, p_reproducibilidad=p_rep,
        reproducibilidad_nulo=(float(nulo_rep.mean()) if nulo_rep.size
                               else float("nan")),
        modularidad=obs_mod, p_modularidad=p_mod,
        modularidad_nulo=float(nulo_mod.mean()),
        modularidad_sentido=sentido,
        modularidad_semilla_std=float(mod_sem.std()),
        frac_semillas_mismo_sentido=float(
            ((mod_sem > nulo_mod.mean()) == (obs_mod > nulo_mod.mean())).mean()),
        ari_entre_semillas=ari_obs, p_ari=p_ari,
        ari_nulo=float(ari_nulo.mean()) if ari_nulo.size else float("nan"),
        ari_contra_familias=ari_familias,
        jaccard_top_p=jac, p_jaccard=p_jac,
        jaccard_nulo=float(jac_nulo.mean()) if jac_nulo.size else float("nan"),
        n_aristas_consenso=len(aristas_cons),
        aristas_consenso=aristas_cons,
        comunidades_consenso=grupos_nombrados,
        icc=var["icc"], var_entre=var["var_entre"], var_dentro=var["var_dentro"],
        snr_mediana=var["snr_mediana"],
        aristas_con_snr_mayor_3=var["aristas_con_snr_mayor_3"],
        top_aristas_snr=var["top_aristas_snr"],
        estabilidad_centralidad=obs_cen,
        nodos_por_centralidad=[INDEX_NAMES[i] for i in orden_pr],
        gap_espectral=float(gaps.mean()), gap_espectral_std=float(gaps.std()))


# ---------------------------------------------------------------------------

def texto(filas: List[Dict], n_perm: int) -> str:
    W = 78
    L: List[str] = []
    L.append("=" * W)
    L.append(" ANALISIS ESTRUCTURAL DEL GRAFO")
    L.append("")
    L.append(" No pregunta si el ranking de las 66 aristas coincide con una")
    L.append(" verdad externa -eso lo hace certificar_todo.py y ninguna fuente")
    L.append(" lo pasa-. Pregunta si el GRAFO tiene estructura: si las semillas")
    L.append(" coinciden, si agrupa por familia espectral, si el orden de los")
    L.append(" nodos es estable y si el operador tiene un modo dominante.")
    L.append("")
    L.append(f" Nulo en todo: permutar las {N_IDX} etiquetas, {n_perm} veces.")
    L.append("=" * W)
    L.append("")
    L.append(f"  {'fuente':<30}{'repro':>8}{'ICC':>7}{'SNR>3':>7}"
             f"{'ARI':>7}{'p':>8}{'Jacc':>7}{'p':>8}{'modul':>8}{'p':>8}")
    for r in filas:
        L.append(f"  {r['etiqueta']:<30}{r['reproducibilidad']:>+8.3f}"
                 f"{r['icc']:>7.3f}{r['aristas_con_snr_mayor_3']:>5}/66"
                 f"{r['ari_entre_semillas']:>7.3f}{r['p_ari']:>8.4f}"
                 f"{r['jaccard_top_p']:>7.3f}{r['p_jaccard']:>8.4f}"
                 f"{r['modularidad']:>+8.3f}{r['p_modularidad']:>8.4f}")
    L.append("")
    L.append("  repro = Spearman medio entre las fuerzas de dos semillas.")
    L.append("  ICC = fraccion de la varianza total que es sistematica entre")
    L.append("     aristas, y no de la semilla. Es la version correcta de 'si")
    L.append("     fuera ruido no habria estructura': cerca de 0 las aristas")
    L.append("     son indistinguibles y solo se ve la semilla; cerca de 1 cada")
    L.append("     arista tiene un valor propio que las semillas reproducen.")
    L.append("  SNR>3 = aristas cuya media supera 3 veces su desviacion entre")
    L.append("     semillas. Es la potencia por arista, no del grafo entero.")
    L.append("  ARI = acuerdo entre las comunidades que detecta cada semilla")
    L.append("     POR SU CUENTA, corregido por azar. No se le da la particion.")
    L.append("  Jacc = solapamiento del grafo binario Top-P entre semillas, o")
    L.append("     sea la reproducibilidad de G y no de A.")
    L.append("  modul = modularidad contra las familias espectrales, test")
    L.append("     BILATERAL: negativa significa que conecta ENTRE familias.")
    L.append("")

    L.append("-" * W)
    L.append(" QUE SE PUEDE AFIRMAR")
    rep_ok = [r for r in filas
              if np.isfinite(r["p_reproducibilidad"])
              and r["p_reproducibilidad"] < 0.05]
    mod_ok = [r for r in filas if r["p_modularidad"] < 0.05]
    if rep_ok:
        L.append(f"  {len(rep_ok)}/{len(filas)} fuentes son reproducibles entre")
        L.append("  semillas por encima del azar. Ahi el promedio sobre 50")
        L.append("  entrenamientos mide algo, no suaviza ruido:")
        for r in sorted(rep_ok, key=lambda x: -x["reproducibilidad"]):
            L.append(f"    {r['etiqueta']:<30}{r['reproducibilidad']:>+7.3f}"
                     f"  contra {r['reproducibilidad_nulo']:+.3f} de nulo"
                     f"  p={r['p_reproducibilidad']:.4f}")
    else:
        L.append("  NINGUNA fuente es reproducible entre semillas por encima")
        L.append("  del azar. Eso invalida el promedio sobre 50 entrenamientos:")
        L.append("  cada modelo aprendio otra cosa y la media no representa a")
        L.append("  ninguno.")
    L.append("")
    ari_ok = [r for r in filas if r["p_ari"] < 0.05]
    jac_ok = [r for r in filas if r["p_jaccard"] < 0.05]
    L.append(f"  {len(ari_ok)}/{len(filas)} fuentes detectan las MISMAS")
    L.append("  comunidades en semillas distintas, sin que se les de ninguna")
    L.append("  particion; y en el grafo discretizado (el G de Phi: A -> G):")
    for r in sorted(filas, key=lambda x: -x["ari_entre_semillas"]):
        L.append(f"    {r['etiqueta']:<30}ARI={r['ari_entre_semillas']:+.3f}"
                 f" (nulo {r['ari_nulo']:+.3f}, p={r['p_ari']:.4f})"
                 f"  Jaccard={r['jaccard_top_p']:.3f}"
                 f" (nulo {r['jaccard_nulo']:.3f}, p={r['p_jaccard']:.4f})")
    L.append("")
    L.append("  Contra las familias espectrales, las comunidades detectadas dan")
    L.append("  un ARI de:")
    for r in sorted(filas, key=lambda x: -x["ari_contra_familias"]):
        L.append(f"    {r['etiqueta']:<30}{r['ari_contra_familias']:+.3f}"
                 f"   modularidad {r['modularidad']:+.3f}"
                 f" ({r['modularidad_sentido']}, p={r['p_modularidad']:.4f})")
    L.append("")
    L.append("  ALCANCE DE ESTAS AFIRMACIONES. Reproducibilidad y modularidad")
    L.append("  no dicen que las aristas concretas sean relaciones reales del")
    L.append("  fenomeno: dicen que el grafo es consistente y que concuerda con")
    L.append("  una particion conocida. Recuperar las familias espectrales es")
    L.append("  recuperar algo que ya se sabia por la formula de cada indice.")
    L.append("  Es la prueba de que el canal funciona, no un hallazgo sobre el")
    L.append("  terreno.")
    L.append("-" * W)
    for r in filas:
        L.append("")
        L.append(f"  --- {r['etiqueta']}  ({r['n_semillas']} semillas) ---")
        L.append(f"  nodos por centralidad (PageRank): "
                 f"{', '.join(r['nodos_por_centralidad'][:6])}...")
        L.append(f"  modularidad por semilla: desviacion "
                 f"{r['modularidad_semilla_std']:.4f}")
        L.append(f"  gap espectral {r['gap_espectral']:.3f} "
                 f"+-{r['gap_espectral_std']:.3f}")
        L.append(f"  ICC {r['icc']:.3f}  (var entre aristas "
                 f"{r['var_entre']:.3e}, var entre semillas "
                 f"{r['var_dentro']:.3e})")
        L.append(f"  aristas con senal/ruido > 3: "
                 f"{r['aristas_con_snr_mayor_3']}/66, mediana "
                 f"{r['snr_mediana']:.2f}")
        for t in r["top_aristas_snr"][:5]:
            L.append(f"    {t['par']:<20} SNR={t['snr']:>6.1f}")
        L.append(f"  grafo Top-P de consenso: {r['n_aristas_consenso']} aristas")
        L.append("    " + ", ".join(r["aristas_consenso"][:10]))
        L.append("  comunidades detectadas en el consenso:")
        for g in r["comunidades_consenso"]:
            L.append("    {" + ", ".join(g) + "}")
    L.append("=" * W)
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Analisis estructural del grafo de interaccion.")
    ap.add_argument("--stack", default=None)
    ap.add_argument("--matrices", default=None)
    ap.add_argument("--paradigmas", default=None)
    ap.add_argument("--salida", default="resultados")
    ap.add_argument("--permutaciones", type=int, default=2000)
    ap.add_argument("--semilla", type=int, default=0)
    a = ap.parse_args()

    stack_p = a.stack or _buscar(CANDIDATOS_STACK, False)
    if not stack_p:
        print("ERROR: no encuentro el stack. Pasa --stack ruta.npy")
        return 1
    stack = np.load(stack_p)
    PC, RR = pc_y_r(stack)
    dirm = a.matrices or _buscar(CANDIDATOS_MATRICES, True)
    dirp = a.paradigmas or _buscar(CANDIDATOS_PARADIGMAS, True)
    print(f"  matrices: {dirm}")

    fam = familia_de_indice()
    print("  familias espectrales: " + ", ".join(
        f"{n}({len(v)})" for n, v in INDEX_FAMILIES.items()))

    # Solo las fuentes con matrices POR SEMILLA: las tres preguntas de
    # reproducibilidad y estabilidad no existen sin ellas.
    fuentes = [(e, p) for e, _n, _A, _s, p in
               recolectar_fuentes(dirm, dirp, PC, RR) if p and len(p) >= 5]
    print(f"  fuentes con matrices por semilla: {len(fuentes)}")
    if not fuentes:
        print("  Sin matrices por semilla no hay analisis estructural posible.")
        return 1

    rng = np.random.default_rng(a.semilla)
    filas = []
    for etiqueta, mats in fuentes:
        print(f"    analizando: {etiqueta}")
        filas.append(analizar_fuente(etiqueta, mats, fam, a.permutaciones, rng))

    txt = texto(filas, a.permutaciones)
    print("\n" + txt)
    os.makedirs(a.salida, exist_ok=True)
    with open(os.path.join(a.salida, "ANALISIS_GRAFO.json"), "w",
              encoding="utf-8") as f:
        json.dump(filas, f, indent=2, ensure_ascii=False)
    with open(os.path.join(a.salida, "ANALISIS_GRAFO.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt)
    print(f"\n  Guardado: {os.path.join(a.salida, 'ANALISIS_GRAFO.txt')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
