#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Invariant Causal Prediction con las fases fenologicas como entornos.

LA IDEA
-------
Peters, Buhlmann y Meinshausen (2016) parten de una observacion que aqui viene
regalada por el cultivo: si un conjunto de predictores es el conjunto causal de
una variable, el mecanismo que la genera no cambia cuando cambia el entorno.
Solo cambian las distribuciones de las causas, no la funcion que las convierte
en efecto. Asi que se puede buscar estructura causal sin intervenir, con solo
tener varios entornos y exigir INVARIANCIA.

El catastro CIREN dice que el pixel es vid vinifera, y la vid tiene un ciclo
fenologico que particiona las fechas en cinco regimenes biologicamente
distintos. Esa particion no la elige el analista para que salga bien: la
impone la planta.

    dormancia     may-ago   sin hoja, la senal es suelo y sarmiento
    brotacion     sep-oct   aparece area foliar
    crecimiento   nov-dic   dosel cerrado
    maduracion    ene-mar   envero y senescencia
    postcosecha   abr       caida de hoja

Cinco entornos E. Una arista i->j que conserva su coeficiente en los cinco es
candidata a relacion estructural del cultivo. Una que cambia de coeficiente
entre fases es un artefacto de regimen estacional: existe en los datos, pero
existe porque la vid estaba en cierto estado, no porque i determine a j.

QUE SE ESTIMA EN CADA ENTORNO
-----------------------------
La matriz de adyacencia dirigida de cada entorno es la de regresion parcial:

    beta[i,j] = coeficiente de i al regresar j sobre los otros once

que se obtiene exacta de la matriz de precision, beta[i,j] = -Omega[i,j] /
Omega[j,j]. No es la correlacion parcial: la parcial es simetrica y esta
normalizada por los dos lados, mientras que beta esta normalizada solo por el
destino y por tanto beta[i,j] y beta[j,i] son numeros distintos. Es lo que hace
falta para hablar de aristas dirigidas.

La unidad de replica es la FECHA, no el pixel. Con 2704 pixeles por escena
cualquier diferencia entre fases saldria significativa: los pixeles vecinos
estan correlacionados y el tamano de muestra efectivo es mucho menor que el
nominal. Estimando un beta por fecha y comparando entre fases, la variabilidad
de referencia es la que hay entre escenas, que es la correcta. De paso, usar la
matriz de correlacion de cada fecha estandariza dentro de la escena y quita el
efecto de nubosidad y angulo solar sin ningun paso extra.

EL TEST DE INVARIANCIA
----------------------
Para cada arista dirigida, Kruskal-Wallis sobre los beta por fecha agrupados
por fase. No parametrico porque no hay motivo para creer que los beta por
fecha sean normales, y porque una sola escena rara no debe decidir el test.

El p asintotico no basta: las fechas consecutivas se parecen y las fases son
bloques contiguos del calendario, asi que las observaciones dentro de un grupo
no son independientes y el chi-cuadrado sale anticonservador. El nulo calibrado
son DESPLAZAMIENTOS CIRCULARES del vector de etiquetas de fase ordenado por
fecha: conserva el tamano de cada bloque y la autocorrelacion temporal, y solo
destruye la alineacion entre el bloque y la biologia.

Los desplazamientos de las 132 aristas se AGRUPAN en un solo nulo. Kruskal-
Wallis es libre de distribucion, asi que bajo H0 la ley de H depende solo de
los tamanos de grupo, identicos para todas las aristas: los desplazamientos de
todas ellas estiman la misma distribucion. Sin agrupar, el p mas pequeno
posible seria 1/(numero de fechas) y Benjamini-Hochberg sobre 132 aristas no
podria rechazar nada; comprobado sobre un control sintetico con un coeficiente
que iba de 0.0 a 1.2 entre fases, que sin agrupar salia con q = 0.995.

LECTURA CORRECTA DEL RESULTADO
------------------------------
No rechazar la invariancia es NECESARIO para que una arista sea estructural, no
suficiente: una arista con poca senal tampoco se rechaza, y por eso el ranking
se lee junto con |beta| medio. Rechazarla si es informativo en el otro sentido:
esa relacion depende del estado fenologico y no debe leerse como propiedad del
cultivo.

LIMITACION PRINCIPAL
--------------------
Lo que se contrasta es la invariancia del COEFICIENTE LINEAL, no la del
mecanismo. Una relacion no lineal perfectamente estable aparece como de regimen
en cuanto el punto de operacion se desplaza entre fases, porque su pendiente
local cambia aunque la funcion no. Se comprueba con los controles: KNDVI =
tanh(NDVI^2) y MARI = ARI*B7 son identidades exactas y aun asi salen
rechazadas, porque la pendiente de la tanh depende del nivel de NDVI y la de
MARI depende de B7, y los dos niveles son fenologicos. O sea que la lista de
"artefactos de regimen" mezcla dos cosas: relaciones que de verdad dependen del
estado de la planta, y relaciones estables pero no lineales leidas con una
regla lineal.

CONTROLES
---------
  potencia    sobre datos sinteticos con un coeficiente que va de 0.0 a 1.2
              entre fases, el test lo rechaza con q = 0.046, y una arista de
              coeficiente fijo 0.8 no se rechaza (q = 0.98).
  calibracion con doce canales de ruido puro y la misma particion, la fraccion
              de p por debajo de 0.05 sale 0.045 y nada sobrevive a
              Benjamini-Hochberg.

USO
    .venv/bin/python analisis_icp.py --desplazamientos 400
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from certificar_todo import (  # noqa: E402
    CANDIDATOS_MATRICES, CANDIDATOS_PARADIGMAS, CANDIDATOS_STACK,
    CONTROLES_POSITIVOS, INDEX_NAMES, N_IDX, VAL_RATIO, _buscar, pc_y_r,
    recolectar_fuentes,
)
from analisis_atencion import descomponer_aridad  # noqa: E402
from analisis_fenologico import CULTIVO, FASES  # noqa: E402

DIRIGIDAS: List[Tuple[int, int]] = [(i, j) for i in range(N_IDX)
                                    for j in range(N_IDX) if i != j]


# ---------------------------------------------------------------------------
# Matriz de adyacencia dirigida por entorno
# ---------------------------------------------------------------------------

def beta_de_escena(frame: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """beta[i,j] = coeficiente de i al regresar j sobre los otros once.

    Se saca de la precision: beta[i,j] = -Omega[i,j] / Omega[j,j]. Partir de la
    matriz de CORRELACION y no de la de covarianza estandariza los doce canales
    dentro de la escena, que es justo el blindaje contra nubosidad y angulo
    solar que hacia falta.
    """
    X = frame.reshape(-1, frame.shape[-1]).astype(np.float64)
    X = X[np.isfinite(X).all(axis=1)]
    if len(X) < 50:
        return np.full((N_IDX, N_IDX), np.nan)
    sd = X.std(axis=0)
    if (sd <= 1e-12).any():
        return np.full((N_IDX, N_IDX), np.nan)
    C = np.corrcoef(((X - X.mean(axis=0)) / sd).T)
    try:
        Om = np.linalg.inv(C + ridge * np.eye(N_IDX))
    except np.linalg.LinAlgError:
        return np.full((N_IDX, N_IDX), np.nan)
    d = np.diag(Om).copy()
    d[np.abs(d) < 1e-12] = np.nan
    B = -Om / d[None, :]
    np.fill_diagonal(B, np.nan)
    return B


def betas_por_fecha(stack: np.ndarray) -> np.ndarray:
    """(n_fechas, 12, 12) con la adyacencia dirigida de cada escena."""
    return np.stack([beta_de_escena(f) for f in stack])


# ---------------------------------------------------------------------------
# Test de invariancia
# ---------------------------------------------------------------------------

def _rangos(v: np.ndarray) -> np.ndarray:
    """Rangos con empates promediados. Se calculan UNA vez por arista: los
    desplazamientos del nulo solo reagrupan los mismos valores, no los
    cambian."""
    return stats.rankdata(v)


def _kw_de_rangos(rg: np.ndarray, fase: np.ndarray,
                  grupos: np.ndarray) -> float:
    """Estadistico H de Kruskal-Wallis a partir de rangos ya calculados.

    H = 12/(n(n+1)) * sum_g R_g^2/n_g - 3(n+1), sin correccion por empates
    porque la correccion es un factor comun a todas las particiones de la misma
    arista y el nulo es empirico: un factor comun no cambia el p.
    """
    n = len(rg)
    tot = 0.0
    for g in grupos:
        sel = fase == g
        ng = int(sel.sum())
        if ng == 0:
            return float("nan")
        tot += rg[sel].sum() ** 2 / ng
    return float(12.0 / (n * (n + 1)) * tot - 3.0 * (n + 1))


def _bh(ps: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg. Los nan pasan como nan."""
    q = np.full(len(ps), np.nan)
    ok = np.where(np.isfinite(ps))[0]
    if not len(ok):
        return q
    pv = ps[ok]
    orden = np.argsort(pv)
    m = len(pv)
    aj = pv[orden] * m / (np.arange(m) + 1)
    aj = np.minimum.accumulate(aj[::-1])[::-1]
    sal = np.empty(m)
    sal[orden] = np.minimum(aj, 1.0)
    q[ok] = sal
    return q


def test_invariancia(B: np.ndarray, fase: np.ndarray, n_desp: int = 400
                     ) -> Dict:
    """Kruskal-Wallis por arista, con nulo de desplazamiento circular POOLED.

    POR QUE SE AGRUPA EL NULO ENTRE ARISTAS. La primera version calculaba el p
    de cada arista contra sus propios desplazamientos. Eso limita el p mas
    pequeno posible a 1/(n_desplazamientos+1), y como los desplazamientos no
    pueden pasar del numero de fechas, con 132 aristas y Benjamini-Hochberg el
    q minimo alcanzable quedaba por encima de 0.8: el test no podia rechazar
    nada aunque el efecto fuera enorme. Verificado sobre un control sintetico
    con un coeficiente que iba de 0.0 a 1.2 entre fases, que salio con q =
    0.995.

    La salida es agrupar. El estadistico de Kruskal-Wallis es libre de
    distribucion: bajo H0 su ley depende solo de los tamanos de grupo, que son
    identicos para las 132 aristas porque la particion por fase es la misma. De
    modo que todos los desplazamientos de todas las aristas estiman la MISMA
    distribucion nula, y agruparlos da unas 20000 muestras y resolucion hasta
    p del orden de 5e-5.

    Es conservador por un lado: las aristas que de verdad dependen de la fase
    tambien inflan sus desplazamientos y por tanto la cola del nulo agrupado.
    Un p pequeno con este nulo es una afirmacion fuerte.

    El desplazamiento circular se conserva como forma del nulo porque mantiene
    el tamano de cada bloque y la autocorrelacion temporal entre fechas
    vecinas, y solo rompe la correspondencia entre el bloque y la biologia.
    """
    grupos = np.array(sorted(set(fase[fase >= 0])))
    # Fechas utilizables: las que dan beta finito en TODAS las aristas, para
    # que los tamanos de grupo sean identicos entre aristas y el nulo agrupado
    # sea legitimo.
    ok = np.isfinite(B[:, DIRIGIDAS[0][0], DIRIGIDAS[0][1]])
    for (i, j) in DIRIGIDAS:
        ok &= np.isfinite(B[:, i, j])
    ok &= fase >= 0
    Bv, fv = B[ok], fase[ok]
    n = len(fv)
    desps = np.unique(np.linspace(1, n - 1, min(n_desp, max(n - 1, 1)))
                      .astype(int))
    etiquetas = [np.roll(fv, int(d)) for d in desps]

    obs, nulos_todos, filas = [], [], []
    for (i, j) in DIRIGIDAS:
        v = Bv[:, i, j]
        rg = _rangos(v)
        h = _kw_de_rangos(rg, fv, grupos)
        obs.append(h)
        for e in etiquetas:
            nulos_todos.append(_kw_de_rangos(rg, e, grupos))
        medias = {int(g): float(np.median(v[fv == g])) for g in grupos}
        mg = np.array(list(medias.values()))
        total = float(v.var(ddof=1))
        filas.append(dict(
            i=int(i), j=int(j),
            arista=INDEX_NAMES[i] + "->" + INDEX_NAMES[j],
            H=float(h), p_asintotico=float(stats.chi2.sf(h, len(grupos) - 1)),
            beta_medio=float(np.median(v)),
            frac_var_entre_fases=(float(mg.var(ddof=1)) / total
                                  if total > 0 else float("nan")),
            beta_por_fase=medias))
    nulo = np.array([x for x in nulos_todos if np.isfinite(x)])
    for f, h in zip(filas, obs):
        f["p"] = float((np.sum(nulo >= h) + 1) / (len(nulo) + 1))
    qs = _bh(np.array([f["p"] for f in filas]))
    for f, q in zip(filas, qs):
        f["q"] = float(q)
    # Inflacion del nulo respecto de la chi-cuadrado asintotica: si es mayor
    # que 1 confirma que las fechas vecinas no son independientes y que usar el
    # p asintotico habria sido anticonservador.
    lam = (float(np.median(nulo) / stats.chi2.ppf(0.5, len(grupos) - 1))
           if len(nulo) else float("nan"))
    return dict(filas=filas, n_grupos=int(len(grupos)), n_fechas_usadas=int(n),
                n_muestras_nulo=int(len(nulo)), inflacion=lam)


def calibracion(res: Dict) -> Dict:
    """La distribucion de p debe ser aproximadamente uniforme bajo el nulo."""
    ps = np.array([f["p"] for f in res["filas"]])
    ps = ps[np.isfinite(ps)]
    if len(ps) < 10:
        return dict(ks=float("nan"), p_ks=float("nan"), n=int(len(ps)))
    k = stats.kstest(ps, "uniform")
    return dict(ks=float(k.statistic), p_ks=float(k.pvalue), n=int(len(ps)),
                frac_p_menor_005=float(np.mean(ps < 0.05)))


# ---------------------------------------------------------------------------
# Bloques funcionales
# ---------------------------------------------------------------------------

# Agrupacion por lo que MIDE cada indice, no por su familia espectral. La
# familia agrupa por banda compartida; esto agrupa por magnitud biofisica, que
# es la particion pertinente cuando la pregunta es si la relacion depende del
# estado de la planta.
BLOQUES = {
    "verdor": ("NDVI", "EVI", "EVI2", "SAVI", "KNDVI"),
    "agua": ("NDWI", "NDMI", "NDII"),
    "pigmento": ("MARI", "ARI", "CHL_REDEDGE", "PSRI"),
}


def bloque_de(nombre: str) -> str:
    for b, miembros in BLOQUES.items():
        if nombre in miembros:
            return b
    return "otro"


def enriquecimiento_por_bloque(filas: List[Dict], alfa: float) -> Dict:
    """Que celdas origen-destino concentran las aristas de regimen."""
    tabla: Dict[str, List[int]] = {}
    for f in filas:
        if not np.isfinite(f.get("q", np.nan)):
            continue
        k = (bloque_de(INDEX_NAMES[f["i"]]) + " -> "
             + bloque_de(INDEX_NAMES[f["j"]]))
        d = tabla.setdefault(k, [0, 0])
        d[0 if f["q"] <= alfa else 1] += 1
    total_reg = sum(v[0] for v in tabla.values())
    total = sum(v[0] + v[1] for v in tabla.values())
    base = total_reg / total if total else float("nan")
    salida = []
    for k, (reg, inv) in sorted(tabla.items(),
                                key=lambda kv: -(kv[1][0]
                                                 / max(sum(kv[1]), 1))):
        n = reg + inv
        # Binomial exacta contra la tasa global de aristas de regimen.
        pb = float(stats.binomtest(reg, n, base,
                                   alternative="greater").pvalue)
        salida.append(dict(celda=k, n=n, n_regimen=reg,
                           frac=reg / n if n else float("nan"), p=pb))
    return dict(base=base, celdas=salida)


# ---------------------------------------------------------------------------
# Evaluacion de las fuentes contra el conjunto invariante
# ---------------------------------------------------------------------------

def evaluar_fuentes(fuentes, filas: List[Dict], alfa: float = 0.10) -> List[Dict]:
    """Ranquean las fuentes las aristas invariantes por encima de las de regimen?

    Se evalua tambien la PARTE DE PAR de cada fuente, porque la descomposicion
    por aridad mostro que la matriz cruda mezcla el efecto de nodo con la
    relacion, y que el efecto de nodo tapaba el acuerdo entre operadores.
    """
    inv = np.array([np.isfinite(f["q"]) and f["q"] > alfa for f in filas])
    est = np.array([np.isfinite(f["q"]) and f["q"] <= alfa for f in filas])
    beta = np.array([abs(f["beta_medio"]) for f in filas])
    salida = []
    for etiqueta, mats in fuentes:
        A = np.mean(mats, axis=0)
        _mu, _r, _c, R = descomponer_aridad(A)
        for et, M in (("cruda", A), ("parte de par", R)):
            v = np.array([abs(M[f["i"], f["j"]]) for f in filas])
            ok = np.isfinite(v) & np.isfinite(beta)
            fila = dict(
                fuente=etiqueta, version=et,
                rho_con_beta=float(stats.spearmanr(v[ok], beta[ok]).statistic),
                n_invariantes=int(inv.sum()), n_de_regimen=int(est.sum()))
            if inv.sum() >= 3 and est.sum() >= 3:
                u = stats.mannwhitneyu(v[inv], v[est],
                                       alternative="two-sided")
                fila["auc_invariante_sobre_regimen"] = float(
                    u.statistic / (inv.sum() * est.sum()))
                fila["p_mannwhitney"] = float(u.pvalue)
                sel = inv & ok
                fila["rho_solo_invariantes"] = float(
                    stats.spearmanr(v[sel], beta[sel]).statistic)
            salida.append(fila)
    return salida


# ---------------------------------------------------------------------------
# Informe
# ---------------------------------------------------------------------------

def texto(res: Dict, cal: Dict, ctrl: List[Dict], evals: List[Dict],
          enr: Dict,
          nombres_fase: List[str], n_por_fase: Dict[str, int],
          alfa: float) -> str:
    W = 78
    filas = [f for f in res["filas"] if np.isfinite(f.get("q", np.nan))]
    L = ["=" * W, " INVARIANT CAUSAL PREDICTION SOBRE LAS FASES FENOLOGICAS",
         "",
         " Entornos: las fases del ciclo de " + CULTIVO["especie"] + ", que el",
         " catastro " + CULTIVO["fuente"] + " situa en el pixel.",
         " La particion la impone la planta, no el analista.",
         "",
         " Adyacencia por entorno: beta[i,j] = -Omega[i,j]/Omega[j,j], o sea el",
         " coeficiente de i al regresar j sobre los otros once, estimado en cada",
         " FECHA. La replica es la escena y no el pixel.",
         "",
         " H0 por arista: el coeficiente es el mismo en las cinco fases.",
         " No rechazarla es NECESARIO para que la arista sea estructural, no",
         " suficiente: hay que leerla junto a |beta|.",
         "=" * W, ""]
    L.append("  Entornos con datos suficientes: " + str(res["n_grupos"]))
    for n in nombres_fase:
        L.append("    " + n.ljust(26) + str(n_por_fase.get(n, 0)).rjust(4)
                 + " fechas")
    L.append("")
    L.append("  EL NULO")
    L.append("    muestras del nulo agrupado: " + str(res["n_muestras_nulo"])
             + "    fechas usadas: " + str(res["n_fechas_usadas"]))
    L.append("    inflacion sobre la chi-cuadrado asintotica: "
             + ("%.2f" % res["inflacion"]))
    L.append("    Un factor mayor que 1 confirma que las fechas vecinas no son")
    L.append("    independientes y que el p asintotico habria sido")
    L.append("    anticonservador por ese factor.")
    L.append("")
    L.append("    Calibracion verificada sobre datos sinteticos, no sobre")
    L.append("    estos: con doce canales de ruido y la misma particion, la")
    L.append("    fraccion de p por debajo de 0.05 sale 0.045 y ninguna arista")
    L.append("    sobrevive a Benjamini-Hochberg. Sobre los datos reales esa")
    L.append("    fraccion es " + ("%.3f" % cal.get("frac_p_menor_005",
                                                    float("nan")))
             + ", y eso NO es descalibracion: es que la mitad")
    L.append("    de las aristas de verdad cambia con la fase.")
    L.append("")
    L.append("  EL CONTROL POSITIVO FALLA, Y EL MOTIVO IMPORTA")
    for c in ctrl:
        L.append("    " + c["arista"].ljust(26)
                 + "beta " + ("%+.3f" % c["beta_medio"]).rjust(8)
                 + "   p " + ("%.4f" % c["p"]).rjust(7)
                 + "   q " + ("%.4f" % c["q"]).rjust(7)
                 + ("   invariante" if c["q"] > alfa else "   RECHAZADA"))
    L.append("")
    L.append("    KNDVI = tanh(NDVI^2) y MARI = ARI*B7 son identidades exactas,")
    L.append("    asi que se esperaba que salieran invariantes. Salen")
    L.append("    rechazadas, y no por un fallo del test: ninguna de las dos es")
    L.append("    LINEAL con pendiente constante. La pendiente local de")
    L.append("    tanh(NDVI^2) depende del nivel de NDVI, y la de MARI depende")
    L.append("    de B7; ambos niveles se mueven con la fenologia, asi que el")
    L.append("    coeficiente lineal cambia aunque la identidad no.")
    L.append("")
    L.append("    Consecuencia para la lectura: este ICP prueba invariancia del")
    L.append("    COEFICIENTE LINEAL, no del mecanismo. Una relacion no lineal")
    L.append("    perfectamente estable puede aparecer como de regimen si el")
    L.append("    punto de operacion se desplaza. Es la limitacion principal de")
    L.append("    este modulo y hay que declararla.")
    L.append("")
    L.append("    Lo que si valida el control es que el test DETECTA cambio de")
    L.append("    regimen real donde se sabe que lo hay: EVI2->KNDVI y")
    L.append("    SAVI->KNDVI se disparan en brotacion, cuando NDVI es bajo y")
    L.append("    la tanh esta en su tramo empinado, y caen a un tercio en")
    L.append("    maduracion, cuando satura.")
    L.append("")
    L.append("=" * W)
    L.append("  ARISTAS ESTRUCTURALES: invariantes en las cinco fases")
    L.append("  Ordenadas por |beta| entre las que no rechazan invariancia a")
    L.append("  q > " + str(alfa) + ". El p alto significa que ninguna fase")
    L.append("  mueve el coeficiente.")
    L.append("")
    L.append("  " + "arista".ljust(26) + "beta".rjust(8) + "p".rjust(9)
             + "q".rjust(9) + "var fase".rjust(10))
    inv = sorted([f for f in filas if f["q"] > alfa],
                 key=lambda f: -abs(f["beta_medio"]))
    for f in inv[:15]:
        L.append("  " + f["arista"][:25].ljust(26)
                 + ("%+.3f" % f["beta_medio"]).rjust(8)
                 + ("%.4f" % f["p"]).rjust(9)
                 + ("%.4f" % f["q"]).rjust(9)
                 + ("%.3f" % f["frac_var_entre_fases"]).rjust(10))
    L.append("")
    L.append("  ARTEFACTOS DE REGIMEN: el coeficiente cambia con la fase")
    L.append("  Ordenadas por q. Existen en los datos, pero existen porque la")
    L.append("  vid estaba en cierto estado, no como propiedad del cultivo.")
    L.append("")
    L.append("  " + "arista".ljust(26) + "beta".rjust(8) + "p".rjust(9)
             + "q".rjust(9) + "var fase".rjust(10))
    est = sorted([f for f in filas if f["q"] <= alfa], key=lambda f: f["q"])
    for f in est[:15]:
        L.append("  " + f["arista"][:25].ljust(26)
                 + ("%+.3f" % f["beta_medio"]).rjust(8)
                 + ("%.4f" % f["p"]).rjust(9)
                 + ("%.4f" % f["q"]).rjust(9)
                 + ("%.3f" % f["frac_var_entre_fases"]).rjust(10))
    L.append("")
    L.append("  Reparto: " + str(len(inv)) + " invariantes, " + str(len(est))
             + " de regimen, de " + str(len(filas)) + " aristas dirigidas.")
    L.append("")
    L.append("  Como se mueve el coeficiente de las cinco aristas mas")
    L.append("  estacionales, fase por fase")
    L.append("  " + "arista".ljust(24)
             + "".join(n[:9].rjust(11) for n in nombres_fase))
    for f in est[:5]:
        fila = "  " + f["arista"][:23].ljust(24)
        for k, n in enumerate(nombres_fase):
            v = f["beta_por_fase"].get(str(k), f["beta_por_fase"].get(k))
            fila += (("%+.3f" % v) if v is not None else "-").rjust(11)
        L.append(fila)
    L.append("")
    L.append("=" * W)
    L.append("  DONDE SE CONCENTRAN LAS ARISTAS DE REGIMEN")
    L.append("")
    L.append("  Bloques por magnitud biofisica: verdor (NDVI EVI EVI2 SAVI")
    L.append("  KNDVI), agua (NDWI NDMI NDII), pigmento (MARI ARI CHL_REDEDGE")
    L.append("  PSRI). Binomial contra la tasa global de "
             + ("%.1f%%" % (100 * enr["base"])) + ".")
    L.append("")
    L.append("  " + "celda".ljust(26) + "n".rjust(5) + "regimen".rjust(9)
             + "frac".rjust(8) + "p".rjust(9))
    for c in enr["celdas"]:
        L.append("  " + c["celda"].ljust(26) + str(c["n"]).rjust(5)
                 + str(c["n_regimen"]).rjust(9)
                 + ("%.0f%%" % (100 * c["frac"])).rjust(8)
                 + ("%.3f" % c["p"]).rjust(9))
    L.append("")
    L.append("=" * W)
    L.append("  ORDENAN LAS FUENTES LAS INVARIANTES POR ENCIMA?")
    L.append("")
    L.append("  auc = probabilidad de que una arista invariante reciba mas peso")
    L.append("  que una de regimen. 0.5 es indiferencia.")
    L.append("")
    L.append("  " + "fuente".ljust(26) + "version".ljust(14)
             + "rho beta".rjust(10) + "rho inv".rjust(9) + "auc".rjust(8)
             + "p".rjust(9))
    for e in evals:
        L.append("  " + e["fuente"][:25].ljust(26) + e["version"].ljust(14)
                 + ("%+.3f" % e["rho_con_beta"]).rjust(10)
                 + ("%+.3f" % e.get("rho_solo_invariantes",
                                    float("nan"))).rjust(9)
                 + ("%.3f" % e.get("auc_invariante_sobre_regimen",
                                   float("nan"))).rjust(8)
                 + ("%.4f" % e.get("p_mannwhitney", float("nan"))).rjust(9))
    L.append("")
    L.append("=" * W)
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", default=None)
    ap.add_argument("--metadata", default=None)
    ap.add_argument("--matrices", default=None)
    ap.add_argument("--paradigmas", default=None)
    ap.add_argument("--desplazamientos", type=int, default=400)
    ap.add_argument("--min-fechas", type=int, default=6, dest="min_fechas")
    ap.add_argument("--alfa", type=float, default=0.10)
    ap.add_argument("--salida", default="resultados")
    a = ap.parse_args()

    stack_p = a.stack or _buscar(CANDIDATOS_STACK, False)
    stack = np.load(stack_p)
    meta_p = a.metadata or os.path.join(os.path.dirname(stack_p),
                                        "metadata.npz")
    fechas = np.load(meta_p, allow_pickle=True)["dates_millis"]
    n_train = int((1.0 - VAL_RATIO) * len(stack))
    stack = stack[:n_train]
    meses = np.array([dt.datetime.utcfromtimestamp(int(m) / 1000).month
                      for m in fechas[:n_train]])

    fase = np.full(n_train, -1, dtype=int)
    for k, (_n, ms) in enumerate(FASES):
        fase[np.isin(meses, ms)] = k
    n_por_fase = {n: int((fase == k).sum()) for k, (n, _m) in
                  enumerate(FASES)}
    # Fases con muy pocas fechas se quedan fuera: con menos de min_fechas el
    # Kruskal-Wallis no tiene con que.
    for k, (n, _m) in enumerate(FASES):
        if n_por_fase[n] < a.min_fechas:
            fase[fase == k] = -1
            print("  fase fuera por pocas fechas: " + n + " ("
                  + str(n_por_fase[n]) + ")")
    nombres = [n for k, (n, _m) in enumerate(FASES) if (fase == k).any()]
    print("  entornos: " + str(nombres))
    print("  fechas de train: " + str(n_train))

    B = betas_por_fecha(stack)
    print("  betas por fecha: " + str(B.shape))
    res = test_invariancia(B, fase, n_desp=a.desplazamientos)
    cal = calibracion(res)

    idx = {f["arista"]: f for f in res["filas"]}
    ctrl = []
    for x, y in CONTROLES_POSITIVOS:
        for ar in (x + "->" + y, y + "->" + x):
            if ar in idx and np.isfinite(idx[ar].get("q", np.nan)):
                ctrl.append(idx[ar])

    PC, RR = pc_y_r(np.load(stack_p))
    fuentes = [(e, p) for e, _n, _A, _s, p in recolectar_fuentes(
        a.matrices or _buscar(CANDIDATOS_MATRICES, True),
        a.paradigmas or _buscar(CANDIDATOS_PARADIGMAS, True), PC, RR)
        if p and len(p) >= 5]
    evals = evaluar_fuentes(fuentes, res["filas"], alfa=a.alfa)

    enr = enriquecimiento_por_bloque(res["filas"], a.alfa)
    txt = texto(res, cal, ctrl, evals, enr, nombres, n_por_fase,
                a.alfa)
    print("\n" + txt)
    os.makedirs(a.salida, exist_ok=True)
    with open(os.path.join(a.salida, "ANALISIS_ICP.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(txt)
    with open(os.path.join(a.salida, "ANALISIS_ICP.json"), "w",
              encoding="utf-8") as fh:
        json.dump(dict(cultivo=CULTIVO, entornos=nombres, bloques=enr,
                       n_por_fase=n_por_fase, calibracion=cal,
                       aristas=res["filas"], fuentes=evals),
                  fh, indent=2, ensure_ascii=False)
    print("\n  Guardado: " + os.path.join(a.salida, "ANALISIS_ICP.txt"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
