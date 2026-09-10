#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
El efecto por fase, sometido a los controles que faltaban.

DE QUE SE DUDA
--------------
El resultado en disputa es que el modelo reconstruye peor en dormancia y
postcosecha, y que el suelo de ruido de Datt sube ahi. Se dieron tres lecturas
sucesivas y las dos primeras se cayeron:

  1. "es biologia: sin hoja los indices degeneran"
     se cayo al ver que el MSE seguia al numero de fechas de entrenamiento;
  2. "es exposicion de entrenamiento"
     se debilito al reentrenar con exposicion balanceada sin que la brecha se
     cerrara, aunque ese control venia sesgado por la particion cronologica;
  3. "queda algo despues de arreglar el control"
     que es lo que hay que someter a prueba aqui.

Cada version se apoyaba en cinco numeros sin intervalo, uno por fase. Con cinco
puntos y sin incertidumbre casi cualquier historia encaja. Este modulo pone la
tercera version contra cuatro controles que puede fallar.

LOS CONTROLES
-------------
BOOTSTRAP SOBRE FECHAS. La unidad de muestreo es la fecha, no el parche: los 16
parches de una escena comparten atmosfera y estado del cultivo. Con 7 fechas en
la fase mas corta, el intervalo puede ser tan ancho que la diferencia entre
fases no signifique nada, y hasta ahora nadie lo habia mirado.

PLACEBO DE ETIQUETAS. Se reparten las fechas al azar en grupos de los mismos
tamanos que las fases. Si un grupo cualquiera de 8 fechas da el mismo salto que
dormancia, el salto es del tamano del grupo y no de la fenologia.

PLACEBO DE CALENDARIO ROTADO. Se giran los meses del calendario fenologico de 1
a 11 meses. Los grupos conservan su tamano, su recurrencia anual y su
contiguidad temporal, y lo unico que se rompe es la coincidencia con el ciclo
real de la vid. Es el control que separa "fenologia" de "cualquier tramo del
ano". Es exacto: hay doce rotaciones y se prueban todas.

IGUALAR FECHAS, MUCHAS VECES. datt_por_fase.py iguala el numero de fechas entre
fases escogiendo una submuestra fija. Aqui se repite el sorteo dos mil veces
para ver cuanto de lo reportado dependia de esa eleccion, y sobre todo para
contestar la pregunta decisiva del suelo de ruido: si se estima el suelo de
maduracion con solo 8 fechas, cuantas veces sale tan alto como el de dormancia.

Y ADEMAS, LA REGRESION POR FECHA
--------------------------------
Las dos explicaciones que compiten -exposicion de entrenamiento y biologia- se
habian comparado sobre una tabla de cinco filas. Con el error por fecha hay 186
puntos, y las dos se pueden meter en el mismo modelo. Con una advertencia que
el propio modulo mide: las dos van juntas en el calendario, porque la fase con
menos fechas de entrenamiento es tambien la que no tiene hoja. Si estan muy
correlacionadas la regresion no las separa, y entonces lo que corresponde es
decirlo, no elegir una.

Lo que si las separa: DENTRO de una fase la exposicion es constante y el NDVI
varia entre fechas. La correlacion intra-fase entre error y NDVI mide biologia
con la exposicion fijada por construccion.

USO
    .venv/bin/python control_fases.py --brazos _base _bal _ctrl _bal2
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from certificar_todo import INDEX_NAMES, N_IDX  # noqa: E402
from analisis_fenologico import FASES  # noqa: E402

RNG = np.random.default_rng(20260910)
NB = 2000          # remuestreos de bootstrap y de placebo
UMBRAL = 3.0       # multiplos del ruido para contar una arista como activa


# ------------------------------------------------------------------ utilidades

def cargar(salida: str, sufijo: str) -> Dict:
    ruta = os.path.join(salida, "ERRORES_POR_FECHA" + sufijo + ".npz")
    d = np.load(ruta, allow_pickle=True)
    E = d["E"].astype(np.float64)          # (semilla, fecha, canal, corte)
    out = dict(E=E, semillas=d["semillas"], meses=d["meses"],
               ndvi=d["ndvi"], sufijo=sufijo,
               # Datt por fecha: corte j+1 menos sin cortar
               Dd=E[:, :, :, 1:] - E[:, :, :, :1],
               mse=E[:, :, :, 0].mean(axis=2))   # (semilla, fecha)
    if "var_espacial" in d:
        # MSE normalizado por lo que sacaria el predictor trivial de esa
        # escena: el que devuelve la media espacial del canal tapado sin mirar
        # los otros once. Su error es exactamente la varianza espacial del
        # canal. Dividir por ella deja el MSE en unidades de "que fraccion de
        # lo que habia que explicar quedo sin explicar", que es comparable
        # entre escenas faciles y dificiles.
        v = d["var_espacial"].astype(np.float64)
        out["var_espacial"] = v
        out["nmse"] = (E[:, :, :, 0] / np.maximum(v, 1e-12)[None]).mean(axis=2)
        out["dificultad"] = v.mean(axis=1)
    return out


def grupos_de(meses: np.ndarray, fases=None) -> List[Tuple[str, np.ndarray]]:
    fases = FASES if fases is None else fases
    out = []
    for nombre, ms in fases:
        idx = np.where(np.isin(meses, ms))[0]
        if len(idx):
            out.append((nombre, idx))
    return out


def rotar(k: int):
    """El calendario fenologico girado k meses.

    Conserva exactamente la forma de las fases -cuatro meses, dos, dos, tres,
    uno- y su recurrencia anual. Solo deja de coincidir con el ciclo de la vid.
    """
    return [(n + " +%d" % k, tuple(((m - 1 + k) % 12) + 1 for m in ms))
            for n, ms in FASES]


def suelo(D: np.ndarray) -> float:
    """Suelo de ruido: magnitud tipica de la cola negativa de Datt.

    Cortar una arista no puede MEJORAR la reconstruccion salvo por error de
    estimacion, asi que lo negativo mide el error de estimacion.
    """
    off = ~np.eye(D.shape[0], dtype=bool)
    neg = D[off & (D < 0)]
    return float(np.abs(neg).mean()) if neg.size else float("nan")


def datt_de(d: Dict, idx: np.ndarray) -> np.ndarray:
    """Matriz Datt promediando las fechas dadas y todas las semillas."""
    D = d["Dd"][:, idx].mean(axis=(0, 1))
    np.fill_diagonal(D, 0.0)
    return D


def ic(v: np.ndarray, q=(2.5, 97.5)) -> Tuple[float, float]:
    return float(np.percentile(v, q[0])), float(np.percentile(v, q[1]))


# --------------------------------------------------------------- los controles

def tabla_por_fase(d: Dict) -> List[Dict]:
    """MSE por fase con dos intervalos: sobre fechas y sobre semillas.

    Los dos hacen falta y miden cosas distintas. El de fechas dice si con 7 u 8
    escenas se puede afirmar algo de esa fase. El de semillas dice cuanto de la
    diferencia es el azar de la inicializacion.
    """
    filas = []
    for nombre, idx in grupos_de(d["meses"]):
        por_fecha = d["mse"].mean(axis=0)[idx]        # promediando semillas
        por_semilla = d["mse"][:, idx].mean(axis=1)   # promediando fechas
        bs = np.array([por_fecha[RNG.integers(0, len(idx), len(idx))].mean()
                       for _ in range(NB)])
        bss = np.array([por_semilla[RNG.integers(0, len(por_semilla),
                                                 len(por_semilla))].mean()
                        for _ in range(NB)])
        filas.append(dict(fase=nombre, n_fechas=int(len(idx)),
                          mse=float(por_fecha.mean()),
                          ic_fechas=ic(bs), ic_semillas=ic(bss),
                          sd_semillas=float(por_semilla.std(ddof=1))))
    return filas


def razon_con_ic(d: Dict, a: str, b: str, clave="mse") -> Dict:
    """Razon entre dos fases, con bootstrap emparejado sobre fechas."""
    g = dict(grupos_de(d["meses"]))
    ia, ib = g[a], g[b]
    m = d[clave].mean(axis=0)
    r = float(m[ia].mean() / m[ib].mean())
    bs = np.array([m[ia[RNG.integers(0, len(ia), len(ia))]].mean()
                   / m[ib[RNG.integers(0, len(ib), len(ib))]].mean()
                   for _ in range(NB)])
    return dict(razon=r, ic=ic(bs),
                p_mayor_que_uno=float((bs <= 1.0).mean()))


def placebo_etiquetas(d: Dict, a: str, b: str, clave="mse") -> Dict:
    """Grupos al azar del mismo tamano. Contesta: un grupo cualquiera de 8
    fechas, da el mismo salto?"""
    g = dict(grupos_de(d["meses"]))
    na, nb = len(g[a]), len(g[b])
    m = d[clave].mean(axis=0)
    obs = float(m[g[a]].mean() / m[g[b]].mean())
    nulo = np.empty(NB)
    n = len(m)
    for k in range(NB):
        p = RNG.permutation(n)
        nulo[k] = m[p[:na]].mean() / m[p[na:na + nb]].mean()
    return dict(observado=obs, nulo_mediana=float(np.median(nulo)),
                nulo_ic=ic(nulo), p=float((nulo >= obs).mean() + 1 / NB))


def razon_con_n_igualado(d: Dict, a: str, b: str) -> Dict:
    """La razon como la calcula datt_por_fase.py, pero sorteando muchas veces.

    Aquel modulo iguala el numero de fechas entre fases con una submuestra fija
    -np.linspace sobre los indices- y publica un solo numero. Esa eleccion es
    arbitraria, y con 9 fechas de 90 puede caer en un tramo favorable. Aqui se
    repite el sorteo para ver la distribucion que hay detras de ese numero, y
    de paso donde cae exactamente la submuestra que se publico.
    """
    g = dict(grupos_de(d["meses"]))
    ia, ib = g[a], g[b]
    n = min(len(ia), len(ib))
    m = d["mse"].mean(axis=0)
    out = np.empty(NB)
    for k in range(NB):
        out[k] = (m[RNG.choice(ia, n, replace=False)].mean()
                  / m[RNG.choice(ib, n, replace=False)].mean())
    fijo = (m[ia[np.linspace(0, len(ia) - 1, n).astype(int)]].mean()
            / m[ib[np.linspace(0, len(ib) - 1, n).astype(int)]].mean())
    return dict(n=int(n), mediana=float(np.median(out)), ic=ic(out),
                submuestra_publicada=float(fijo),
                percentil_publicada=float((out <= fijo).mean() * 100))


def dificultad_de_la_escena(d: Dict, a: str, b: str) -> Dict:
    """El control que separa "el modelo falla ahi" de "esa escena es dificil".

    Un MSE alto tiene dos causas posibles que nada de lo anterior distingue: el
    modelo no aprendio ese regimen, o la escena tiene mas varianza espacial y
    cualquier predictor fallaria mas. En un vinedo en invierno, con el dosel
    caido, suelo desnudo entre hileras y sombras largas, la segunda no es una
    posibilidad remota.

    La referencia es el predictor trivial: devolver la media espacial del canal
    tapado sin mirar los otros once. Su error es la varianza espacial de ese
    canal. El cociente MSE/varianza es la fraccion que quedo sin explicar, y ya
    no depende de lo dificil que sea la escena.

    Si la brecha entre fases desaparece al normalizar, lo que se estaba
    midiendo era la heterogeneidad del invierno y no una falla del modelo.
    """
    if "nmse" not in d:
        return {}
    g = dict(grupos_de(d["meses"]))
    cruda = razon_con_ic(d, a, b, "mse")
    norm = razon_con_ic(d, a, b, "nmse")
    dif = d["dificultad"]
    cerrada = ((cruda["razon"] - norm["razon"])
               / max(cruda["razon"] - 1.0, 1e-9))
    return dict(razon_cruda=cruda["razon"], razon_normalizada=norm["razon"],
                ic_normalizada=norm["ic"],
                p_normalizada=placebo_etiquetas(d, a, b, "nmse")["p"],
                brecha_cerrada=float(cerrada),
                dificultad_a=float(dif[g[a]].mean()),
                dificultad_b=float(dif[g[b]].mean()),
                razon_dificultad=float(dif[g[a]].mean() / dif[g[b]].mean()))


def placebo_calendario(d: Dict, i_a: int, i_b: int) -> Dict:
    """Las once rotaciones del calendario. Contesta: es la fenologia de la vid
    o es cualquier tramo del ano con esa forma?

    Exacto y con doce valores posibles, asi que el p mas pequeno alcanzable es
    1/12 = 0.083. Un p de 0.083 aqui es la evidencia maxima que este control
    puede dar; no puede dar menos que eso y hay que leerlo asi.
    """
    m = d["mse"].mean(axis=0)
    vals = []
    for k in range(12):
        gs = grupos_de(d["meses"], rotar(k))
        if len(gs) < max(i_a, i_b) + 1:
            vals.append(np.nan)
            continue
        vals.append(float(m[gs[i_a][1]].mean() / m[gs[i_b][1]].mean()))
    v = np.array(vals)
    obs = v[0]
    otras = v[1:][np.isfinite(v[1:])]
    return dict(observado=float(obs), rotadas=[float(x) for x in otras],
                rotadas_max=float(otras.max()) if otras.size else float("nan"),
                p=float((np.sum(otras >= obs) + 1) / (len(otras) + 1)))


def suelo_con_n_igualado(d: Dict, fase: str, n: int, reps=NB) -> Dict:
    """Suelo de ruido de una fase estimado con solo n fechas, muchas veces.

    Es la prueba decisiva del suelo. El suelo se estima de la cola negativa de
    Datt, y esa cola se encoge cuando hay mas fechas porque el error de
    estimacion baja. Comparar el suelo de dormancia -8 fechas- con el de
    maduracion -76- es comparar dos estimadores con precisiones distintas. Aqui
    se le da a maduracion el mismo numero de fechas que a dormancia y se mira
    la distribucion.
    """
    g = dict(grupos_de(d["meses"]))
    idx = g[fase]
    n = min(n, len(idx))
    # Cuando n es el total de la fase, sortear sin reemplazo devuelve siempre
    # el mismo conjunto y el intervalo sale de ancho cero, que es un intervalo
    # falso y no un intervalo estrecho. En ese caso se remuestrea con
    # reemplazo, que es el bootstrap de verdad sobre esas fechas.
    con_reemplazo = (n == len(idx))
    Dm = d["Dd"].mean(axis=0)          # (fecha, canal, canal)
    out = np.empty(reps)
    for k in range(reps):
        sel = RNG.choice(idx, size=n, replace=con_reemplazo)
        D = Dm[sel].mean(axis=0)
        np.fill_diagonal(D, 0.0)
        out[k] = suelo(D)
    D0 = Dm[idx if con_reemplazo else RNG.choice(idx, n, False)].mean(axis=0)
    np.fill_diagonal(D0, 0.0)
    return dict(n=int(n), punto=float(suelo(D0)),
                mediana=float(np.median(out)), ic=ic(out),
                con_reemplazo=bool(con_reemplazo), muestras=out)


def regresion_por_fecha(d: Dict, n_train_por_fase: Dict[str, int]) -> Dict:
    """Exposicion contra biologia, con 186 puntos en vez de cinco.

    Regresores estandarizados: log del numero de fechas de ENTRENAMIENTO de la
    fase a la que pertenece la fecha, y NDVI medio de esa fecha. El primero es
    "cuanto vio el modelo este regimen", el segundo "cuanta hoja habia".

    La correlacion entre los dos se reporta siempre. Si es alta, la regresion
    no los separa y lo que corresponde es decirlo.
    """
    meses = d["meses"]
    expo = np.zeros(len(meses))
    for nombre, idx in grupos_de(meses):
        expo[idx] = n_train_por_fase.get(nombre, np.nan)
    y = np.log(d["mse"].mean(axis=0))
    x1 = np.log(expo)
    x2 = d["ndvi"].astype(np.float64)
    # Tercer competidor: lo dificil que es la escena, medida como la varianza
    # espacial media. Sin el, "el modelo falla en invierno" y "las escenas de
    # invierno son mas heterogeneas" se confunden en el mismo coeficiente.
    x3 = (np.log(d["dificultad"]) if "dificultad" in d
          else np.zeros(len(y)))
    ok = (np.isfinite(y) & np.isfinite(x1) & np.isfinite(x2)
          & np.isfinite(x3))
    y, x1, x2, x3 = y[ok], x1[ok], x2[ok], x3[ok]
    z = lambda v: ((v - v.mean()) / v.std() if v.std() > 0  # noqa: E731
                   else np.zeros_like(v))
    X = np.column_stack([np.ones(len(y)), z(x1), z(x2), z(x3)])
    beta, *_ = np.linalg.lstsq(X, z(y), rcond=None)
    resid = z(y) - X @ beta
    r2 = 1.0 - resid.var() / z(y).var()

    # Dentro de fase la exposicion es constante por construccion, asi que lo
    # que quede de asociacion con el NDVI es biologia sin confundir.
    intra = []
    for nombre, idx in grupos_de(meses):
        if len(idx) >= 6:
            r = stats.spearmanr(d["mse"].mean(axis=0)[idx], x2[idx])
            intra.append((nombre, int(len(idx)), float(r.statistic),
                          float(r.pvalue)))
    return dict(beta_exposicion=float(beta[1]), beta_ndvi=float(beta[2]),
                beta_dificultad=float(beta[3]), r2=float(r2),
                corr_regresores=float(stats.spearmanr(x1, x2).statistic),
                rho_expo=float(stats.spearmanr(x1, y).statistic),
                rho_ndvi=float(stats.spearmanr(x2, y).statistic),
                rho_dificultad=float(stats.spearmanr(x3, y).statistic),
                intra_fase=intra)


def prueba_icp(d: Dict, salida: str) -> Dict:
    """La persistencia de Datt contra las etiquetas del ICP, con permutacion
    exacta y con la potencia declarada.

    Con 19 aristas y 7 invariantes hay C(19,7) = 50388 repartos posibles, asi
    que el p exacto se enumera sin aproximar. Y se reporta el p MINIMO
    alcanzable: si las 7 invariantes fueran justo las 7 de mayor puntaje. Sin
    esa cifra un p alto no distingue "no hay senal" de "el test no podia
    detectarla".
    """
    with open(os.path.join(salida, "ABLACION_ATENCION_base.json"),
              encoding="utf-8") as f:
        b = json.load(f)
    Dg, rg = np.array(b["Datt"]), float(b["ruido"])
    ref = [(i, j) for i in range(N_IDX) for j in range(N_IDX)
           if i != j and Dg[i, j] > UMBRAL * rg]
    with open(os.path.join(salida, "ANALISIS_ICP.json"), encoding="utf-8") as f:
        q = {(x["i"], x["j"]): x["q"] for x in json.load(f)["aristas"]}

    gs = grupos_de(d["meses"])
    Ds, ruidos = {}, {}
    for nombre, idx in gs:
        D = datt_de(d, idx)
        Ds[nombre], ruidos[nombre] = D, suelo(D)

    filas = []
    for (i, j) in sorted(ref, key=lambda t: -Dg[t[0], t[1]]):
        veces = {n: float(Ds[n][i, j] / ruidos[n]) for n, _ in gs}
        filas.append(dict(
            arista=INDEX_NAMES[i] + " <- " + INDEX_NAMES[j],
            i=int(i), j=int(j),
            invariante=bool(q.get((i, j), 0.0) > 0.10),
            veces=veces,
            n_sobre=int(sum(1 for v in veces.values() if v > UMBRAL)),
            mediana=float(np.median(list(veces.values())))))

    def exacto(puntajes, es_inv):
        n, k = len(puntajes), int(sum(es_inv))
        obs = float(np.sum(np.asarray(puntajes)[np.asarray(es_inv)]))
        todos = np.array([sum(c) for c in
                          itertools.combinations(puntajes, k)])
        return dict(obs=obs, p=float((todos >= obs).mean()),
                    p_minimo=float((todos >= todos.max()).mean()),
                    n_repartos=int(len(todos)))

    es_inv = [f["invariante"] for f in filas]
    return dict(aristas=filas,
                por_fase_ruido={n: ruidos[n] for n, _ in gs},
                discreto=exacto([f["n_sobre"] for f in filas], es_inv),
                continuo=exacto([f["mediana"] for f in filas], es_inv),
                n_invariantes=int(sum(es_inv)),
                n_regimen=int(len(filas) - sum(es_inv)))


# ------------------------------------------------------------------- el informe

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brazos", nargs="+",
                    default=["_base", "_bal", "_ctrl", "_bal2"])
    ap.add_argument("--salida", default="resultados")
    ap.add_argument("--fase-alta", default="dormancia (may-ago)",
                    dest="alta")
    ap.add_argument("--fase-baja", default="maduracion (ene-mar)",
                    dest="baja")
    a = ap.parse_args()

    datos, faltan = {}, []
    for suf in a.brazos:
        try:
            datos[suf] = cargar(a.salida, suf)
        except FileNotFoundError:
            faltan.append(suf)
    if not datos:
        raise SystemExit("no hay ningun ERRORES_POR_FECHA*.npz en " + a.salida)

    W = 82
    L = ["=" * W, " EL EFECTO POR FASE, CONTRA CUATRO CONTROLES", "",
         " La afirmacion en prueba: el modelo reconstruye peor en dormancia y",
         " postcosecha, y el suelo de ruido de Datt sube ahi, por algo que no",
         " es el numero de fechas de entrenamiento.",
         "=" * W]
    if faltan:
        L.append("  brazos sin datos: " + " ".join(faltan))

    # exposicion de entrenamiento del brazo original: primeras 80% de fechas
    d0 = datos[a.brazos[0] if a.brazos[0] in datos else list(datos)[0]]
    n_tr = int(round(0.8 * len(d0["meses"])))
    n_train_fase = {n: int(np.isin(d0["meses"][:n_tr], ms).sum())
                    for n, ms in FASES}

    informe = {}
    for suf, d in datos.items():
        L += ["", "=" * W, " BRAZO " + suf.lstrip("_").upper()
              + "   (%d semillas, %d fechas)" % (len(d["semillas"]),
                                                 len(d["meses"])),
              "=" * W, ""]

        # 1. tabla con incertidumbre
        filas = tabla_por_fase(d)
        L.append("  MSE POR FASE, CON INTERVALOS AL 95%")
        L.append("  " + "fase".ljust(24) + "n".rjust(4) + "MSE".rjust(9)
                 + "IC sobre fechas".rjust(22)
                 + "IC sobre semillas".rjust(22))
        for f in filas:
            L.append("  " + f["fase"][:23].ljust(24)
                     + str(f["n_fechas"]).rjust(4)
                     + ("%.4f" % f["mse"]).rjust(9)
                     + ("[%.4f, %.4f]" % f["ic_fechas"]).rjust(22)
                     + ("[%.4f, %.4f]" % f["ic_semillas"]).rjust(22))
        L.append("")
        L.append("  El intervalo sobre fechas es el que manda: con 7 u 8")
        L.append("  escenas puede ser tan ancho que la fase no diga nada.")

        # 2. la razon en disputa
        r = razon_con_ic(d, a.alta, a.baja)
        L += ["", "  RAZON " + a.alta.split()[0] + " / " + a.baja.split()[0]
              + ":  %.2f   IC95 [%.2f, %.2f]" % (r["razon"], r["ic"][0],
                                                 r["ic"][1]),
              "    fraccion del bootstrap que no supera 1: %.4f"
              % r["p_mayor_que_uno"]]

        rn = razon_con_n_igualado(d, a.alta, a.baja)
        L += ["", "  LA MISMA RAZON IGUALANDO FECHAS (n = %d cada una), "
              "sorteada %d veces" % (rn["n"], NB),
              "    mediana %.2f   IC95 [%.2f, %.2f]"
              % (rn["mediana"], rn["ic"][0], rn["ic"][1]),
              "    la submuestra fija que publica datt_por_fase da %.2f, "
              "percentil %.0f" % (rn["submuestra_publicada"],
                                  rn["percentil_publicada"]),
              "    Un percentil cerca de 50 dice que aquella eleccion no "
              "sesgaba;",
              "    cerca de 0 o de 100, que el numero publicado dependia de "
              "ella."]

        # 3. placebos
        pe = placebo_etiquetas(d, a.alta, a.baja)
        L += ["", "  PLACEBO DE ETIQUETAS (grupos al azar del mismo tamano)",
              "    observado %.2f   nulo mediana %.2f  IC95 [%.2f, %.2f]"
              % (pe["observado"], pe["nulo_mediana"], pe["nulo_ic"][0],
                 pe["nulo_ic"][1]),
              "    p = %.4f" % pe["p"],
              "    Contesta: un grupo cualquiera de ese tamano da el mismo",
              "    salto? Si el p es alto, el salto es del tamano del grupo."]

        i_a = [n for n, _ in FASES].index(a.alta)
        i_b = [n for n, _ in FASES].index(a.baja)
        pc = placebo_calendario(d, i_a, i_b)
        L += ["", "  PLACEBO DE CALENDARIO ROTADO (11 giros del ciclo)",
              "    observado %.2f   maximo de las rotadas %.2f   p = %.4f"
              % (pc["observado"], pc["rotadas_max"], pc["p"]),
              "    rotadas: " + " ".join("%.2f" % x for x in pc["rotadas"]),
              "    El p minimo alcanzable con doce rotaciones es 0.0833.",
              "    Contesta: es la fenologia de la vid, o cualquier tramo",
              "    del ano con esa misma forma?"]

        # 4. el suelo de ruido con n igualado
        n_alta = int(np.isin(d["meses"], dict(FASES)[a.alta]).sum())
        s_alta = suelo_con_n_igualado(d, a.alta, n_alta)
        s_baja = suelo_con_n_igualado(d, a.baja, n_alta)
        p_suelo = float((s_baja["muestras"] >= s_alta["mediana"]).mean())
        L += ["", "  SUELO DE RUIDO CON EL MISMO NUMERO DE FECHAS (n = %d)"
              % n_alta,
              "    %-22s mediana %.3e  IC95 [%.3e, %.3e]%s"
              % (a.alta.split()[0], s_alta["mediana"], s_alta["ic"][0],
                 s_alta["ic"][1],
                 "  (bootstrap, usa todas sus fechas)"
                 if s_alta["con_reemplazo"] else ""),
              "    %-22s mediana %.3e  IC95 [%.3e, %.3e]"
              % (a.baja.split()[0], s_baja["mediana"], s_baja["ic"][0],
                 s_baja["ic"][1]),
              "    razon de medianas %.2f   p = %.4f"
              % (s_alta["mediana"] / s_baja["mediana"], p_suelo),
              "    p = con cuanta frecuencia " + a.baja.split()[0]
              + ", estimada con las mismas",
              "    %d fechas, alcanza el suelo de %s." % (n_alta,
                                                          a.alta.split()[0])]

        # 4b. dificultad de la escena
        dif = dificultad_de_la_escena(d, a.alta, a.baja)
        if dif:
            L += ["", "  DIFICULTAD DE LA ESCENA (contra el predictor "
                  "trivial)",
                  "    varianza espacial media: %s %.3f, %s %.3f  "
                  "(razon %.2f)"
                  % (a.alta.split()[0], dif["dificultad_a"],
                     a.baja.split()[0], dif["dificultad_b"],
                     dif["razon_dificultad"]),
                  "    razon de MSE cruda        %.2f" % dif["razon_cruda"],
                  "    razon ya normalizada      %.2f   IC95 [%.2f, %.2f]  "
                  "p = %.4f"
                  % (dif["razon_normalizada"], dif["ic_normalizada"][0],
                     dif["ic_normalizada"][1], dif["p_normalizada"]),
                  "    la normalizacion cierra el %.0f%% de la brecha"
                  % (100 * dif["brecha_cerrada"]),
                  "    Si la razon normalizada baja a 1, lo que se medía era",
                  "    la heterogeneidad del invierno y no el modelo."]

        # 5. exposicion contra biologia
        rg = regresion_por_fecha(d, n_train_fase)
        L += ["", "  QUE PREDICE EL ERROR DE UNA FECHA, 186 FECHAS",
              "    coeficientes estandarizados en un modelo con los tres:",
              "      exposicion (log fechas de train de su fase) %+.3f"
              % rg["beta_exposicion"],
              "      NDVI de la fecha                            %+.3f"
              % rg["beta_ndvi"],
              "      dificultad (log varianza espacial)          %+.3f"
              % rg["beta_dificultad"],
              "    R2 %.3f" % rg["r2"],
              "    rho de Spearman por separado: exposicion %+.3f, "
              "NDVI %+.3f, dificultad %+.3f"
              % (rg["rho_expo"], rg["rho_ndvi"], rg["rho_dificultad"]),
              "    correlacion entre exposicion y NDVI: %+.3f"
              % rg["corr_regresores"]]
        if abs(rg["corr_regresores"]) > 0.7:
            L.append("    Con esa correlacion los dos regresores no se "
                     "separan: los coeficientes")
            L.append("    de arriba se reparten un efecto comun y no hay que "
                     "leerlos por separado.")
        L.append("    Intra-fase (exposicion constante por construccion), "
                 "rho MSE vs NDVI:")
        for nombre, n, rho, p in rg["intra_fase"]:
            L.append("      %-24s n=%2d  rho %+.3f  p %.4f"
                     % (nombre[:24], n, rho, p))

        # 6. el ICP
        icp = prueba_icp(d, a.salida)
        L += ["", "  PERSISTENCIA DE Datt CONTRA LAS ETIQUETAS DEL ICP",
              "    %d invariantes y %d de regimen, %d repartos posibles"
              % (icp["n_invariantes"], icp["n_regimen"],
                 icp["discreto"]["n_repartos"]),
              "    conteo de fases sobre 3x: p exacto %.4f  "
              "(minimo alcanzable %.4f)"
              % (icp["discreto"]["p"], icp["discreto"]["p_minimo"]),
              "    mediana de Datt/ruido:    p exacto %.4f  "
              "(minimo alcanzable %.4f)"
              % (icp["continuo"]["p"], icp["continuo"]["p_minimo"]),
              "    El minimo alcanzable dice si el test podia detectar algo.",
              "    Si es pequeno y el p sale alto, no hay senal; si el minimo",
              "    ya es grande, el test no tenia con que."]
        informe[suf] = dict(
            tabla=[{k: v for k, v in f.items()} for f in filas],
            razon=r, razon_n_igualado=rn,
            placebo_etiquetas={k: v for k, v in pe.items()},
            placebo_calendario=pc, dificultad=dif,
            suelo_igualado=dict(
                n=n_alta, alta=s_alta["mediana"], baja=s_baja["mediana"],
                ic_alta=s_alta["ic"], ic_baja=s_baja["ic"], p=p_suelo),
            regresion=rg,
            icp={k: v for k, v in icp.items() if k != "aristas"},
            aristas=icp["aristas"])

    # comparacion entre brazos
    if len(datos) > 1:
        L += ["", "=" * W, " LOS BRAZOS, UNO AL LADO DEL OTRO", "=" * W, "",
              "  " + "brazo".ljust(8) + "razon".rjust(7) + "IC95".rjust(16)
              + "normaliz".rjust(10) + "placebo p".rjust(11)
              + "rotado p".rjust(10) + "suelo p".rjust(9) + "ICP p".rjust(8)]
        for suf, inf in informe.items():
            L.append("  " + suf.lstrip("_").ljust(8)
                     + ("%.2f" % inf["razon"]["razon"]).rjust(7)
                     + ("[%.2f, %.2f]" % tuple(inf["razon"]["ic"])).rjust(16)
                     + (("%.2f" % inf["dificultad"]["razon_normalizada"])
                        if inf["dificultad"] else "-").rjust(10)
                     + ("%.4f" % inf["placebo_etiquetas"]["p"]).rjust(11)
                     + ("%.4f" % inf["placebo_calendario"]["p"]).rjust(10)
                     + ("%.4f" % inf["suelo_igualado"]["p"]).rjust(9)
                     + ("%.4f" % inf["icp"]["continuo"]["p"]).rjust(8))
        L += ["",
              "  Un resultado que sobrevive a los cinco controles en los",
              "  cuatro brazos es un resultado. Uno que solo aparece en un",
              "  brazo es una propiedad de ese entrenamiento."]
    L.append("=" * W)

    txt = "\n".join(L)
    print(txt)
    os.makedirs(a.salida, exist_ok=True)
    with open(os.path.join(a.salida, "CONTROL_FASES.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt)
    with open(os.path.join(a.salida, "CONTROL_FASES.json"), "w",
              encoding="utf-8") as f:
        json.dump(informe, f, indent=2, ensure_ascii=False,
                  default=lambda o: (o.tolist() if isinstance(o, np.ndarray)
                                     else float(o)))
    print("\n  Guardado " + os.path.join(a.salida, "CONTROL_FASES.txt"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
