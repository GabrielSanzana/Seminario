#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Certificacion contra verdad conocida. INDEPENDIENTE del pipeline.

No importa "pipeline (2).py", no entrena, no toca la GPU. Solo lee ficheros
que la corrida ya dejo escritos, asi que funciona con cualquier version del
pipeline y tarda segundos.

QUE RESPONDE. El framework ordena los pares de indices como la dependencia
CONDICIONAL (lo que la tesis afirma medir) o como la correlacion MARGINAL (lo
que ya da np.corrcoef)? Y sobre todo: pone ARRIBA los pares que de verdad
existen?

ENTRADA (se buscan solas, o se pasan por argumento)
    salidas/multi_indices/indices_12.npy            el stack de 12 indices
    resultados/sensitivity_K_experiment/matrices/   attention_seed_*.npy
                                                    dependencia_seed_*.npy

SALIDA
    resultados/CERTIFICACION.json
    resultados/CERTIFICACION.txt

USO
    .venv/bin/python certificar.py
    .venv/bin/python certificar.py --matrices otra/carpeta --stack otro.npy
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

# Orden fijo de los 12 indices. Es el mismo de INDEX_NAMES en el pipeline y no
# ha cambiado en ninguna version, asi que se replica aqui para no depender de
# poder importar el pipeline.
INDEX_NAMES: List[str] = [
    "NDVI", "MARI", "ARI", "EVI", "EVI2", "NDWI",
    "NDMI", "CHL_REDEDGE", "NDII", "SAVI", "PSRI", "KNDVI",
]

# Fraccion de validacion del pipeline. Los baselines se calculan solo sobre el
# tramo de TRAIN: darles el bloque de validacion seria regalarles informacion
# que el modelo no vio y la comparacion dejaria de ser honesta.
VAL_RATIO = 0.2

# Identidades algebraicas EXACTAS sobre este stack, verificadas numericamente:
# KNDVI = tanh(NDVI^2) y MARI = ARI*B7. No son hipotesis, son verdad por
# construccion, asi que sirven de control positivo: un framework que no las
# encuentre no puede pedir que se le crean las que si encuentra.
CONTROLES_POSITIVOS: List[Tuple[str, str]] = [("NDVI", "KNDVI"), ("MARI", "ARI")]

CANDIDATOS_STACK = [
    "salidas/multi_indices/indices_12.npy",
    "multi_indices_extraido/indices_12.npy",
    "resultados/indices_12.npy",
    "indices_12.npy",
]
# El pipeline ha vivido con dos nombres (hello.py en el server, "pipeline (2).py"
# en el portatil). certificar.py NO lo importa —es independiente a proposito—
# pero si busca las matrices en las rutas que ese pipeline escribe.
CANDIDATOS_MATRICES = [
    "resultados/sensitivity_K_experiment/matrices",
    "resultados/paradigms_experiment/matrices",
    "resultados/stability_experiment/matrices",
    "resultados/stability_50_experiment/matrices",
]


# ---------------------------------------------------------------------------
# Baselines de verdad
# ---------------------------------------------------------------------------

def pc_y_r(stack: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """|correlacion parcial| y |correlacion marginal|, sobre el tramo de train.

    La parcial es la matriz de precision normalizada: mide la dependencia de i
    con j CONTROLANDO por los otros 10. Es la que el framework dice medir. La
    marginal es lo que da np.corrcoef y sobre estos 12 indices es densa y casi
    inutil (|r| medio 0.836): un metodo que solo la reproduzca no aporta nada.
    """
    a = np.asarray(stack, dtype=np.float64)
    a = a[:int((1.0 - VAL_RATIO) * len(a))]
    X = a.reshape(-1, a.shape[-1])
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    C = np.corrcoef(X.T)
    Pm = np.linalg.inv(C + 1e-8 * np.eye(C.shape[0]))
    d = np.sqrt(np.diag(Pm))
    return np.abs(-Pm / np.outer(d, d)), np.abs(C)


def fuerza_por_par(A: np.ndarray) -> np.ndarray:
    """max(A[i,j], A[j,i]) sobre los N(N-1)/2 pares no dirigidos.

    Se simetriza porque la correlacion parcial no tiene direccion; comparar un
    grafo dirigido contra un baseline simetrico sin declararlo penaliza al
    grafo por algo que el baseline no puede expresar.
    """
    n = A.shape[0]
    return np.array([max(A[i, j], A[j, i])
                     for i in range(n) for j in range(i + 1, n)])


# ---------------------------------------------------------------------------
# Certificacion
# ---------------------------------------------------------------------------

def certificar(matrices: List[np.ndarray], etiqueta: str,
               PC: np.ndarray, RR: np.ndarray) -> Dict:
    n = PC.shape[0]
    pares = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pc = np.array([PC[i, j] for i, j in pares])
    rr = np.array([RR[i, j] for i, j in pares])

    A_med = np.mean(np.stack([np.asarray(m, dtype=np.float64)
                              for m in matrices]), axis=0)
    fr = fuerza_por_par(A_med)
    s_cond = stats.spearmanr(fr, pc)
    s_marg = stats.spearmanr(fr, rr)

    # Estabilidad entre semillas: si la condicional solo gana al promediar, es
    # un artefacto del promedio y no una propiedad de los modelos.
    por_semilla = []
    for m in matrices:
        f_m = fuerza_por_par(np.asarray(m, dtype=np.float64))
        rc = stats.spearmanr(f_m, pc).statistic
        rm = stats.spearmanr(f_m, rr).statistic
        if np.isfinite(rc) and np.isfinite(rm):
            por_semilla.append((float(rc), float(rm)))
    arr = np.array(por_semilla) if por_semilla else np.zeros((0, 2))
    gana_cond = float((arr[:, 0] > arr[:, 1]).mean()) if arr.size else float("nan")

    orden = np.argsort(-fr)
    puesto = np.empty(len(fr), dtype=int)
    puesto[orden] = np.arange(1, len(fr) + 1)
    idx_par = {p: k for k, p in enumerate(pares)}
    orden_pc = np.argsort(-pc)
    puesto_pc = np.empty(len(pc), dtype=int)
    puesto_pc[orden_pc] = np.arange(1, len(pc) + 1)

    controles = []
    for na, nb in CONTROLES_POSITIVOS:
        if na not in INDEX_NAMES or nb not in INDEX_NAMES:
            continue
        i, j = INDEX_NAMES.index(na), INDEX_NAMES.index(nb)
        k = idx_par[(min(i, j), max(i, j))]
        controles.append(dict(par=f"{na}-{nb}", pc=float(pc[k]),
                              puesto_pc=int(puesto_pc[k]),
                              puesto_framework=int(puesto[k]),
                              n_pares=len(pares)))
    # Criterio laxo a proposito (tercio superior): si ni asi encuentra una
    # identidad exacta, el problema no es de calibracion fina.
    umbral_ctrl = max(1, len(pares) // 3)
    n_ok = sum(1 for c in controles if c["puesto_framework"] <= umbral_ctrl)

    top_reales = [dict(par=f"{INDEX_NAMES[pares[k][0]]}-{INDEX_NAMES[pares[k][1]]}",
                       pc=float(pc[k]), puesto=int(puesto[k]))
                  for k in orden_pc[:6]]

    # ---- TEST DE ENRIQUECIMIENTO ----
    # El Spearman global pregunta "coincide el ranking ENTERO", y de los 66
    # pares hay ~60 con |pc| pequeno cuyo orden es ruido: diluyen el
    # estadistico. Medido el 08/09, la ablacion dio Spearman +0.087 (p=0.49, o
    # sea nada) mientras ponia 4 de los 6 pares reales en el top-18 de 66
    # (esperado por azar 1.6). El Spearman decia que no pasaba nada y si pasaba.
    # La pregunta de la tesis es "pone arriba las que existen", y eso se mide
    # con rank-sum e hipergeometrica. Se barre K de 3 a 12 y se reporta la
    # curva entera: la distribucion de |pc| no tiene salto que justifique un
    # corte concreto, y un resultado que solo aparece a un K es seleccion del
    # corte, no hallazgo.
    enriq = []
    for K in range(3, min(13, len(pares))):
        sel = set(int(t) for t in orden_pc[:K])
        r_sel = np.array([puesto[k] for k in sel], dtype=float)
        r_res = np.array([puesto[k] for k in range(len(pares))
                          if k not in sel], dtype=float)
        try:
            p_rs = float(stats.mannwhitneyu(r_sel, r_res,
                                            alternative="less").pvalue)
        except Exception:
            p_rs = float("nan")
        en_top = int((r_sel <= K).sum())
        enriq.append(dict(K=K, p_rank_sum=p_rs, en_top_K=en_top,
                          esperado=float(K * K / len(pares)),
                          p_hipergeom=float(stats.hypergeom.sf(
                              en_top - 1, len(pares), K, K)),
                          puesto_mediano=float(np.median(r_sel))))
    ps = np.array([e["p_rank_sum"] for e in enriq], dtype=float)
    ps = ps[np.isfinite(ps)]
    p_med = float(np.median(ps)) if ps.size else float("nan")
    frac_sig = float((ps < 0.05).mean()) if ps.size else float("nan")
    encuentra = bool(p_med < 0.05)

    if s_cond.statistic > s_marg.statistic and s_cond.pvalue < 0.05:
        forma = "SIGUE DEPENDENCIA CONDICIONAL"
    elif s_marg.statistic > s_cond.statistic and s_marg.pvalue < 0.05:
        forma = "SIGUE CORRELACION MARGINAL (atajo colineal)"
    else:
        forma = "NO DISTINGUE (ninguna de las dos domina)"
    if encuentra and "MARGINAL" not in forma:
        veredicto = f"ENCUENTRA LOS PARES REALES · {forma}"
    elif encuentra:
        veredicto = f"ENCUENTRA LOS PARES REALES pero {forma}"
    else:
        veredicto = f"NO los pone arriba · {forma}"

    return dict(etiqueta=etiqueta, n_pares=len(pares), n_semillas=len(matrices),
                rho_condicional=float(s_cond.statistic),
                p_condicional=float(s_cond.pvalue),
                rho_marginal=float(s_marg.statistic),
                p_marginal=float(s_marg.pvalue),
                frac_semillas_gana_condicional=gana_cond,
                enriquecimiento=enriq, p_rank_sum_mediana=p_med,
                frac_K_significativos=frac_sig,
                encuentra_pares_reales=encuentra,
                controles_positivos=controles,
                n_controles_recuperados=n_ok, n_controles=len(controles),
                umbral_control=int(umbral_ctrl),
                top_pares_reales=top_reales,
                veredicto_forma=forma, veredicto=veredicto)


def texto(cert: Dict) -> str:
    L = ["=" * 78,
         " CERTIFICACION CONTRA VERDAD CONOCIDA",
         " El framework ordena los pares como la dependencia CONDICIONAL (lo",
         " que la tesis afirma) o como la correlacion MARGINAL (np.corrcoef)?",
         " Y sobre todo: pone ARRIBA los pares que de verdad existen?",
         "=" * 78]
    if not cert:
        L.append("  SIN DATOS.")
        return "\n".join(L)
    L.append(f"  {'fuente':<30}{'rho cond':>10}{'rho marg':>10}"
             f"{'p rank-sum':>12}{'ctrl':>7}")
    for r in cert.values():
        L.append(f"  {r['etiqueta']:<30}{r['rho_condicional']:>+10.3f}"
                 f"{r['rho_marginal']:>+10.3f}{r['p_rank_sum_mediana']:>12.4f}"
                 f"{r['n_controles_recuperados']:>4}/{r['n_controles']}")
    for r in cert.values():
        L.append("")
        L.append(f"  --- {r['etiqueta']}  ({r['n_semillas']} semillas) ---")
        L.append(f"  VEREDICTO: {r['veredicto']}")
        L.append(f"  Spearman contra |pc| condicional {r['rho_condicional']:+.3f} "
                 f"(p={r['p_condicional']:.4f})")
        L.append(f"  Spearman contra |r|  marginal    {r['rho_marginal']:+.3f} "
                 f"(p={r['p_marginal']:.4f})")
        L.append(f"  semillas donde la condicional le gana a la marginal: "
                 f"{r['frac_semillas_gana_condicional'] * 100:.0f}%")
        L.append("")
        L.append(f"  controles positivos (identidades exactas del dominio):")
        for c in r["controles_positivos"]:
            marca = "OK" if c["puesto_framework"] <= r["umbral_control"] else "NO"
            L.append(f"    {c['par']:<14} |pc|={c['pc']:.3f}  verdad "
                     f"{c['puesto_pc']:>2}/{c['n_pares']}  framework "
                     f"{c['puesto_framework']:>2}/{c['n_pares']}  [{marca}]")
        L.append("")
        L.append(f"  {'par real':<20}{'|pc|':>7}{'puesto':>10}")
        for t in r["top_pares_reales"]:
            L.append(f"  {t['par']:<20}{t['pc']:>7.3f}"
                     f"{t['puesto']:>6}/{r['n_pares']}")
        L.append("")
        L.append(f"  enriquecimiento: los K pares mas dependientes, arriba?")
        L.append(f"    {'K':>3}{'puesto medio':>14}{'p rank-sum':>13}"
                 f"{'en top-K':>10}{'azar':>7}{'p hiperg':>10}")
        for e in r["enriquecimiento"]:
            L.append(f"    {e['K']:>3}{e['puesto_mediano']:>14.1f}"
                     f"{e['p_rank_sum']:>13.4f}{e['en_top_K']:>10d}"
                     f"{e['esperado']:>7.2f}{e['p_hipergeom']:>10.4f}")
        L.append(f"    puesto medio de azar = {(r['n_pares'] + 1) / 2:.1f}")
        L.append(f"    p mediana sobre K=3..12: {r['p_rank_sum_mediana']:.4f}   "
                 f"K con p<0.05: {r['frac_K_significativos'] * 100:.0f}%")
    L += ["",
          "  COMO LEERLO:",
          "   p rank-sum < 0.05  -> el metodo PONE ARRIBA los pares que de",
          "          verdad existen. Es el test que importa: el Spearman global",
          "          lo diluyen los ~60 pares sin relacion real.",
          "   rho cond > rho marg con p<0.05 -> ademas el ranking entero sigue",
          "          la dependencia condicional.",
          "   rho marg > rho cond -> atajo colineal: repite np.corrcoef.",
          "   ctrl -> identidades exactas (KNDVI=tanh(NDVI^2), MARI=ARI*B7)",
          "          recuperadas en el tercio superior. Si no encuentra las que",
          "          son verdad por construccion, lo demas no es citable.",
          "=" * 78]
    return "\n".join(L)


# ---------------------------------------------------------------------------

def _buscar(candidatos: List[str], es_dir: bool) -> Optional[str]:
    for c in candidatos:
        if (os.path.isdir(c) if es_dir else os.path.exists(c)):
            return c
    return None


def _cargar(dirm: str, prefijo: str, n: int) -> Tuple[List[np.ndarray], List[int]]:
    mats, seeds = [], []
    for f in sorted(glob.glob(os.path.join(dirm, f"{prefijo}*.npy"))):
        s = re.search(r"_(\d+)\.npy$", f)
        try:
            M = np.load(f)
        except Exception as e:
            print(f"  no se pudo leer {os.path.basename(f)}: {e}")
            continue
        if M.shape != (n, n):
            print(f"  {os.path.basename(f)} tiene forma {M.shape}, se esperaba "
                  f"({n}, {n}) — de una corrida con otro numero de indices. Se ignora.")
            continue
        mats.append(M)
        seeds.append(int(s.group(1)) if s else len(seeds))
    return mats, seeds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", default=None)
    ap.add_argument("--matrices", default=None)
    ap.add_argument("--salida", default="resultados")
    args = ap.parse_args()

    stack_p = args.stack or _buscar(CANDIDATOS_STACK, False)
    if stack_p is None:
        sys.exit("No encuentro indices_12.npy. Pasalo con --stack.")
    dirm = args.matrices or _buscar(CANDIDATOS_MATRICES, True)
    if dirm is None:
        sys.exit("No encuentro la carpeta de matrices. Pasala con --matrices.")

    print(f"\nStack    : {stack_p}")
    print(f"Matrices : {dirm}")
    stack = np.load(stack_p)
    n = int(stack.shape[-1])
    if n != len(INDEX_NAMES):
        sys.exit(f"El stack tiene {n} indices y se esperaban {len(INDEX_NAMES)}.")
    PC, RR = pc_y_r(stack)

    cert: Dict[str, Dict] = {}
    att, s_att = _cargar(dirm, "attention_seed_", n)
    print(f"  attention_seed_*.npy   : {len(att)}")
    dep, _ = _cargar(dirm, "dependencia_seed_", n)
    print(f"  dependencia_seed_*.npy : {len(dep)}")
    if len(att) >= 2:
        cert["atencion"] = certificar(att, "atencion (rollout)", PC, RR)
    if len(dep) >= 2:
        cert["ablacion"] = certificar(dep, "dependencia por ablacion", PC, RR)
    else:
        print("  Sin dependencia_seed_*.npy solo se certifica la atencion.")
        print("  Esas matrices las genera el ENTRENAMIENTO con")
        print("  CALCULAR_DEPENDENCIA_ABLACION = True, no se pueden reconstruir")
        print("  desde los ficheros ya guardados.")
    if not cert:
        sys.exit("No hay matrices suficientes (hacen falta al menos 2).")

    txt = texto(cert)
    print("\n" + txt)
    os.makedirs(args.salida, exist_ok=True)
    with open(os.path.join(args.salida, "CERTIFICACION.json"), "w",
              encoding="utf-8") as f:
        json.dump(cert, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.salida, "CERTIFICACION.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt)
    print(f"\n  Guardado: {args.salida}/CERTIFICACION.json")
    print(f"  Guardado: {args.salida}/CERTIFICACION.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
