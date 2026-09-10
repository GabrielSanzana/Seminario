#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analisis espectral, de Markov y de comunidades. INDEPENDIENTE del pipeline.

POR QUE ESTE MODULO
-------------------
Los dos modulos anteriores tratan la matriz A como una lista de 66 numeros:
certificar_todo.py la ordena y la compara contra una verdad externa,
analisis_grafo.py mide reproducibilidad y comunidades. Ninguno usa la
propiedad mas fuerte que tiene A y que el pipeline le impone explicitamente:

    rollout = rollout / rollout.sum(axis=1, keepdims=True)

Las filas suman 1. A NO es una matriz de pesos cualquiera: es una MATRIZ
ESTOCASTICA, o sea el operador de transicion de una cadena de Markov sobre los
12 indices. Eso da acceso a un instrumental que hasta ahora no se uso y que
ataca de raiz el problema que obligo a inventar parches:

  CORRIENTE NETA        El efecto sumidero -que el 85% de la asimetria venga de
                        que un indice recibe mucho peso- se corrige de forma
                        principled con la corriente de probabilidad neta
                        J_ij = pi_i A_ij - pi_j A_ji. Bajo balance detallado J
                        es cero: cualquier J distinto de cero es flujo
                        direccional REAL, ya descontada la estacionaria. Es lo
                        que _parte_de_par aproximaba a mano.

  TIEMPO DE PRIMERA PASADA  Cuantos pasos tarda la cadena en llegar de i a j.
                        Es una distancia dirigida que integra todos los caminos,
                        no solo la arista directa, asi que ve relaciones
                        indirectas que la lectura arista a arista pierde.

  SUBESPACIOS PRINCIPALES  Comparar entradas de A entre semillas es fragil.
                        Comparar el SUBESPACIO que generan los primeros
                        vectores singulares no lo es: dos matrices pueden
                        diferir entrada a entrada y describir la misma
                        estructura de bajo rango. Los angulos principales
                        miden justo eso.

  ESPECTRO DEL LAPLACIANO  Conectividad algebraica, numero de comunidades
                        naturales por el gap, y conductancia de cada corte.

TODO se contrasta contra la misma nulidad del resto del proyecto: permutar las
12 etiquetas de indice.

USO
    .venv/bin/python analisis_espectral.py
    .venv/bin/python analisis_espectral.py --permutaciones 5000
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
    CANDIDATOS_MATRICES, CANDIDATOS_PARADIGMAS, CANDIDATOS_STACK,
    INDEX_FAMILIES, INDEX_NAMES, N_IDX, PARES, _buscar, familia_de_indice,
    pc_y_r, recolectar_fuentes,
)
from analisis_grafo import (  # noqa: E402
    _indice_rand_ajustado, _simetrica_positiva, comunidades, modularidad,
)


# ---------------------------------------------------------------------------
# Cadena de Markov
# ---------------------------------------------------------------------------

def estocastica(A: np.ndarray, quitar_diagonal: bool = True) -> np.ndarray:
    """Normaliza A a matriz estocastica por filas.

    El rollout ya llega normalizado, pero las capas sueltas y la ablacion no,
    y para compararlas todas bajo el mismo formalismo hace falta el mismo
    objeto. Se quita la diagonal porque un lazo i->i no aporta informacion
    relacional y solo alarga los tiempos de pasada.
    """
    M = np.abs(np.asarray(A, dtype=np.float64)).copy()
    if quitar_diagonal:
        np.fill_diagonal(M, 0.0)
    s = M.sum(axis=1, keepdims=True)
    return np.where(s > 0, M / np.maximum(s, 1e-300), 1.0 / N_IDX)


def estacionaria(P: np.ndarray) -> np.ndarray:
    """Distribucion estacionaria pi de la cadena (autovector izquierdo de 1)."""
    w, V = np.linalg.eig(P.T)
    k = int(np.argmin(np.abs(w - 1.0)))
    v = np.real(V[:, k])
    v = np.abs(v)
    return v / v.sum() if v.sum() > 0 else np.full(N_IDX, 1.0 / N_IDX)


def corriente_neta(P: np.ndarray) -> np.ndarray:
    """J_ij = pi_i P_ij - pi_j P_ji: flujo direccional descontada la estacionaria.

    ESTA ES LA MEDIDA DE DIRECCION QUE FALTABA. La asimetria cruda A_ij - A_ji
    esta dominada por cuan sumidero es cada nodo -medido: el 85% de su
    varianza-, porque con filas que suman 1 un indice que recibe mucho peso
    tiene por fuerza A_ij alto para casi todo i.

    La corriente neta no tiene ese problema por construccion. Una cadena
    reversible (balance detallado) cumple pi_i P_ij = pi_j P_ji y da J = 0 en
    todas las aristas, por muy desigual que sea pi. Cualquier J distinto de
    cero es circulacion real: probabilidad que fluye en un sentido y no vuelve
    por el mismo camino. Es la definicion estandar de irreversibilidad en
    cadenas de Markov, no un parche.
    """
    pi = estacionaria(P)
    F = pi[:, None] * P
    return F - F.T


def tiempos_primera_pasada(P: np.ndarray) -> np.ndarray:
    """T[i,j] = pasos esperados para llegar de i a j por primera vez.

    Integra TODOS los caminos de i a j, no solo la arista directa. Dos indices
    sin arista fuerte entre ellos pueden estar cerca si comparten vecinos, y
    esa cercania es invisible en la lectura arista a arista que hace el resto
    del proyecto.
    """
    pi = estacionaria(P)
    T = np.zeros((N_IDX, N_IDX))
    for j in range(N_IDX):
        # Sistema estandar: t_i = 1 + sum_{k != j} P_ik t_k, con t_j = 0.
        idx = [i for i in range(N_IDX) if i != j]
        Q = P[np.ix_(idx, idx)]
        try:
            t = np.linalg.solve(np.eye(len(idx)) - Q, np.ones(len(idx)))
        except np.linalg.LinAlgError:
            t = np.full(len(idx), np.nan)
        for pos, i in enumerate(idx):
            T[i, j] = t[pos]
    return T


def irreversibilidad(P: np.ndarray) -> float:
    """Fraccion del flujo total que es circulacion neta y no ida y vuelta.

    0 = cadena reversible, el grafo no tiene direccion. Cuanto mas alto, mas
    del trafico es un ciclo con sentido.
    """
    pi = estacionaria(P)
    F = pi[:, None] * P
    np.fill_diagonal(F, 0.0)
    total = F.sum()
    return float(np.abs(F - F.T).sum() / (2 * total)) if total > 0 else 0.0


def entropia_tasa(P: np.ndarray) -> float:
    """Entropia por paso de la cadena, en bits. Maximo log2(N-1) = 3.46.

    Baja significa que desde cada indice la cadena sabe a donde ir; alta, que
    reparte casi uniforme y el grafo no compromete estructura.
    """
    pi = estacionaria(P)
    with np.errstate(divide="ignore", invalid="ignore"):
        L = np.where(P > 0, P * np.log2(np.maximum(P, 1e-300)), 0.0)
    return float(-(pi[:, None] * L).sum())


# ---------------------------------------------------------------------------
# Espectral
# ---------------------------------------------------------------------------

def laplaciano_normalizado(A: np.ndarray) -> np.ndarray:
    S = _simetrica_positiva(A)
    d = np.maximum(S.sum(axis=1), 1e-12)
    return np.eye(N_IDX) - S / np.sqrt(np.outer(d, d))


def perfil_espectral(A: np.ndarray) -> Dict:
    """Autovalores del laplaciano y lo que dicen sobre la forma del grafo."""
    w = np.sort(np.real(np.linalg.eigvalsh(laplaciano_normalizado(A))))
    # El primer autovalor es 0 por construccion. El segundo (Fiedler) mide la
    # conectividad algebraica; los saltos grandes en la serie indican cuantas
    # comunidades naturales tiene el grafo.
    saltos = np.diff(w[:7])
    return dict(fiedler=float(w[1]),
                autovalores=[float(x) for x in w],
                k_sugerido=int(np.argmax(saltos) + 1),
                salto_maximo=float(saltos.max()) if saltos.size else 0.0)


def rango_efectivo(A: np.ndarray) -> Dict:
    """Cuantos modos hacen falta para describir la matriz.

    Rango efectivo bajo con 12 nodos significa que la interaccion se resume en
    pocos patrones globales; alto, que cada arista es independiente. Se usa la
    entropia de los valores singulares normalizados, que no depende de un
    umbral arbitrario.
    """
    s = np.linalg.svd(np.abs(A), compute_uv=False)
    p = s / max(s.sum(), 1e-300)
    p = p[p > 0]
    H = -(p * np.log(p)).sum()
    return dict(rango_efectivo=float(np.exp(H)),
                energia_primer_modo=float(s[0] ** 2 / max((s ** 2).sum(), 1e-300)),
                valores_singulares=[float(x) for x in s[:5]])


def angulos_principales(A: np.ndarray, B: np.ndarray, k: int = 3) -> float:
    """Similitud entre los subespacios principales de dos matrices.

    Compara ESTRUCTURA y no entradas: dos matrices pueden diferir mucho valor
    a valor y generar el mismo subespacio de rango k, que es la forma correcta
    de preguntar si describen la misma interaccion. Devuelve el coseno medio
    de los angulos principales: 1 = mismo subespacio, 0 = ortogonales.
    """
    Ua = np.linalg.svd(np.abs(A))[0][:, :k]
    Ub = np.linalg.svd(np.abs(B))[0][:, :k]
    s = np.linalg.svd(Ua.T @ Ub, compute_uv=False)
    return float(np.clip(s, 0, 1).mean())


def conductancia(A: np.ndarray, etiquetas: np.ndarray) -> List[float]:
    """Conductancia de cada comunidad: peso que sale sobre peso total.

    Baja significa comunidad bien separada del resto; cerca de 1, que el corte
    no separa nada y la comunidad es un artefacto del algoritmo.
    """
    S = _simetrica_positiva(A)
    out = []
    for c in sorted(set(etiquetas.tolist())):
        m = etiquetas == c
        if m.sum() == 0 or m.all():
            out.append(float("nan"))
            continue
        corte = S[np.ix_(m, ~m)].sum()
        vol = S[m].sum()
        vol_c = S[~m].sum()
        out.append(float(corte / max(min(vol, vol_c), 1e-12)))
    return out


# ---------------------------------------------------------------------------

def analizar(etiqueta: str, mats: List[np.ndarray], fam: np.ndarray,
             n_perm: int, rng: np.random.Generator) -> Dict:
    A = np.mean(mats, axis=0)
    P = estocastica(A)
    pi = estacionaria(P)

    # --- Markov ---
    J = corriente_neta(P)
    j_vec = np.array([J[i, j] for i, j in PARES])
    orden_j = np.argsort(-np.abs(j_vec))
    irr = irreversibilidad(P)
    irr_sem = np.array([irreversibilidad(estocastica(m)) for m in mats])
    nulo_irr = np.array([irreversibilidad(
        estocastica(A[np.ix_(pi_, pi_)]))
        for pi_ in (rng.permutation(N_IDX) for _ in range(min(n_perm, 500)))])
    p_irr = float((1 + int((nulo_irr >= irr).sum())) / (1 + nulo_irr.size))

    T = tiempos_primera_pasada(P)
    Hrate = entropia_tasa(P)

    # --- Espectral ---
    esp = perfil_espectral(A)
    rgo = rango_efectivo(A)
    parejas = [(i, j) for i in range(len(mats)) for j in range(i + 1, len(mats))]
    if len(parejas) > 200:
        sel = rng.choice(len(parejas), 200, replace=False)
        parejas = [parejas[t] for t in sel]
    ang = float(np.mean([angulos_principales(mats[i], mats[j])
                         for i, j in parejas]))
    ang_nulo = float(np.mean([
        angulos_principales(mats[i], mats[j][np.ix_(p_, p_)])
        for (i, j), p_ in zip(parejas[:60],
                              (rng.permutation(N_IDX) for _ in range(60)))]))

    # --- Comunidades, barrido de k ---
    barrido = []
    for k in range(2, 7):
        parts = [comunidades(m, k=k) for m in mats]
        ari = float(np.mean([_indice_rand_ajustado(parts[i], parts[j])
                             for i, j in parejas[:120]]))
        cons = comunidades(A, k=k)
        cond = conductancia(A, cons)
        barrido.append(dict(
            k=k, ari_entre_semillas=ari,
            ari_vs_familias=float(_indice_rand_ajustado(cons, fam)),
            modularidad=float(modularidad(A, cons)),
            conductancia_media=float(np.nanmean(cond)),
            grupos=[[INDEX_NAMES[i] for i in range(N_IDX) if cons[i] == c]
                    for c in sorted(set(cons.tolist()))]))
    mejor = max(barrido, key=lambda b: b["ari_entre_semillas"])

    return dict(
        etiqueta=etiqueta, n_semillas=len(mats),
        estacionaria={INDEX_NAMES[i]: float(pi[i]) for i in range(N_IDX)},
        irreversibilidad=irr, p_irreversibilidad=p_irr,
        irreversibilidad_nulo=float(nulo_irr.mean()),
        irreversibilidad_semilla_std=float(irr_sem.std()),
        entropia_tasa=Hrate, entropia_maxima=float(np.log2(N_IDX - 1)),
        corrientes_top=[
            dict(de=INDEX_NAMES[PARES[k][1] if j_vec[k] < 0 else PARES[k][0]],
                 a=INDEX_NAMES[PARES[k][0] if j_vec[k] < 0 else PARES[k][1]],
                 J=float(abs(j_vec[k]))) for k in orden_j[:8]],
        fiedler=esp["fiedler"], k_sugerido=esp["k_sugerido"],
        autovalores=esp["autovalores"],
        rango_efectivo=rgo["rango_efectivo"],
        energia_primer_modo=rgo["energia_primer_modo"],
        angulos_principales=ang, angulos_principales_nulo=ang_nulo,
        pasada_media=float(np.nanmean(T[~np.eye(N_IDX, dtype=bool)])),
        barrido_k=barrido, mejor_k=mejor["k"],
        comunidades_mejor_k=mejor["grupos"],
        ari_mejor_k=mejor["ari_entre_semillas"],
        conductancia_mejor_k=mejor["conductancia_media"])


def texto(filas: List[Dict]) -> str:
    W = 78
    L: List[str] = []
    L.append("=" * W)
    L.append(" ANALISIS ESPECTRAL Y DE MARKOV")
    L.append("")
    L.append(" Las filas de la matriz de atencion suman 1: es el operador de")
    L.append(" transicion de una cadena de Markov sobre los 12 indices. Eso")
    L.append(" permite medir direccion sin parches -la corriente neta descuenta")
    L.append(" la estacionaria por construccion- y comparar estructura entre")
    L.append(" semillas por subespacios en vez de por entradas.")
    L.append("=" * W)
    L.append("")
    L.append(f"  {'fuente':<30}{'irrev':>8}{'p':>8}{'H tasa':>8}{'rango':>7}"
             f"{'subesp':>8}{'nulo':>7}{'Fiedler':>9}")
    for r in filas:
        L.append(f"  {r['etiqueta']:<30}{r['irreversibilidad']:>8.3f}"
                 f"{r['p_irreversibilidad']:>8.4f}{r['entropia_tasa']:>8.2f}"
                 f"{r['rango_efectivo']:>7.2f}{r['angulos_principales']:>8.3f}"
                 f"{r['angulos_principales_nulo']:>7.3f}{r['fiedler']:>9.4f}")
    L.append("")
    L.append(f"  irrev = fraccion del flujo que es circulacion neta. 0 = cadena")
    L.append("     reversible, el grafo no tiene direccion propia.")
    L.append(f"  H tasa = entropia por paso en bits (maximo "
             f"{filas[0]['entropia_maxima']:.2f} si reparte uniforme).")
    L.append("  rango = numero efectivo de modos que describen la matriz.")
    L.append("  subesp = coseno medio de los angulos principales entre los")
    L.append("     subespacios de rango 3 de dos semillas. Compara ESTRUCTURA,")
    L.append("     no entradas: es la version robusta de la reproducibilidad.")
    L.append("  Fiedler = conectividad algebraica del laplaciano normalizado.")
    L.append("")
    for r in filas:
        L.append("-" * W)
        L.append(f" {r['etiqueta']}  ({r['n_semillas']} semillas)")
        pi_ord = sorted(r["estacionaria"].items(), key=lambda kv: -kv[1])
        L.append("  estacionaria (donde pasa el tiempo la cadena):")
        L.append("    " + ", ".join(f"{k} {v:.3f}" for k, v in pi_ord[:6]))
        L.append(f"  irreversibilidad {r['irreversibilidad']:.3f} "
                 f"(nulo {r['irreversibilidad_nulo']:.3f}, "
                 f"p={r['p_irreversibilidad']:.4f}, "
                 f"desviacion entre semillas "
                 f"{r['irreversibilidad_semilla_std']:.4f})")
        L.append("  corrientes netas mas fuertes (flujo que no vuelve):")
        for c in r["corrientes_top"][:6]:
            L.append(f"    {c['de']:>12} -> {c['a']:<12} J={c['J']:.5f}")
        L.append(f"  rango efectivo {r['rango_efectivo']:.2f} de {N_IDX}; "
                 f"primer modo {r['energia_primer_modo'] * 100:.1f}% de energia")
        L.append(f"  k sugerido por el salto espectral: {r['k_sugerido']}; "
                 f"k con mejor estabilidad: {r['mejor_k']}")
        L.append(f"  comunidades en k={r['mejor_k']} "
                 f"(ARI entre semillas {r['ari_mejor_k']:+.3f}, "
                 f"conductancia {r['conductancia_mejor_k']:.3f}):")
        for g in r["comunidades_mejor_k"]:
            L.append("    {" + ", ".join(g) + "}")
        L.append(f"  {'k':>3}{'ARI semillas':>14}{'ARI familias':>14}"
                 f"{'modularidad':>13}{'conductancia':>14}")
        for b in r["barrido_k"]:
            L.append(f"  {b['k']:>3}{b['ari_entre_semillas']:>14.3f}"
                     f"{b['ari_vs_familias']:>14.3f}"
                     f"{b['modularidad']:>13.3f}"
                     f"{b['conductancia_media']:>14.3f}")
    L.append("=" * W)
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="Analisis espectral y de Markov.")
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
    PC, RR = pc_y_r(np.load(stack_p))
    dirm = a.matrices or _buscar(CANDIDATOS_MATRICES, True)
    dirp = a.paradigmas or _buscar(CANDIDATOS_PARADIGMAS, True)
    fam = familia_de_indice()

    fuentes = [(e, p) for e, _n, _A, _s, p in
               recolectar_fuentes(dirm, dirp, PC, RR) if p and len(p) >= 5]
    print(f"  fuentes con matrices por semilla: {len(fuentes)}")
    rng = np.random.default_rng(a.semilla)
    filas = []
    for etiqueta, mats in fuentes:
        print(f"    analizando: {etiqueta}")
        filas.append(analizar(etiqueta, mats, fam, a.permutaciones, rng))

    txt = texto(filas)
    print("\n" + txt)
    os.makedirs(a.salida, exist_ok=True)
    with open(os.path.join(a.salida, "ANALISIS_ESPECTRAL.json"), "w",
              encoding="utf-8") as f:
        json.dump(filas, f, indent=2, ensure_ascii=False)
    with open(os.path.join(a.salida, "ANALISIS_ESPECTRAL.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt)
    print(f"\n  Guardado: {os.path.join(a.salida, 'ANALISIS_ESPECTRAL.txt')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
