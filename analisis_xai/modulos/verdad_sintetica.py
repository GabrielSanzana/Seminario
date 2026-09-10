#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Datos sinteticos con la estructura causal conocida por construccion.

QUE DECIDE ESTE EXPERIMENTO
---------------------------
Todo el trabajo sobre el vinedo establecio equivalencia epistemica entre Datt y
un modelo grafico clasico, y dejo pendiente lo unico que importa de verdad:
cual de los dos recupera la estructura VERDADERA. Sobre datos reales no se
puede saber, porque no hay estructura verdadera contra la cual comparar; el
intento con verdad externa dio auc de 0.29 a 0.56, o sea nada.

Aqui la estructura se fija por construccion y despues se pregunta quien la
recupera. Es el unico diseno en el que la pregunta tiene respuesta.

DOS REGIMENES, Y EL PRIMERO ES UNA TRAMPA PARA MI MISMO
-------------------------------------------------------
  lineal      mecanismos lineales, ruido gaussiano. Es el terreno donde los
              supuestos del metodo clasico se cumplen EXACTAMENTE. Si el
              framework gana aqui, el experimento esta mal montado y hay que
              revisarlo antes que celebrarlo.

  no lineal   interacciones entre padres (X = a*p1*p2) y no linealidades
              simetricas (X = a*p^2). La segunda es la que mata a los metodos
              de correlacion: si p es simetrico alrededor de cero, la
              correlacion entre p y p^2 es CERO aunque la dependencia sea
              total. No es un caso rebuscado, es la forma de media saturacion
              biofisica.

En los dos casos los campos son espaciales, no pixeles independientes: el ruido
exogeno y los generadores son campos aleatorios suavizados por filtrado en
frecuencia. Sin eso la parte convolucional del modelo no tendria nada que
hacer y la comparacion estaria amanada a favor del metodo clasico.

QUE SE PUEDE RECUPERAR Y QUE NO
-------------------------------
Importa decirlo antes de medir. La tarea del modelo es reconstruir el canal i
tapado a partir de los otros once. El predictor optimo de X_i dado todo lo
demas usa su MANTO DE MARKOV: padres, hijos, y padres de sus hijos. No sus
padres solamente.

Asi que ningun metodo basado en dependencia condicional -ni Datt, ni la
correlacion parcial, ni LOCO- puede recuperar el DAG. Todos recuperan el grafo
moralizado. Medir contra el DAG y declarar fracaso seria medir mal. Se punta
contra las tres cosas por separado:

    esqueleto del DAG      las aristas verdaderas, sin direccion
    grafo moralizado       lo que la dependencia condicional puede ver
    orientacion            de las aristas verdaderas, cuantas apuntan bien

La tercera es la que interesa. Los metodos clasicos solo orientan hasta la
clase de equivalencia de Markov; Datt afirma sacar direccion de una asimetria
medida. Con la verdad conocida eso se contrasta contra un azar de 0.5, y es la
primera vez que se puede.

USO
    .venv/bin/python verdad_sintetica.py --salida datos_sinteticos
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

N_IDX = 12
LADO = 52
N_FECHAS = 186


def campo_suave(rng, n, lado, escala):
    """Campos aleatorios gaussianos con correlacion espacial.

    Se filtra ruido blanco en frecuencia con un nucleo gaussiano: el resultado
    tiene la misma distribucion marginal pero los pixeles vecinos se parecen,
    que es lo que hace que un modelo con contexto espacial tenga algo que
    aprovechar. Con pixeles independientes la parte convolucional sobra y la
    comparacion favorece al metodo que trata cada pixel por separado.
    """
    b = rng.normal(size=(n, lado, lado))
    fx = np.fft.fftfreq(lado)[:, None]
    fy = np.fft.fftfreq(lado)[None, :]
    nucleo = np.exp(-0.5 * (fx ** 2 + fy ** 2) * (escala ** 2))
    out = np.real(np.fft.ifft2(np.fft.fft2(b, axes=(1, 2)) * nucleo,
                               axes=(1, 2)))
    return (out - out.mean(axis=(1, 2), keepdims=True)) / (
        out.std(axis=(1, 2), keepdims=True) + 1e-9)


def dag_aleatorio(rng, n=N_IDX, n_aristas=19):
    """DAG disperso con orden topologico 0..n-1. Densidad como la del grafo
    real de 19 aristas sobre 12 nodos, para que la comparacion sea al mismo
    nivel de dificultad."""
    posibles = [(i, j) for i in range(n) for j in range(i)]   # j padre de i
    sel = rng.choice(len(posibles), size=min(n_aristas, len(posibles)),
                     replace=False)
    A = np.zeros((n, n), dtype=int)
    for k in sel:
        i, j = posibles[k]
        A[i, j] = 1          # A[i, j] = 1 significa j es padre de i
    return A


def moralizar(A):
    """Grafo moralizado: esqueleto mas las aristas entre padres de un hijo.

    Es el limite de lo identificable por dependencia condicional, y por tanto
    el blanco justo para Datt, para la correlacion parcial y para LOCO.
    """
    n = len(A)
    M = ((A + A.T) > 0).astype(int)
    for i in range(n):
        padres = np.where(A[i])[0]
        for a in range(len(padres)):
            for b in range(a + 1, len(padres)):
                M[padres[a], padres[b]] = 1
                M[padres[b], padres[a]] = 1
    np.fill_diagonal(M, 0)
    return M


def suavizar(campo, sigma=1.6):
    """Media local por filtrado gaussiano en frecuencia."""
    lado = campo.shape[-1]
    fx = np.fft.fftfreq(lado)[:, None]
    fy = np.fft.fftfreq(lado)[None, :]
    k = np.exp(-2 * (np.pi ** 2) * (fx ** 2 + fy ** 2) * (sigma ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(campo, axes=(1, 2)) * k,
                                axes=(1, 2)))


def laplaciano(campo):
    """Laplaciano discreto: responde a bordes, no al nivel.

    Es el operador que hace imposible el analisis por pixel. El valor del
    padre en el pixel no dice casi nada del hijo; lo que lo dice es el
    contraste con sus vecinos. Un metodo que trata cada pixel como una fila
    independiente no tiene forma de verlo, por muchos datos que reciba.
    """
    c = campo
    return (-4 * c + np.roll(c, 1, 1) + np.roll(c, -1, 1)
            + np.roll(c, 1, 2) + np.roll(c, -1, 2))


def generar(rng, A, regimen, n_fechas=N_FECHAS, lado=LADO):
    """Genera el stack (T, lado, lado, 12) siguiendo el DAG.

    Cada canal se calcula en orden topologico a partir de sus padres, con
    ruido exogeno espacialmente correlacionado. Se anade una modulacion lenta
    entre fechas para que el eje temporal tenga variacion, como en el dato
    real, sin que introduzca dependencia entre canales.
    """
    n = len(A)
    X = np.zeros((n_fechas, lado, lado, n), dtype=np.float32)
    coef = {}
    estacion = np.sin(np.linspace(0, 6 * np.pi, n_fechas))[:, None, None]
    for i in range(n):
        padres = list(np.where(A[i])[0])
        ruido = campo_suave(rng, n_fechas, lado, rng.uniform(3.0, 9.0))
        z = np.zeros((n_fechas, lado, lado), dtype=np.float64)
        cs = []
        if regimen == "lineal":
            for p in padres:
                c = rng.uniform(0.5, 1.5) * rng.choice([-1.0, 1.0])
                z += c * X[..., p]
                cs.append(float(c))
        elif regimen == "espacial":
            # El mecanismo pasa por el VECINDARIO del padre, no por su valor
            # en el pixel. Es el unico regimen donde el contexto espacial
            # aporta algo, y por tanto el unico donde la arquitectura
            # convolucional puede justificar su coste. En los otros dos la
            # estructura es por pixel y el transformer compite con una mano
            # atada: se le pide que gane sin usar aquello para lo que sirve.
            for k, p in enumerate(padres):
                c = rng.uniform(0.6, 1.4) * rng.choice([-1.0, 1.0])
                cs.append(float(c))
                if k % 2 == 0:
                    z += c * laplaciano(X[..., p])
                else:
                    z += c * np.tanh(1.5 * suavizar(X[..., p]))
        else:
            for k, p in enumerate(padres):
                c = rng.uniform(0.6, 1.4) * rng.choice([-1.0, 1.0])
                cs.append(float(c))
                if k + 1 < len(padres) and rng.random() < 0.5:
                    # interaccion: ningun metodo de correlacion parcial la ve
                    z += c * X[..., p] * X[..., padres[k + 1]]
                elif rng.random() < 0.5:
                    # no linealidad simetrica: correlacion cero, dependencia
                    # total. Es la forma de una saturacion biofisica.
                    z += c * (X[..., p] ** 2 - 1.0)
                else:
                    z += c * np.tanh(1.5 * X[..., p])
        if padres:
            z = z / (z.std() + 1e-9)
        X[..., i] = (0.75 * z + 0.65 * ruido
                     + 0.25 * estacion * rng.uniform(0.5, 1.5)).astype(
                         np.float32)
        coef[str(i)] = dict(padres=[int(p) for p in padres], coef=cs)
    return X, coef


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", default="datos_sinteticos")
    ap.add_argument("--semilla", type=int, default=7)
    ap.add_argument("--aristas", type=int, default=19)
    ap.add_argument("--fechas", type=int, default=N_FECHAS)
    a = ap.parse_args()

    os.makedirs(a.salida, exist_ok=True)
    rng = np.random.default_rng(a.semilla)
    A = dag_aleatorio(rng, N_IDX, a.aristas)
    M = moralizar(A)
    print("  DAG con %d aristas, grafo moralizado con %d"
          % (int(A.sum()), int(M.sum() // 2)))

    # Semillas fijas por regimen. hash() de una cadena esta aleatorizado por
    # proceso en Python 3, asi que derivar la semilla de el hacia que dos
    # corridas del generador dieran datos distintos con la misma bandera
    # --semilla. Reproducible no es lo mismo que determinista dentro de una
    # corrida, y aqui hacia falta lo primero.
    SEMILLAS = {"lineal": 101, "no_lineal": 202, "espacial": 303}
    for regimen in ("lineal", "no_lineal", "espacial"):
        r2 = np.random.default_rng(a.semilla + SEMILLAS[regimen])
        X, coef = generar(r2, A, regimen, a.fechas)
        ruta = os.path.join(a.salida, "stack_" + regimen + ".npy")
        np.save(ruta, X)
        print("  %s  %s  media %.3f  sd %.3f"
              % (regimen, X.shape, X.mean(), X.std()))
        with open(os.path.join(a.salida, "mecanismos_" + regimen + ".json"),
                  "w", encoding="utf-8") as f:
            json.dump(coef, f, indent=2)

    with open(os.path.join(a.salida, "verdad.json"), "w",
              encoding="utf-8") as f:
        json.dump(dict(dag=A.tolist(), moral=M.tolist(),
                       n_aristas=int(A.sum()),
                       nota="A[i][j] = 1 significa que j es padre de i"),
                  f, indent=2)
    print("  guardado en " + a.salida)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
