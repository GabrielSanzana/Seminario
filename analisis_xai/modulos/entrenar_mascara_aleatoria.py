#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reentrenar con mascara de tamano aleatorio, para que la atencion tenga que
condicionarse a lo que hay visible.

DE DONDE SALE LA HIPOTESIS
--------------------------
Con MASCARA_MODO = "familia_balanceada" el patron de ocultacion es casi
determinista dado el indice que se reconstruye: la familia espectral de v
siempre va oculta, y el relleno Bernoulli solo agita el resto. El modelo puede
entonces aprender una POLITICA FIJA por objetivo -"para reconstruir v, mirar
este conjunto"- y no necesita nunca condicionarse a que indices quedaron
visibles en esta muestra concreta. Una politica fija por objetivo es
exactamente una fila de atencion que depende solo de v, o sea efecto de nodo.

La medicion de atencion_cruda.py lo respalda desde el otro lado: el MISMO
modelo, leido bajo dos condiciones de mascara distintas, pasa de 9% a 54% de
aridad 2. Cuando la redundancia esta presente la estructura de par desaparece
de la matriz; cuando se la quita, aparece. Eso es en la LECTURA. La hipotesis
de este modulo es que lo mismo vale en el ENTRENAMIENTO.

QUE CAMBIA
----------
Mascara de tamano aleatorio: para cada (muestra, variante) se sortea
k ~ Uniforme{1..10} y se ocultan el objetivo mas k-1 indices al azar. El tope
en 10 deja siempre al menos 2 visibles, porque con uno solo el gradiente de esa
variante es casi ruido.

Con esto el conjunto visible cambia de muestra en muestra, asi que la atencion
optima ya no puede ser una funcion solo de v: tiene que mirar quien esta. Esa
dependencia es relacional por construccion.

El sorteo uniforme ya deja equilibrada la disponibilidad marginal de los doce
indices, que es lo que "familia_balanceada" conseguia a mano con su q_t, asi
que no hace falta el correctivo.

QUE SE COMPARA, Y QUE NO ES COMPARABLE
--------------------------------------
La perdida de validacion de los dos brazos NO es comparable: son tareas
distintas, una oculta la familia y la otra un numero variable de indices al
azar. Se reporta igual, pero como descripcion y no como comparacion de
calidad.

Lo que si es comparable es todo lo que se mide DESPUES sobre las matrices, si
se lee con el mismo protocolo. Para eso los dos brazos usan las MISMAS semillas
-no las mejores de cada uno por perdida, que seria una seleccion distinta en
cada brazo- y la lectura posterior se hace con atencion_cruda.py y
ablacion_atencion.py apuntando con --modelos a cada carpeta.

RESULTADO: LA HIPOTESIS SALE REFUTADA
-------------------------------------
15 semillas por brazo, las mismas en los dos, leidas con el mismo protocolo:

                    fila    col    PAR     ICC   rho con Datt
    base/identidad   28%    68%     4%   0.745      +0.064
    alea/identidad    2%    93%     5%   0.452      +0.015
    base/familia     15%    29%    56%   0.856      +0.308
    alea/familia      4%    22%    74%   0.714      +0.290

Bajo la condicion de lectura que usa el pipeline -solo el objetivo oculto- la
aridad 2 pasa de 4% a 5%: nada. Y el efecto sumidero EMPEORA, de 68% a 93% de
la varianza en la columna.

Bajo la condicion de familia la cuota de par sube de 56% a 74%, pero eso no es
senal nueva sino denominador: el efecto de fila se derrumba de 15% a 4%, que es
lo esperable cuando la mascara uniforme hace que todos los indices se oculten
con la misma frecuencia y desaparece el "algunas filas necesitan mirar mas".
Las dos medidas que no dependen del reparto empeoran las dos: el ICC entre
semillas baja de 0.856 a 0.714 y la correlacion con el efecto causal medido
baja de +0.308 a +0.290.

O sea que aleatorizar el tamano de mascara al entrenar hizo la atencion MAS
generica, no menos. La razon plausible es que una politica robusta frente a
mascaras de cualquier tamano es precisamente una politica que no depende de la
mascara, y esa es una fila de atencion plana.

Lo que no cambia es la lectura interventiva: cortar aristas da 82% de aridad 2
en el brazo base y 84% en el aleatorio, con rho +0.851 y +0.858 contra la
dependencia por ablacion de entrada. El sumidero sobrevive a todo lo que se le
haga a la lectura y a la mascara de entrenamiento; la intervencion lo esquiva
en los dos brazos.

Las perdidas de validacion (base 0.2159, aleatorio 0.2026) NO son comparables:
cada brazo se valida contra su propia tarea. Se dejan por descartar que el
brazo aleatorio estuviera peor entrenado.

USO
    .venv/bin/python entrenar_mascara_aleatoria.py --semillas 15
    .venv/bin/python entrenar_mascara_aleatoria.py --indices 0,3,6   # en paralelo
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, "/srv/pfigueroa")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hello as H  # noqa: E402

_ORIG_MASCARA = H.mascara_multiple
ENTRENANDO = False
K_MAX = 10


def mascara_aleatoria(B: int, N: int, device,
                      generator: Optional[torch.Generator] = None
                      ) -> torch.Tensor:
    """(B, N, N) booleano con k ~ Uniforme{1..K_MAX} por (muestra, variante).

    La diagonal siempre oculta -la variante v reconstruye v- y k-1 distractores
    al azar. El muestreo sin reemplazo se hace con el truco de siempre: puntuar
    al azar, poner la diagonal fuera de concurso y quedarse con los k-1
    mayores, solo que aqui el corte k cambia por fila.
    """
    ident = torch.eye(N, dtype=torch.bool, device=device)
    k = torch.randint(1, K_MAX + 1, (B, N, 1), device=device,
                      generator=generator)
    pts = torch.rand(B, N, N, device=device, generator=generator)
    pts = pts.masked_fill(ident.unsqueeze(0), -1.0)
    # rango 0..N-1 de cada token dentro de su fila, de mayor a menor puntaje
    rango = pts.argsort(dim=-1, descending=True).argsort(dim=-1)
    return (rango < (k - 1)) | ident.unsqueeze(0)


def _mascara_parcheada(B, N, k, device, generator=None):
    if ENTRENANDO and N == H.NUM_INDICES:
        return mascara_aleatoria(B, N, device, generator)
    return _ORIG_MASCARA(B, N, k, device, generator)


def main() -> int:
    global ENTRENANDO
    ap = argparse.ArgumentParser()
    ap.add_argument("--semillas", type=int, default=15)
    ap.add_argument("--indices", default=None,
                    help="indices concretos de la lista de semillas, separados "
                         "por coma, para repartir el trabajo entre procesos")
    ap.add_argument("--epocas", type=int, default=H.TOTAL_EPOCHS)
    ap.add_argument("--destino", default="resultados/mascara_aleatoria")
    a = ap.parse_args()

    H.mascara_multiple = _mascara_parcheada
    os.makedirs(a.destino, exist_ok=True)
    stack = np.load(H.OUTPUT_NPY)
    fechas = H.cargar_fechas_stack(len(stack))

    semillas = list(H.SEEDS)[:a.semillas]
    if a.indices:
        idx = [int(x) for x in a.indices.split(",") if x.strip() != ""]
        semillas = [semillas[i] for i in idx if i < len(semillas)]
    print("  semillas de este proceso: " + str(semillas))
    print("  mascara: k ~ Uniforme{1.." + str(K_MAX) + "} por muestra y "
          "variante")

    for s in semillas:
        ck = os.path.join(a.destino, "model_seed_" + str(s) + ".pth")
        if os.path.exists(ck):
            print("  semilla " + str(s) + " ya estaba, se salta")
            continue
        t0 = time.time()
        H.set_seed(s)
        ENTRENANDO = True
        try:
            _m, _val, _h, best = H.train_convtransformer(
                stack, seq_length=H.SEQ_LENGTH, total_epochs=a.epocas,
                batch_size=H.BATCH_SIZE, lr=H.LR_CONVTRANSFORMER,
                patience=H.PATIENCE, num_workers=0, ckpt_path=ck,
                dates_millis=fechas)
        finally:
            ENTRENANDO = False
        # Un archivo por semilla: varios procesos escriben a la vez y un
        # val_losses.json compartido se corromperia.
        with open(os.path.join(a.destino, "val_" + str(s) + ".json"), "w") as f:
            json.dump({str(s): float(best)}, f)
        print("  semilla " + str(s) + " lista en %.1f min  val=%.6f"
              % ((time.time() - t0) / 60, best))

    # Consolidar lo que haya, sin pisar lo de otros procesos.
    vl = {}
    for nom in os.listdir(a.destino):
        if nom.startswith("val_") and nom.endswith(".json"):
            with open(os.path.join(a.destino, nom)) as f:
                vl.update(json.load(f))
    with open(os.path.join(a.destino, "val_losses.json"), "w") as f:
        json.dump(vl, f, indent=2)
    print("  val_losses.json con " + str(len(vl)) + " semillas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
