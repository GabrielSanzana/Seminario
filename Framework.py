"""
Descargador de 12 índices espectrales desde Sentinel-2 (Google Earth Engine).

Genera un array (N_dates, H, W, 12_indices) alineado espaciotemporalmente.
Cada índice se calcula a partir de las bandas de Sentinel-2 L2A.

Los 12 índices son (bandas según la tabla de la presentación):
  1. NDVI         = (B8 - B4) / (B8 + B4)
  2. MARI         = (1/B3 - 1/B5) * B7               [Modified Anthocyanin Reflectance]
  3. ARI          = 1/B3 - 1/B5                      [Anthocyanin Reflectance Index]
  4. EVI          = 2.5*(B8-B4)/(B8+6*B4-7.5*B2+1)   [Enhanced Vegetation Index]
  5. EVI2         = 2.5*(B8-B4)/(B8+2.4*B4+1)        [EVI 2]
  6. NDWI         = (B3 - B8) / (B3 + B8)            [Normalized Difference Water Index]
  7. NDMI         = (B8A - B11) / (B8A + B11)        [Normalized Difference Moisture Index]
  8. CHL_REDEDGE  = B7/B5 - 1                        [Chlorophyll Index Red-Edge]
  9. NDII         = (B8 - B11) / (B8 + B11)          [Normalized Difference Infrared Index]
 10. SAVI         = 1.5*(B8-B4)/(B8+B4+0.5)          [Soil-Adjusted Vegetation Index]
 11. PSRI         = (B4 - B2) / B6                   [Plant Senescence Reflectance Index]
 12. KNDVI        = tanh(NDVI²)                      [Kernelized NDVI]

NDMI (B8A, 20 m) y NDII (B8, 10 m) miden el mismo contraste NIR-SWIR con dos
bandas NIR distintas: correlacionan alto (~0.97) pero ya no son el mismo canal.

RESOLUCIÓN: B5, B6, B7, B8A y B11 son nativas de 20 m; B2, B3, B4 y B8 de 10 m.
Con RESOLUCION=10 las de 20 m se remuestrean hacia arriba, así que ARI, MARI,
CHL_REDEDGE, NDMI, NDII y PSRI tienen la mitad del detalle espacial real aunque
se almacenen como 10 m. Documentarlo o bajar RESOLUCION a 20.

Reflectancia: las bandas se dividen por S2_SCALE (=10000). Los índices con
constantes aditivas (EVI, SAVI) y los de inversos (ARI, MARI, CHL_REDEDGE)
sólo son válidos en reflectancia real, no en DN.

Las dimensiones de pixel se calculan automáticamente según el AOI y la resolución.

Salida: .npy con shape (N_dates, H, W, 12), dtype float32. El rango NO es
[-1,1] para todos: cada índice se recorta a su rango físico (ver INDEX_CLAMP).
"""
from __future__ import annotations

import os, json, glob
import time
import random
import gc
import copy
import itertools
import numpy as np
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple, Dict, Set, Any
from dataclasses import dataclass, field, asdict
import networkx as nx

# ---------------------------------------------------------------------------
# CORRECCIÓN: las dependencias de Google Earth Engine (ee, geemap, cv2) solo
# hacen falta en la PARTE 1 (descarga). Antes se importaban y se llamaba a
# ee.Authenticate() al importar el módulo, lo que bloqueaba la ejecución de las
# PARTES 2 y 3 en máquinas sin GEE (o sin credenciales cacheadas). Ahora la
# carga es perezosa: si falla, HAS_EE queda en False y solo PARTE 1 se ve
# afectada.
# ---------------------------------------------------------------------------
try:
    import ee
    import geemap
    import cv2
    HAS_EE = True
except Exception:                                     # pragma: no cover
    ee = None; geemap = None; cv2 = None
    HAS_EE = False
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from sklearn.preprocessing import StandardScaler
import matplotlib
from scipy import stats, special
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================================
# SECCIÓN 0 · CONFIGURACIÓN GENERAL
# ============================================================================
# Sitio de prueba tomado del catastro de viveros MINAGRI/SAG
# (viveros_2024_h19.shp, EPSG:32719 = UTM 19S, huso 19):
#   nom_vivero 'Lourdes' - comuna Pencahue, Valle del Maule
#   sp_1 'Vid Vinifera' - sup_ha 41.05
#   x_coor 243117  y_coor 6073342  ->  lon -71.830319  lat -35.450253
# Es el registro de mayor superficie cuya especie es vid vinifera (de 72
# registros con vid/uva en sp_1..sp_3, sobre 3185 viveros del catastro).
# El anterior era [-70.683716, -32.552261] (Aconcagua, Valparaiso).
COORDENADAS: List[float] = [-71.830319, -35.450253]
FECHA_INICIO: str = "2018-01-01"
FECHA_FIN: str = "2025-12-31"
RESOLUCION: int = 10
TAMANO_AREA: int = 500
PROJECT_ID: str = "ee-patricio21"

# ----------------------------------------------------------------------------
# Estructura de carpetas (todo vive junto al script, independiente del CWD):
#   <proyecto>/
#   ├── salidas/multi_indices/            → PARTE 1: indices_12.npy + metadata.npz
#   ├── entrenamiento/                    → PARTE 2: checkpoint del ConvTransformer
#   └── resultados/
#       ├── sensitivity_K_experiment/     → PARTE 3: resultados y checkpoints
#       │   └── matrices/                 → attention_seed_*.npy (cache compartida)
#       └── paradigms_experiment/
#           ├── 01_probabilistic/         → PARTE 4  (paradigma 1)
#           ├── 02_interventional/        → PARTE 5  (paradigma 2)
#           ├── 03_geometric_topological/ → PARTE 6  (paradigma 3)
#           ├── 04_triangulation/         → PARTE 7  (triangulación)
#           ├── 05_theorem_h1h3/          → PARTE 8  (H1,H2,H3)
#           ├── 06_information_theory/    → PARTE 9  (paradigma 4)
#           ├── 07_baselines/             → PARTE 10 (paradigma 5)
#           ├── 13_ciren_species_v4_real/ → PARTE 12 (método 2)
#           └── significant_edges.json    → PARTE 11 escribe / PARTE 12 lee
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# RAIZ DEL PROYECTO — local (Windows/Linux) o Google Colab sobre Drive
#
# En Colab no existe __file__ si el codigo se pega en una celda, y el disco de
# /content se borra al cerrar la sesion. Como la corrida completa deja ~2 GB
# entre el stack de indices, 50 checkpoints y 50 matrices de atencion, todo eso
# tiene que vivir en Drive o se pierde y hay que reentrenar.
#
# Orden de resolucion de la raiz:
#   1. la variable de entorno CIREN_PROJECT_DIR, si esta puesta (gana siempre)
#   2. en Colab: <Drive>/MyDrive/<CARPETA_DRIVE>, montando Drive y creando la
#      carpeta si no existe
#   3. local: la carpeta del propio .py, y si no hay __file__, el cwd
# ----------------------------------------------------------------------------
import sys

CARPETA_DRIVE: str = "ConvTransformer_CIREN"   # nombre dentro de "Mi unidad"
PUNTO_MONTAJE: str = "/content/drive"


def en_colab() -> bool:
    """True si se ejecuta dentro de Google Colab."""
    if "google.colab" in sys.modules:
        return True
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False


def _raiz_drive() -> str:
    """Monta Drive (pide acceso al usuario) y devuelve la carpeta del proyecto.

    No pide acceso si Drive ya esta montado: `drive.mount` es idempotente pero
    imprime ruido, y en una sesion reanudada el punto de montaje ya existe.
    """
    from google.colab import drive

    def _mi_unidad() -> Optional[str]:
        # Colab actual crea "MyDrive"; versiones viejas creaban "My Drive".
        for nombre in ("MyDrive", "My Drive"):
            ruta = os.path.join(PUNTO_MONTAJE, nombre)
            if os.path.isdir(ruta):
                return ruta
        return None

    if _mi_unidad() is None:
        print("Montando Google Drive — acepta el permiso en la ventana que abre.")
        drive.mount(PUNTO_MONTAJE)

    base = _mi_unidad()
    if base is None:
        raise RuntimeError(
            f"Drive quedo montado en {PUNTO_MONTAJE} pero no aparece MyDrive. "
            f"Contenido: {os.listdir(PUNTO_MONTAJE) if os.path.isdir(PUNTO_MONTAJE) else 'no existe'}"
        )

    raiz = os.path.join(base, CARPETA_DRIVE)
    if not os.path.isdir(raiz):
        os.makedirs(raiz, exist_ok=True)
        print(f"Carpeta creada en Drive: {raiz}")
    else:
        print(f"Carpeta de Drive encontrada: {raiz}")
    return raiz


def resolver_raiz_proyecto() -> str:
    env = os.environ.get("CIREN_PROJECT_DIR")
    if env:
        os.makedirs(env, exist_ok=True)
        print(f"Raiz del proyecto por CIREN_PROJECT_DIR: {env}")
        return env
    if en_colab():
        return _raiz_drive()
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # codigo pegado en un notebook local o en un REPL
        return os.getcwd()


RUTA_PROYECTO = resolver_raiz_proyecto()
RUTA_BASE = os.path.join(RUTA_PROYECTO, "salidas")
CARPETA_ENTRENAMIENTO = os.path.join(RUTA_PROYECTO, "entrenamiento")
CARPETA_RESULTADOS = os.path.join(RUTA_PROYECTO, "resultados")

# --- PARTE 1 · Descarga de índices (guardar y leer) ---
DIR_INDICES = os.path.join(RUTA_BASE, "multi_indices")
OUTPUT_NPY = os.path.join(DIR_INDICES, "indices_12.npy")
OUTPUT_METADATA = os.path.join(DIR_INDICES, "metadata.npz")

# --- PARTE 2 · Checkpoint del ConvTransformer ---
CKPT_CONVTRANSFORMER = os.path.join(CARPETA_ENTRENAMIENTO, "convtransformer_12indices.pth")

# --- FASE 1 · Estabilidad continua de la atención (Etapa 1 del informe) ---
DIR_FASE1 = os.path.join(CARPETA_RESULTADOS, "fase1_estabilidad_continua")

# --- PARTE 3 · Discretización Top-P + Paradigma 1 (probabilístico) ---
DIR_SENSITIVITY_K = os.path.join(CARPETA_RESULTADOS, "sensitivity_K_experiment")
DIR_MATRICES = os.path.join(DIR_SENSITIVITY_K, "matrices")   # attention_seed_*.npy
CHECKPOINT_DIR = DIR_SENSITIVITY_K                            # model_seed_*.pth / theta_seed_*.npy

# --- PARTES 4-10 · Paradigmas (cada parte tiene SU carpeta) ---
DIR_PARADIGMAS = os.path.join(CARPETA_RESULTADOS, "paradigms_experiment")
DIR_P1 = os.path.join(DIR_PARADIGMAS, "01_probabilistic")
DIR_P2 = os.path.join(DIR_PARADIGMAS, "02_interventional")
DIR_P3 = os.path.join(DIR_PARADIGMAS, "03_geometric_topological")
DIR_TRIANGULACION = os.path.join(DIR_PARADIGMAS, "04_triangulation")
DIR_H1H3 = os.path.join(DIR_PARADIGMAS, "05_theorem_h1h3")
DIR_P4 = os.path.join(DIR_PARADIGMAS, "06_information_theory")
DIR_P5 = os.path.join(DIR_PARADIGMAS, "07_baselines")

# --- PARTE 11 · Aristas significativas ---
BASE_DIR = DIR_PARADIGMAS
OUTPUT_PATH = os.path.join(BASE_DIR, "significant_edges.json")

# --- PARTE 12 · Método 2 (especies CIREN) ---
SIG_EDGES_PATH = OUTPUT_PATH   # PARTE 11 escribe, PARTE 12 lee
# El shapefile del catastro MINAGRI/SAG (5 ficheros: .shp .shx .dbf .prj .cpg)
# tiene que estar SUBIDO a la carpeta del proyecto. Solo lo usa la PARTE 12; si
# falta, esa parte avisa y se salta en vez de reventar.
SHP_PATH = os.path.join(RUTA_PROYECTO, "viveros_2024_h19.shp")


def shapefile_disponible() -> bool:
    """True si estan los 4 ficheros que geopandas necesita para leer el .shp."""
    base = os.path.splitext(SHP_PATH)[0]
    faltan = [ext for ext in (".shp", ".shx", ".dbf", ".prj")
              if not os.path.exists(base + ext)]
    if faltan:
        print(f"  Shapefile incompleto en {RUTA_PROYECTO}: faltan {faltan}. "
              f"La PARTE 12 (metodo 2, especies CIREN) se saltara.")
        return False
    return True
DIR_M2 = os.path.join(DIR_PARADIGMAS, "13_ciren_species_v4_real")

# --- Alias de compatibilidad con el resto del código ---
OUTPUT_NPY_DIR = DIR_MATRICES
MATRICES_DIR = DIR_MATRICES
CACHE_DIR = os.path.join(DIR_PARADIGMAS, "matrices")
OUTPUT_DIR_M2 = DIR_M2

# Los checkpoints son lo mas pesado (model_seed_*.pth, ~30 MB x 50 semillas =
# ~1.5 GB) y escribirlos en Drive es lento. CIREN_CKPT_DIR los redirige, por
# ejemplo a /content/ckpt para que la corrida vaya rapido — a costa de perderlos
# al cerrar la sesion, con lo que la proxima corrida reentrena. Las matrices de
# atencion (12x12, unos KB) y los resultados se quedan siempre en Drive.
_CKPT_OVERRIDE = os.environ.get("CIREN_CKPT_DIR")
if _CKPT_OVERRIDE:
    CHECKPOINT_DIR = _CKPT_OVERRIDE
    print(f"Checkpoints redirigidos por CIREN_CKPT_DIR: {CHECKPOINT_DIR}")


def crear_arbol_directorios(verbose: bool = False) -> None:
    """Crea todas las carpetas del proyecto. Idempotente.

    Se llama al importar el modulo: en Colab la carpeta de Drive esta vacia la
    primera vez y cada parte del pipeline daba por hecho que su directorio ya
    existia (algunas hacen makedirs, otras no).
    """
    dirs = [RUTA_BASE, CARPETA_ENTRENAMIENTO, CARPETA_RESULTADOS,
            DIR_INDICES, DIR_FASE1, DIR_SENSITIVITY_K, DIR_MATRICES,
            CHECKPOINT_DIR, DIR_PARADIGMAS, DIR_P1, DIR_P2, DIR_P3,
            DIR_TRIANGULACION, DIR_H1H3, DIR_P4, DIR_P5, DIR_M2, CACHE_DIR]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        if verbose:
            print(f"  {d}")


crear_arbol_directorios()

# --- Búsqueda de matrices de atención ya creadas (orden de prioridad) ---
SEARCH_DIRS = [
    DIR_MATRICES,                                                        # canónico
    CACHE_DIR,
    os.path.join(CARPETA_RESULTADOS, "stability_experiment", "matrices"),
    os.path.join(CARPETA_RESULTADOS, "stability_50_experiment", "matrices"),
    os.path.join(RUTA_PROYECTO, "stability_experiment", "matrices"),      # legacy
    os.path.join(RUTA_PROYECTO, "stability_50_experiment", "matrices"),  # legacy
]

# --- Creación de carpetas (en un solo lugar, ordenado) ---
for _dir_a_crear in [DIR_INDICES, CARPETA_ENTRENAMIENTO, DIR_FASE1,
                     DIR_SENSITIVITY_K, DIR_MATRICES,
                     DIR_P1, DIR_P2, DIR_P3, DIR_TRIANGULACION,
                     DIR_H1H3, DIR_P4, DIR_P5, DIR_M2]:
    os.makedirs(_dir_a_crear, exist_ok=True)

EJECUTAR_PARTE1: bool = False #obtener índices
EJECUTAR_PARTE2_MULTISEED: bool = True  #entrenar 1 modelo por semilla → attention_seed_*.npy
EJECUTAR_FASE1: bool = True  #Etapa 1 del informe: estabilidad continua de A
EJECUTAR_PARTE3: bool = True #discretización Top-P + Paradigma 1 (probabilístico)
EJECUTAR_PARTE5: bool = True #paradigma 2
EJECUTAR_PARTE6: bool = True #paradigma 3
EJECUTAR_PARTE7: bool = True #triangulacion (5 paradigmas)
EJECUTAR_PARTE8: bool = True #h1,h2,h3 
# Paradigma 4 DESACTIVADO tras la prueba de independencia:
#   - D_JS y Wasserstein duplican las de la FASE 1 (misma familia).
#   - MI(arista; val_loss) con M=20 no tiene potencia; el 31/08 dio
#     I(G;L) = 0.000000 exacto con H(L|G) = H(L).
#   - I(G;L) >= I(e;L) se cumple por la desigualdad de procesamiento de
#     datos para cualquier modelo: la comparacion no puede fallar.
# Su unico rasgo a favor era no correlacionar con nada (|rho| <= 0.19),
# pero eso es lo que hace tambien el ruido. Se deja el codigo por si se
# reactiva con M >= 100 semillas, que es lo que la MI necesitaria.
EJECUTAR_PARTE9: bool = False #paradigma 4 (descartado, ver arriba)
EJECUTAR_PARTE10: bool = True #paradigma 5

MAX_RETRIES: int = 4
# 0 = carga en el proceso principal. Los frames ya estan enteros en RAM y el
# Dataset solo hace un slice, asi que los subprocesos no aportan y en Windows
# (spawn) ademas cuestan. En Colab (Linux, fork) tampoco hace falta y evita el
# "DataLoader worker killed" cuando la RAM queda justa. Cambiarlo solo si el
# perfilado muestra que la carga es el cuello de botella.
N_WORKERS: int = 0
DOWNLOAD_WORKERS: int = 4  # hilos para descarga paralela de imágenes GEE (I/O-bound, no relacionado con N_WORKERS del DataLoader)
SEQ_LENGTH: int = 1  # ConvTransformer de ÍNDICES: 1 snapshot (12 canales) → predice
                     # el snapshot siguiente. Coincide con el diagrama original
                     # ("Índices Multiespectrales: 12 canales, H×W", sin ventana
                     # temporal) — antes esto estaba en 7 y el tiempo se colaba
                     # como canales de entrada al encoder de índices, algo que el
                     # diagrama de "Atención entre Índices" nunca mostró. El manejo
                     # de tiempo real vive en el modelo temporal separado, abajo.
# Con TOKEN_PARCHE>0 hay ~16x mas muestras y cada una es ~16x mas pequena,
# asi que 16 dejaria pasos diminutos y epocas lentisimas. 128 es el valor con
# el que se midio la mejora (val 0.616 -> 0.210). Con TOKEN_PARCHE=0 se
# conserva el 16 historico.
BATCH_SIZE: int = 128
TOTAL_EPOCHS: int = 50
LR: float = 1e-3
LR_CONVTRANSFORMER: float = 3e-4  # el ConvTransformer de índices pasó de 2.6M a 12.77M
                                   # parámetros (embed_dim 32→256); con LR=1e-3 (afinado
                                   # para el modelo chico) diverge a NaN en la época 2-3
                                   # en las 50 semillas. El modelo temporal sigue usando
                                   # LR (1e-3) sin cambios — ya converge bien con eso.
PATIENCE: int = 5

# Épocas de rampa lineal del LR antes del descenso coseno. Sin rampa los pesos
# Q/K se mueven caóticamente en las primeras épocas y la atención colapsa en
# columnas sumidero antes de aprender nada; con rampa las 50 semillas parten
# del mismo régimen. Las épocas de warmup NO cuentan para el early stopping.
WARMUP_EPOCHS: int = 5

ALPHA: float = 0.01
FDR_TARGET: float = 0.05 
ALPHA_GRID = np.linspace(1.0, 0.0, 21)
N_MODELS_PER_SPECIES = 5
VAL_RATIOVAL = 0.2
P_VALUES = [0.5, 0.6, 0.7, 0.8, 0.9]
UMBRAL_DE_ESTABILIDAD: float = 0.8
TAU_CONSENSUS: float = UMBRAL_DE_ESTABILIDAD  # Paradigma 5: frecuencia mínima para consenso

# ---------------------------------------------------------------------------
# DISCRETIZACIÓN A → G: SOLO TOP-P, EN TODAS LAS ETAPAS
#
# Antes el código mezclaba dos discretizaciones: el Paradigma 1 usaba Top-P y
# los paradigmas 2 a 5 usaban Top-K con K=5 fijo. La triangulación cruzaba
# entonces conjuntos de aristas obtenidos de grafos distintos y con densidades
# distintas: a P=0.5 el Top-P dejaba ~7.7 aristas por fila (≈92 en total) y el
# Top-K exactamente 5 (60 en total). La Tabla 6 del informe combina coordenadas
# que deben venir del MISMO grafo, así que eso invalidaba la triangulación.
# Todo el Top-K está eliminado; la única discretización es discretizar().
# ---------------------------------------------------------------------------
P_DISCRETIZACION: float = 0.5   # P que usan los paradigmas 2-5. Debe coincidir
                                # con el P* que elige el Paradigma 1; si existe
                                # paradigm1_results.json se lee de ahí y este
                                # valor solo actúa de reserva.

# --- Margen η del operador adaptativo Top-P_η ---
#
# η se comparaba contra un 0.01 fijo, pero el margen real entre valores
# consecutivos de atención es mucho menor: medido sobre las matrices reales, la
# media del margen en el corte es ~0.006 a P=0.9. Al ser η mayor que casi todos
# los márgenes, el bucle de absorción de _select_top_p se comía toda la fila y
# k subía a 10.6 de 11. Con k ≈ N-1 el modelo nulo P_azar = k/(N-1) tiende a 1
# y NINGUNA arista puede salir significativa: era la causa mecánica de los
# "0 aristas BH" y del veredicto "Margen FALLA".
#
# Con ETA_MODO="percentil", η se fija en un percentil de la distribución real de
# márgenes (calcular_eta), así que el operador absorbe solo los cortes
# genuinamente ambiguos en lugar de todos.
ETA_MODO: str = "percentil"      # "percentil" | "fijo"
ETA_PERCENTIL: float = 25.0      # percentil de los márgenes observados
ETA_FIJO: float = 0.01           # valor si ETA_MODO == "fijo"

# ---------------------------------------------------------------------------
# IDENTIDAD DE LOS TOKENS (embedding posicional)
#
# Medido en un checkpoint entrenado: ||pos_embed|| = 0.23 frente a
# ||contenido|| = 108.0, o sea el 0.2% de la norma del token. Los 12 tokens
# quedaban identificados SOLO por el contenido de su imagen, y las imágenes de
# los 12 índices se parecen mucho entre sí (|r| medio 0.84). Si el modelo no
# puede distinguir QUÉ índice es cada token, no puede aprender relaciones
# específicas entre ellos: es candidato directo a explicar por qué la matriz de
# atención sale de rango 1 (R² = 0.98, todas las filas iguales).
#
# Dos cambios:
#   - LayerNorm después de frame_proj: pone la escala del token en el orden de
#     1 por dimensión, de modo que el pos_embed no queda sepultado por una
#     norma de 108. Es lo que hacen los ViT tras la proyección de parches.
#   - std de inicialización del pos_embed más alta (0.02 -> 0.2), para que la
#     señal de identidad sea del orden del 20% de la del contenido.
#
# OJO: añade el parámetro token_norm, así que los model_seed_*.pth existentes
# NO cargan. Hay que reentrenar la PARTE 2b al activar o desactivar esto.
USAR_TOKEN_NORM: bool = True
POS_EMBED_STD: float = 0.2

# ---------------------------------------------------------------------------
# NORMALIZACIÓN DE LA ATENCIÓN  (cambio 1)
# "softmax" | "sigmoid" | "softpick".  Ver normalizar_atencion() para el
# razonamiento y las referencias. NO cambia el diagrama: el bloque sigue siendo
# "MHA 4 cabezas, atención entre índices"; lo único distinto es cómo se
# normalizan los pesos dentro de esa atención.
ATENCION_NORMALIZACION: str = "sigmoid"

# ---------------------------------------------------------------------------
# TAREA DE ENTRENAMIENTO  (cambio 2)
#
#   "prediccion"  — original: dados los 12 índices en t, predice los 12 en t+1.
#                   Medido: sustituir toda la matriz de atención por la uniforme
#                   1/12 cuesta solo +2.2% de MSE, o sea la tarea NO obliga a
#                   usar información cruzada entre índices. Si el modelo no
#                   necesita mirar a los otros índices, su atención no puede
#                   contener una relación entre índices.
#
#   "enmascarado" — se pone a cero el canal j de la entrada y se pide
#                   reconstruir ESE canal en el mismo instante t. El modelo no
#                   tiene otra fuente que los otros 11 índices, así que la
#                   atención cruzada es obligatoria por construcción y la fila
#                   j de la matriz mide literalmente "de quién depende j".
#                   Respaldo: ChA-MAEViT (arXiv:2503.19331) y DiChaViT
#                   (NeurIPS 2024, arXiv:2405.16419).
#                   NO cambia la arquitectura: mismos seis bloques del diagrama,
#                   mismas dimensiones, mismos parámetros. Solo cambian entrada
#                   y objetivo. Como efecto lateral multiplica por 12 el número
#                   de muestras (una por índice enmascarado y fecha).
TAREA: str = "enmascarado"

# Número de índices ocultos SIMULTÁNEAMENTE en la tarea enmascarada.
#
#   k = 1  el índice j se reconstruye desde los otros 11. Con 12 índices de los
#          que EVI, EVI2, SAVI y KNDVI son casi colineales, basta con mirar UNO
#          para acertar, así que reparto uniforme y acierto son compatibles. Y
#          es lo que sale: entropía por fila 0.9919 sobre un máximo de 1.0, o
#          sea atención plana, y con ella ΔMSE máximo del 0.22% del error base.
#
#   k > 1  además de j se ocultan k-1 distractores sorteados en cada muestra. El
#          vecino colineal en el que el modelo se apoyaría está oculto parte del
#          tiempo, así que aprender un atajo fijo deja de funcionar y la fila j
#          tiene que discriminar. Es el mecanismo de DiChaViT (NeurIPS 2024,
#          arXiv:2405.16419) y ChA-MAEViT (arXiv:2503.19331).
#
# k=3→9 canales visibles ya sacó el gate Se (A/P) de 0.93x a 1.10x. Se prueba
# k=5 (7 canales visibles, mayor que el bloque colineal EVI/EVI2/SAVI/KNDVI de
# 4) para ver si ρrole sigue subiendo (0.533→0.664 con k=1→3) o si ya saturó
# y el techo pasa a ser la capacidad del encoder, no el enmascarado.
#
# La matriz de atención se LEE siempre con k=1 (ver MASCARA_K_LECTURA): así A
# sigue estando bien definida —la fila j es "de quién depende j teniendo los
# otros 11 disponibles"— y es comparable con las corridas anteriores. La tarea
# más dura moldea los pesos; la lectura se hace en la condición canónica.
MASCARA_K: int = 5
MASCARA_K_LECTURA: int = 1

# LECTURA DIAGONAL DE LA ATENCIÓN.
#
# forward_enmascarado devuelve B·N secuencias: la variante v de cada frame tiene
# el token v en ceros y es el único que se reconstruye (el decoder selecciona
# trans_out[:, idx, idx, :], o sea la diagonal). La fila v de la variante v es
# por tanto la única que responde a "de quién depende v cuando v está oculto";
# las otras N-1 filas de esa variante son tokens VISIBLES atendiendo mientras se
# reconstruye otro, una condición distinta.
#
# True   A[v, :] = fila v del rollout de la variante v (coherente con el
#        decoder y con la definición de A que usa el framework).
# False  promedio de las B·N matrices completas: cada fila mezcla 1 lectura
#        informativa con N-1 de espectador. Es el comportamiento anterior; se
#        conserva solo para reproducir las corridas ya publicadas.
# ---------------------------------------------------------------------------
# REPRODUCIBILIDAD.
#
# True  = una semilla fija produce SIEMPRE el mismo modelo y la misma matriz de
#         atencion. Obligatorio para que cualquier experimento sea falsable:
#         sin esto el efecto de un cambio queda dentro de la deriva entre
#         corridas (medido: aristas robustas 2/0/6/1 en cuatro corridas de la
#         misma configuracion). Cuesta ~10-20% de velocidad.
# False = cuDNN elige algoritmo por benchmarking y se habilita TF32. Mas
#         rapido, NO reproducible. Solo para pruebas exploratorias.
# ---------------------------------------------------------------------------
# TOKEN = PARCHE, NO IMAGEN COMPLETA.
#
# DIAGNOSTICO QUE LO MOTIVA. Con el token siendo la imagen entera hay 148
# frames de entrenamiento para 12.77M de parametros, y el modelo memoriza:
#
#     epoca   train    val
#         1  0.9797  0.8671
#        20  0.2458  0.6155
#        40  0.1353  0.6162     <- val plano desde la 20, train sigue bajando
#
# Brecha train/val de 4.5x. Y el resultado es 10.9x PEOR que una regresion
# lineal cerrada por pixel sobre la misma tarea (MSE 0.0563), que gana porque
# dispone de 148*52*52 = 400.192 puntos para 11 coeficientes mientras el
# ConvTransformer tiene 148 puntos para 12.77M de parametros.
#
# La relacion entre indices es POR PIXEL: la regresion lineal no usa ninguna
# informacion espacial y aun asi llega a 0.0563. Trocear el frame en parches
# conserva esa relacion, multiplica las muestras y encoge frame_proj y
# pred_linear, que son ~11M de los 12.77M.
#
# Medido con parche 13x13 (misma arquitectura, mismo optimizador, 20 epocas):
#
#     muestras train  148 -> 2368     parametros  12.77M -> 2.70M
#     val MSE       0.616 -> 0.210    brecha train/val  4.5x -> 1.5x
#     val / cota lineal 10.9x -> 3.7x
#
# 0 = comportamiento anterior (token = imagen completa).
# P > 0 = rejilla no solapada de parches PxP; cada parche es una muestra y la
#         atencion entre los 12 indices se mide dentro del parche.
TOKEN_PARCHE: int = 13

MODO_DETERMINISTA: bool = True

LECTURA_DIAGONAL: bool = True

# Nº de frames de validación usados para estimar A. None = todos.
# Antes estaba fijo en 4, que por el corte temporal son las 4 fechas más
# antiguas del bloque de validación, prácticamente un único momento fenológico.
N_FRAMES_LECTURA: Optional[int] = None
# Numero de frames/parches por pasada al leer la atencion. Cada uno genera N
# variantes, asi que la memoria escala con lote*N. Con parches de 13x13 cabe
# mucho mas que con frames de 52x52.
FRAMES_POR_LOTE_LECTURA: int = 64

# MODO DE ENMASCARADO.
#
# "aleatorio"  k-1 distractores sorteados de los 11 restantes. Es lo que había.
#              PROBLEMA MEDIDO: con k=5 se ocultan 4 distractores de 11, y la
#              probabilidad de que los 4 sean exactamente los hermanos de EVI
#              (NDVI/EVI2/SAVI/KNDVI) es 1/C(11,4) = 1/330 = 0.3%. En el 99.7%
#              de los sorteos queda al menos un hermano colineal visible, así
#              que reconstruir EVI nunca EXIGE mirar fuera de su familia
#              espectral. El comentario de MASCARA_K afirma que k=5 supera "el
#              bloque colineal de 4"; la aritmética dice que no: el número
#              esperado de hermanos ocultos es 4*(4/11) = 1.45 de 4.
#              Y los tres sumideros medidos (EVI, SAVI, EVI2) son justo los
#              miembros más redundantes de ese bloque.
#
# "familia"    se oculta la FAMILIA ESPECTRAL COMPLETA del índice a
#              reconstruir. Reconstruir EVI con NDVI, EVI2, SAVI y KNDVI
#              ocultos obliga a enrutar desde G2/G3/G4/G5 o a fallar. Es la
#              única variante en que la tarea REQUIERE dependencia
#              inter-familia, que es lo que la hipótesis principal afirma que
#              el modelo aprende.
MASCARA_MODO: str = "aleatorio"

# Familias espectrales: índices que comparten las bandas de las que se
# calculan (ver compute_12_indices). No es una elección libre — sale de las
# fórmulas, así que es ground truth del dominio y no un hiperparámetro.
#   G1  B8, B4 (+B2)     NDVI EVI EVI2 SAVI KNDVI
#   G2  B3, B5 (+B7)     MARI ARI CHL_REDEDGE
#   G3  B8|B8A, B11      NDMI NDII
#   G4  B3, B8           NDWI
#   G5  B4, B2, B6       PSRI
INDEX_FAMILIES: Dict[str, List[str]] = {
    "G1_NIR_ROJO":  ["NDVI", "EVI", "EVI2", "SAVI", "KNDVI"],
    "G2_RED_EDGE":  ["MARI", "ARI", "CHL_REDEDGE"],
    "G3_NIR_SWIR":  ["NDMI", "NDII"],
    "G4_VERDE_NIR": ["NDWI"],
    "G5_SENESC":    ["PSRI"],
}

# Renormalizar las filas de A a suma 1 al materializar attention_seed_*.npy.
#
# True  = comportamiento histórico. Necesario para leer A como cadena de
#         Markov, y obligatorio si se van a aplicar métodos compositivos
#         (CLR/ILR): con sigmoid las entradas son estrictamente positivas y la
#         fila renormalizada es una composición en el interior del simplex,
#         sin ceros y sin necesidad de imputación.
# False = conserva la suma de fila cruda. Con ATENCION_NORMALIZACION="sigmoid"
#         la suma de fila ES un dato: una fila que suma poco significa "este
#         índice decidió no mirar a ninguno". Renormalizarla la convierte en
#         UNIFORME, no en cero, e infla ratio_H hacia 1 — o sea puede estar
#         fabricando parte del veredicto "atención plana" (0.9956 medido).
#         Con False, entropia_normalizada sigue normalizando internamente, así
#         que hay que leer la suma de fila aparte (ver diagnostico_hipotesis).
RENORMALIZAR_FILAS: bool = True

# ---------------------------------------------------------------------------
# REGISTER TOKENS  (Darcet, Oquab, Mairal, Bojanowski — ICLR 2024,
# "Vision Transformers Need Registers")
#
# Los ViT depositan artefactos de norma alta en tokens de baja informacion:
# eso es el sumidero. Con tokens [reg] aprendibles el sumidero migra a ellos y
# el bloque token-token queda limpio. Medido aqui sin registros: entropia por
# fila 0.89-0.99 sobre un maximo de 1.0, y el nulo de perfil de columna
# reproduce el 84% de la estabilidad observada (0.464 de 0.554).
#
# AGNOSTICO: N_REGISTROS no depende del numero de tokens del dominio. El paper
# reporta que 1-4 bastan y que mas no ayuda; se deja 4.
#
# NO cambia los seis bloques del diagrama: la secuencia pasa de N a N+R tokens
# dentro del transformer, y los R se descartan antes del decoder. El analisis
# sigue siendo sobre una matriz N x N: se recorta el bloque token-token y se
# renormaliza por fila, guardando aparte cuanta masa se fue a los registros
# (FRACCION_A_REGISTROS en el reporte) — que es justamente el sumidero medido.
#
# CUIDADO: activar o cambiar esto INVALIDA las matrices attention_seed_*.npy
# ya cacheadas. Siguen siendo 12x12, asi que ningun chequeo de forma lo
# detecta. Hay que borrar resultados/sensitivity_K_experiment/matrices/.
# POR DEFECTO 0: el caso de ejemplo usa la arquitectura ORIGINAL del
# diagrama, sin registros. Con 0 el camino de datos es identico byte a
# byte al de antes. Poner 4 convierte los registros en una ABLACION
# declarada y medible, no en un cambio silencioso de arquitectura.
N_REGISTROS: int = 0

# dim_feedforward del bloque MLP. Antes se usaba el default de PyTorch (2048),
# que con d_model=128 es una expansion 16x: el MLP tiene capacidad para
# resolver la tarea token a token sin cruzar informacion, y entonces la
# atencion puede quedarse plana sin costo en la perdida. La razon habitual en
# la literatura es 4x. Con d_model=128 eso es 512.
# POR DEFECTO 2048: el default de PyTorch, que es lo que uso el caso de
# ejemplo. 512 (razon 4x sobre d_model=128) es la ALTERNATIVA a medir en
# la ablacion de hiperparametros, no el valor de produccion.
DIM_FEEDFORWARD: int = 2048

# ---------------------------------------------------------------------------
# AGREGACIÓN RASHOMON  (cambio 3)
#
# Los 50 entrenamientos son funcionalmente equivalentes: correlación media entre
# sus predicciones r = 0.9955, varianza común 99.5%, CoV del MSE 0.0096. Y aun
# así sus atenciones discrepan (ρ_role = 0.247). Eso es el efecto Rashomon, no
# ruido de estimación. La FASE 1 y el Paradigma 1 tratan esa discrepancia como
# algo a eliminar; la agregación Rashomon la trata como la incertidumbre real de
# la explicación y la reporta con intervalo de confianza.
# Respaldo: Rashomon Importance Distribution, arXiv:2309.13775 (NeurIPS).
RASHOMON_EPSILON: float = 0.05   # pertenece al conjunto si val_loss ≤ (1+ε)·mejor
RASHOMON_PCT_MIN: float = 0.70
# Tamaño mínimo del conjunto Rashomon. Con menos modelos, el IC al 90% del
# percentil por arista no tiene resolución (con 4 modelos daba 3 aristas
# "robustas" disjuntas de las 8 fuertes de la triangulación).
RASHOMON_M_MIN: int = 12   # una arista es robusta si su IC de percentil
                                 # queda entero por encima de este valor

# ---------------------------------------------------------------------------
# UMBRALES CALIBRADOS CONTRA MODELO NULO  (cambio 5)
#
# Los umbrales de la Tabla 7 (ρ > 0.8, GED < 0.2, ρ_role > 0.9, D_JS < 0.1...)
# son constantes puestas a mano: el informe no da derivación ni referencia para
# ellas. Eso ya nos costó dos lecturas equivocadas en este proyecto:
#   - el Jaccard "estable" de 0.98 resultó ser 1.0x el de un grafo aleatorio de
#     la misma densidad, o sea puro artefacto de densidad;
#   - el Z-score de tamaño contra Erdős-Rényi daba 4.70 solo porque el nulo
#     tenía otra densidad.
# En los dos casos el problema no era la métrica sino la ausencia de nulo.
#
# Con esto activado, cada métrica de estabilidad se compara además contra dos
# modelos nulos construidos a partir de las MISMAS matrices (ver
# generar_nulo_atencion), y se reporta Z-score y percentil. Un valor "malo"
# contra el umbral fijo pero muy por encima del nulo sigue siendo señal; un
# valor "bueno" que el nulo también alcanza no lo es.
# Referencias de contexto: M4 (arXiv:2409.16756), F-Fidelity (arXiv:2410.02970).
CALIBRAR_UMBRALES: bool = True

# ---------------------------------------------------------------------------
# ANALISIS RETIRADOS
#
# No estan comentados ni borrados del fichero: quedan desactivados y con el
# motivo escrito, porque el motivo ES un resultado del trabajo y hay que poder
# citarlo. Lo que se elimina es su participacion en el VEREDICTO.
#
#   Granger          0 aristas con muestreo irregular (huecos de semanas a
#                    meses). El Jaccard 0 es por definicion, no por hallazgo.
#   Deps. condic.    0 de 2402 pares sobreviven a BH; con M=20 el p minimo
#                    alcanzable (2.97e-3) no llega al que BH exige (2.08e-5).
#   AUC de entropia  cambio de entropia de la MATRIZ: nunca ejecuta el modelo.
#                    Rango 0.0025-0.0032 en las 132 aristas: no discrimina, y
#                    el umbral 0.005 de la Tabla 7 esta por encima del maximo
#                    alcanzable.
#   GED normalizada  = 1 - Jaccard exacto (suma 1.000000, desviaciones
#                    identicas). No es una segunda evidencia, es la misma.
#   Rob_K            definida sobre K in {2,3,4,5}, o sea Top-K. Contradice la
#                    decision de usar solo Top-P por agnosticismo. Su
#                    equivalente correcto es Rob_P sobre valores de P.
#   Z de tamano ER   ~0 por construccion al igualar densidad. Sanity check del
#                    nulo, no evidencia de estructura.
#   Swap Jaccard     (|E|-k)/(|E|+k): aritmetica del tamano de la perturbacion,
#                    no del efecto de la identidad de las aristas.
USAR_GRANGER: bool = False
USAR_DEPS_CONDICIONALES: bool = False
USAR_VEREDICTO_AUC_ENTROPIA: bool = False
USAR_ROB_K: bool = False           # su reemplazo agnostico es Rob_P

# ---------------------------------------------------------------------------
# VEREDICTO AGREGADO: RETIRADO
#
# Bajo cualquier operador de corte por umbral la esparsidad es S ~ 1-P y el
# Jaccard entre replicas crece con P (medido con 50 semillas: P=0.5 -> J=0.619,
# P=0.7 -> 0.781, P=0.9 -> 0.872). Entonces S>0.7 exige P<0.3 y J>0.7 exige
# P>=0.7: la interseccion es VACIA. No es un problema de calibracion, es
# aritmetica del operador, y no depende de N.
#
# El propio informe enuncia esto como condicion de falsacion del eslabon 5
# ("falsaria la hipotesis que ningun K satisfaga simultaneamente los umbrales
# de estabilidad y de compacidad"). Con los datos medidos, se cumple.
#
# Por eso no se emite un GLOBAL de APRUEBA/FALLA agregando criterios
# incompatibles: se reporta la FRONTERA, o sea que cumple cada P y que no. Un
# marco donde todos los criterios pueden aprobar a la vez no esta midiendo
# nada.
EMITIR_VEREDICTO_GLOBAL: bool = False
N_NULOS_UMBRAL: int = 40         # realizaciones del ensemble nulo
# Nulos para calibrar la BATERÍA completa de 9 métricas de la FASE 1. Van
# aparte porque cada uno exige recalcular las 9 sobre C(M,2) pares, mucho
# más caro que las 3 de calibrar_metricas_estabilidad.
N_NULOS_FASE1: int = 20
# Una métrica sólo puede decir APRUEBA si además le gana a su nulo uniforme
# con este p empírico. Con 20 nulos el suelo alcanzable es 1/21 = 0.048.
P_MAX_FASE1: float = 0.10
MAX_PARES_CALIBRACION: int = 120  # pares de matrices por ensemble

# ---------------------------------------------------------------------------
# ΔMSE REAL DE ABLACIÓN (Etapa 4 del informe, §8.2)
#
# El Paradigma 2 medía el cambio de ENTROPÍA de la matriz al ablacionar una
# arista, sin tocar el modelo ni los datos. El informe define otra cosa:
#     ΔMSE(e) = MSE(Ã(e)) − MSE(A)
# que exige re-ejecutar el modelo con la matriz ablacionada. Los umbrales de la
# Tabla 7 (ΔMSE > 0.01 aprueba, AUC > 0.005 aprueba) están definidos para esa
# curva, así que compararlos con el AUC de entropía era un error de categoría:
# de ahí el "0/132 aristas aprueban" de todas las corridas anteriores.
#
# Con esto activado el Paradigma 2 carga el mejor checkpoint y la validación, e
# impone cada matriz ablacionada mediante ConvTransformer.forward_con_atencion.
# Cuesta una pasada de validación por (arista × α), así que se limita a las
# aristas del grafo Top-P.
EJECUTAR_DELTA_MSE: bool = True
# Umbral de fidelidad del paradigma 2. El de la Tabla 7 (ΔMSE > 0.01 absoluto)
# es inalcanzable con este mecanismo: la ablación suave renormaliza la fila, y
# quitar 1 de ~6 aristas de una fila casi plana mueve el MSE en el tercer o
# cuarto decimal. Medido: máx +0.00562 en Aconcagua sobre mse_base 0.795, y
# +0.00217 en Pencahue sobre 1.000. Ninguna de las 142 aristas evaluadas en las
# dos corridas se acercó a 0.01, así que el umbral no separaba nada: descartaba
# todo por igual.
# Se sustituye por dos criterios que sí son informativos:
#   1) RELATIVO: ΔMSE(e)/mse_base por encima de DELTA_MSE_REL_MIN.
#   2) CALIBRADO: por encima del percentil DELTA_MSE_PCTL_NULO de los ΔMSE de
#      aristas que NO están en el grafo Top-P. Si quitar una arista del grafo no
#      degrada más que quitar una que el operador ya había descartado, el grafo
#      no está señalando nada.
DELTA_MSE_REL_MIN: float = 0.002        # 0.2% del error base
DELTA_MSE_PCTL_NULO: float = 95.0
N_NOARISTAS_NULO: int = 24              # aristas fuera del grafo a ablacionar

# Edge swapping (paradigma 2). Intercambiar UNA arista da un Jaccard constante
# por aritmética: con |E|=69, cambiar 1 arista deja 67/69 = 0.9710 siempre, y
# eso es exactamente lo que salió (356 pares, σ = 1.1e-16, un solo valor). No
# medía sensibilidad estructural, medía que se cambió una arista. Ahora se
# intercambia una fracción del grafo y se repite con distintos sorteos, así el
# Jaccard tiene varianza y la comparación con 0.7 significa algo.
SWAP_FRACCION: float = 0.20            # fracción de aristas intercambiadas
SWAP_REPETICIONES: int = 200           # sorteos independientes

# Dependencias condicionales (paradigma 5). El ratio P(e2|e1=1)/P(e2|e1=0)
# divergía cuando el denominador era 0 exacto: los 30 pares del top salían con
# ratio = 1e6 (el tope de max(p, 1e-6)), o sea empatados, y el orden era
# arbitrario. Con M=20 y n(e1=1)=5, un P(e2|e1=1)=1.0 son 5 de 5, cuyo IC al
# 95% baja hasta 0.48: no hay potencia. Ahora se usa el odds ratio con
# corrección de Haldane (+0.5 a las cuatro celdas), que es finito siempre, se
# ordena por la diferencia de riesgos (acotada en [-1,1]) y se exige un mínimo
# de observaciones en las dos ramas.
COND_N_MIN: int = 5                    # mínimo de modelos en cada rama

# ---------------------------------------------------------------------------
# FASE 1 · Umbrales de aprobación/fallo (Tabla 7 del informe, etapa 1)
# ---------------------------------------------------------------------------
FASE1_DROP_DIAGONAL: bool = True     # el grafo no tiene bucles: se evalúa fuera de la diagonal
FASE1_MAX_PARES: int = 0             # 0 = todos los C(M,2) pares; >0 = submuestreo
UMBRALES_FASE1: Dict[str, Tuple[float, float]] = {
    # métrica: (aprueba_si, falla_si)   — la dirección la fija cada chequeo
    "djs":      (0.10, 0.30),   # D_JS media entre pares (nats)
    "rho":      (0.80, 0.50),   # Spearman medio por fila
    "tau":      (0.60, 0.40),   # Kendall medio por fila
    "ratio_H":  (0.90, 0.95),   # H_i / ln(N_cand): plano = malo
    "wasserstein": (0.20, 0.50),  # Wp medio entre pares (Tabla 7, etapa 1)
    "linf":     (0.20, 0.50),   # norma inducida L_inf entre pares
    "cv":       (0.50, 1.00),   # CV_ij por entrada
    "hcol":     (0.90, 0.70),   # H_col / ln(N): fracción del máximo
    "cmax":     (0.20, 0.30),   # concentración máxima por columna (sinks)
}

# ---------------------------------------------------------------------------
# ALCANZABILIDAD DE LOS UMBRALES  (agnostica al dominio)
#
# El framework se declara agnostico: N puede ser cualquier numero de tokens, y
# por eso el operador es Top-P y no Top-K. Pero varios umbrales del Cuadro 5 no
# son alcanzables para cualquier combinacion de (N, P, M), y un umbral que
# ningun dato puede pasar no mide el fenomeno: mide el presupuesto de computo.
# Estas funciones lo calculan ANTES de correr.
# ---------------------------------------------------------------------------

def m_minimo_significancia(n_nodos: int, top_p: float,
                           alpha: float = None,
                           n_empates: int = 1) -> Dict:
    """Cuantas replicas M hacen falta para que ALGUNA arista pueda pasar BH.

    Bajo Top-P la probabilidad nula por fila es p_azar ~ P (medido: P=0.5 da
    0.511, P=0.9 da 0.904). El p-valor mas pequeño alcanzable corresponde a una
    arista presente en las M replicas: p_min = p_azar^M.

    Benjamini-Hochberg rechaza el mayor k con p_(k) <= alpha*k/n_tests. Si solo
    una arista es significativa hace falta p_min <= alpha/n_tests; si m aristas
    empatan en el minimo basta p_min <= alpha*m/n_tests, que es mas facil. Se
    devuelven las dos cotas porque la realidad esta en medio.

        M >= ln(n_tests / (alpha * m)) / ln(1 / P)

    M crece con log(N), no con N^2: duplicar los tokens casi no cuesta
    replicas. Es el resultado que hace viable el agnosticismo.

    Comprobado contra el submuestreo de las 50 semillas (N=12, P=0.5,
    alpha=0.05): la cota conservadora da M>=12 y la de empates con m=13 da
    M>=8; el conteo empirico de aristas significativas pasa de 0 a 13.2 justo
    entre M=5 y M=8.
    """
    alpha = FDR_TARGET if alpha is None else float(alpha)
    n_tests = max(1, n_nodos * (n_nodos - 1))
    p_azar = min(max(float(top_p), 1e-9), 1 - 1e-9)
    denom = np.log(1.0 / p_azar)
    m_conservador = int(np.ceil(np.log(n_tests / alpha) / denom))
    m_empates = int(np.ceil(np.log(n_tests / (alpha * max(1, n_empates)))
                            / denom))
    return dict(n_nodos=n_nodos, n_tests=n_tests, top_p=p_azar, alpha=alpha,
                m_min_conservador=m_conservador,
                m_min_con_empates=m_empates, n_empates=n_empates)


def sparsity_alcanzable(top_p: float, umbral_sparsity: float = 0.7) -> Dict:
    """Sparsity y Jaccard son CONTRADICTORIOS bajo Top-P. Aqui se cuantifica.

    Con Top-P la densidad del grafo es ~P, asi que S = 1 - densidad ~ 1 - P,
    independiente de N (por eso el operador es agnostico). Entonces:

        S > 0.7  exige  P < 0.3

    Pero el Jaccard entre replicas CRECE con P (medido con 50 semillas:
    P=0.5 -> J=0.619, P=0.7 -> 0.781, P=0.9 -> 0.872), asi que J > 0.7 exige
    P >= 0.7. No existe P que cumpla los dos: el intervalo esta vacio.

    El propio informe enuncia esto como condicion de falsacion del eslabon 5
    ("Falsaria la hipotesis que ningun K satisfaga simultaneamente los umbrales
    de estabilidad y de compacidad"). Con los datos medidos, esa condicion se
    cumple: el eslabon 5 esta falsado tal como esta especificado.

    Salida: el P que haria falta para cada umbral y si la interseccion existe.
    """
    p_max_sparsity = 1.0 - umbral_sparsity
    return dict(top_p=float(top_p),
                sparsity_esperada=1.0 - float(top_p),
                umbral_sparsity=umbral_sparsity,
                p_max_para_sparsity=p_max_sparsity,
                cumple_sparsity=bool((1.0 - float(top_p)) > umbral_sparsity),
                interseccion_vacia=bool(p_max_sparsity < 0.7),
                nota=("S ~ 1-P y J crece con P: S>0.7 pide P<0.3, J>0.7 pide "
                      "P>=0.7. Interseccion vacia — no es un problema de "
                      "calibracion, es aritmetica del operador."))


def reportar_alcanzabilidad(n_nodos: int, top_p: float, m_replicas: int,
                            n_empates: int = 1) -> Dict:
    """Imprime, antes de correr, que umbrales puede pasar esta configuracion."""
    sig = m_minimo_significancia(n_nodos, top_p, n_empates=n_empates)
    spa = sparsity_alcanzable(top_p)
    print(f"\n{'='*86}")
    print(f" ALCANZABILIDAD DE LOS UMBRALES  ·  N={n_nodos} tokens · "
          f"P={top_p} · M={m_replicas} replicas")
    print(f"{'='*86}")
    print(f"  Significancia BH (alpha={sig['alpha']}, "
          f"{sig['n_tests']} tests):")
    print(f"    M minimo si solo 1 arista es significativa : "
          f"{sig['m_min_conservador']}")
    print(f"    M minimo si {sig['n_empates']} empatan en el minimo   : "
          f"{sig['m_min_con_empates']}")
    if m_replicas < sig["m_min_con_empates"]:
        print(f"    M={m_replicas} NO ALCANZA: ninguna arista puede pasar BH "
              f"con esta P y este N.")
        print(f"    Opciones: subir M, bajar P, o preinscribir menos hipotesis "
              f"(n_tests baja y con el la cota).")
    else:
        print(f"    M={m_replicas} alcanza.")
    print(f"  Compacidad vs estabilidad:")
    print(f"    Sparsity esperada con P={top_p}: "
          f"{spa['sparsity_esperada']:.3f}  "
          f"(umbral >{spa['umbral_sparsity']}: "
          f"{'cumple' if spa['cumple_sparsity'] else 'NO cumple'})")
    if spa["interseccion_vacia"]:
        print(f"    CONTRADICCION: {spa['nota']}")
    print(f"{'='*86}")
    return dict(significancia=sig, sparsity=spa, m_replicas=m_replicas)


# ---------------------------------------------------------------------------
# PARADIGMA 1 · Estadística frecuentista sobre Top-P (Etapas 3-4 del informe)
# ---------------------------------------------------------------------------
ETA_MARGEN: float = ETA_FIJO # valor EFECTIVO de η. Si ETA_MODO=="percentil",
                             # calcular_eta() lo recalcula desde las matrices
                             # reales antes de usarlo. No lo pongas como valor
                             # por defecto de un argumento: los defaults se
                             # evalúan al definir la función y no verían el
                             # recálculo. Léelo dentro del cuerpo.
N_PERMUTACIONES: int = 1000  # B del test de permutaciones
N_NULOS_GRAFO: int = 500     # realizaciones del modelo nulo para los Z-scores
CI_LEVEL: float = 0.95       # nivel del IC de Clopper-Pearson para Φ(e)
SEMILLA_NULO: int = 20240517 # reproducibilidad de los modelos nulos

# Tamaños del top por fila, indexados por (seed, P) para no pisarse entre P
GLOBAL_TOP_P_SIZES: Dict[Tuple[int, float], List[int]] = {}

# Los 12 índices en orden fijo
INDEX_NAMES: List[str] = [
    "NDVI", "MARI", "ARI", "EVI", "EVI2", "NDWI",
    "NDMI", "CHL_REDEDGE", "NDII", "SAVI", "PSRI", "KNDVI",
]

NUM_INDICES: int = len(INDEX_NAMES)

# Sentinel-2 SR (COPERNICUS/S2_SR_HARMONIZED) entrega la reflectancia escalada
# x10000 (enteros ~0-10000, no 0-1). Los indices de diferencia normalizada
# (NDVI, NDWI, NDMI, ...) son invariantes a escala y sobreviven sin dividir,
# pero EVI y SAVI llevan constantes aditivas EXPRESADAS EN REFLECTANCIA:
#   EVI  = 2.5(N-R) / (N + 6R - 7.5B + 1)     <- el "+1"
#   SAVI = 1.5(N-R) / (N + R + 0.5)           <- el "+0.5"
# Con valores de miles esas constantes son despreciables y el indice degenera:
# SAVI colapsa a 1.5*NDVI (medido en los datos viejos: SAVI = 1.219*NDVI+0.097,
# r=0.975) y el denominador de EVI puede cruzar cero. Sin este factor, 4 de los
# 12 "indices" no eran indices.
S2_SCALE: float = 10000.0

# Piso de reflectancia para los índices que invierten una banda (ARI,
# MARI, CHL_REDEDGE). Sin él, un píxel enmascarado —que llega como 0
# exacto desde ee_to_numpy— produce 1/0 = inf. 0.01 (1% de reflectancia)
# está por debajo de cualquier superficie real, así que no distorsiona
# píxeles válidos y sí acota los enmascarados.
REFL_MIN: float = 0.01

# Rango fisico de cada indice. Antes habia un clamp global a [-1,1] que no
# corresponde a todos: ARI es un cociente de inversos (no acotado) y SAVI llega
# a +-1.5 por construccion. Ese clamp unico dejaba 43% de ARI, 48% de EVI, 41%
# de EVI2 y 22% de SAVI pegados exactamente en +1 — canales casi constantes.
INDEX_CLAMP: Dict[str, Tuple[float, float]] = {
    "NDVI":        (-1.0, 1.0),   # diferencia normalizada
    "MARI":        (-10.0, 10.0),  # ARI escalado por B7
    "ARI":         (-30.0, 30.0),  # 1/B3 - 1/B5: inversos, no acotado
    "EVI":         (-1.0, 1.0),   # rango util con reflectancia correcta
    "EVI2":        (-1.0, 1.0),
    "NDWI":        (-1.0, 1.0),
    "NDMI":        (-1.0, 1.0),
    "CHL_REDEDGE": (-1.0, 15.0),  # razón B7/B5 - 1: >= -1, sin techo
    "NDII":        (-1.0, 1.0),
    "SAVI":        (-1.5, 1.5),   # factor 1.5 por construccion
    "PSRI":        (-2.0, 2.0),   # (B4-B3)/B7: no acotado
    "KNDVI":       ( 0.0, 1.0),   # tanh(.) >= 0
}
INDEX_CLAMP_LO = np.array([INDEX_CLAMP[nm][0] for nm in INDEX_NAMES], dtype=np.float32)
INDEX_CLAMP_HI = np.array([INDEX_CLAMP[nm][1] for nm in INDEX_NAMES], dtype=np.float32)

# Hueco temporal maximo admitido dentro de una ventana de entrenamiento. Las
# imagenes sobrevivientes al filtro de nubes NO estan equiespaciadas (Sentinel-2
# revisita cada 5 dias, pero tras filtrar nubes quedan huecos de semanas o
# meses). El modelo trata la ventana como equiespaciada, asi que una ventana que
# cruza un hueco grande le ensena una dinamica que no existe. 0 = desactivado.
MAX_GAP_DAYS: int = 45

# ---------------------------------------------------------------------------
# Conexión residual entrada -> salida:  y(t+1) = x(t) + delta
#
# Sin ella, el modelo tiene que RECONSTRUIR los 2652 píxeles de cada índice
# desde un vector de 128 dimensiones en cada predicción, y la persistencia
# (copiar el frame t, MSE 0.147) ni siquiera es representable. Con ella, la
# persistencia es el punto de partida gratis y el modelo solo aprende el
# cambio de t a t+1, que es lo único que hay que predecir.
#
# Medido con 3 semillas x 40 épocas sobre los datos reales:
#     variante                              val MSE   vs persistencia
#     persistencia (baseline) ............   0.1471        1.00x
#     residual + modelo pequeño ..........   0.1415        0.96x  GANA
#     residual (modelo original) .........   0.1432        0.97x  GANA
#     SIN residual, modelo pequeño .......   0.2397        1.63x  pierde
#     SIN residual (original) ............   0.2409        1.64x  pierde
# El tamaño del modelo es irrelevante; el residual lo es todo (-41% de MSE).
#
# OJO: cambia el significado de los checkpoints. Los model_seed_*.pth
# entrenados sin residual cargan igual (mismos parámetros) pero predicen otra
# cosa. Hay que reentrenar la PARTE 2b al activar o desactivar esto.
USAR_RESIDUAL: bool = True

# ---------------------------------------------------------------------------
# Qué matriz consumen la FASE 1 y los 5 paradigmas.
#
# El rollout con residual_fraction=0.5 y L=2 equivale a
#     0.25·I + 0.25·A1 + 0.25·A2 + 0.25·A2A1,
# así que ~el 77% de su diagonal la inyecta la medición. Además promedia las
# 4 cabezas, y las cabezas NO hacen lo mismo: medido sobre 50 semillas, la
# estabilidad entre semillas (Spearman por fila, sin diagonal) es
#     rollout residual=0.5 .... 0.223
#     rollout residual=0.0 .... 0.338
#     A1 promediada ........... 0.183
#     A1 cabeza más estable ... 0.507   <- 2.3x mejor que el rollout
# La cabeza más estable tiene además R²_rango1 = 0.216 (estructura que SÍ
# depende de qué índice pregunta), frente a 0.91 del rollout (todas las
# filas iguales = solo columnas sumidero).
#
#   "rollout"   → comportamiento original (reproduce corridas viejas)
#   "producto"  → A2·A1 sin el término residual: mismo encadenado de capas
#                 que el rollout pero SIN la identidad inyectada. Es la
#                 opción recomendada: sube rho de 0.223 a 0.338 y no
#                 depende de ninguna elección arbitraria.
#   "capa1"/"capa2" → A1 o A2 crudas, promediadas sobre cabezas
#   "cabeza_estable" → NO RECOMENDADA. Emparejar cabezas entre semillas
#                 sólo tiene sentido si las cabezas forman grupos
#                 reproducibles, y NO los forman: medido sobre 50
#                 semillas, silueta = 0.016 y el 44% de las cabezas está
#                 más cerca del centroide de otra ranura que de la suya.
#                 Con emparejamiento degenerado, "la mejor cabeza" cambia
#                 de identidad y su rho oscila entre 0.42 y 0.58 según el
#                 orden de las semillas. El código avisa y se niega a
#                 usarla si la silueta baja de SILUETA_MINIMA.
# Por defecto "rollout": cambiar la fuente altera TODOS los números aguas
# abajo (FASE 1 y los 5 paradigmas), así que es una decisión explícita,
# no un cambio silencioso. Ver la tabla comparativa en el README del
# diagnóstico: ninguna opción llega al umbral de 0.8.
FUENTE_ATENCION: str = "rollout"
SILUETA_MINIMA: float = 0.25   # calidad mínima del agrupamiento de cabezas
CAPA_CABEZA: int = 0          # capa de la que se toma la cabeza (0 = A1)
CABEZA_FORZADA: Optional[int] = None   # None = elegir la más estable

# 50 semillas. Las primeras 20 son las mismas de antes (mismos nombres de
# fichero attention_seed_*.npy, cache reutilizable), más 30 nuevas para
# recuperar la potencia estadística que la corrida de 20 no tenía (Rashomon
# fallaba el IC por margen, Paradigma 1 salía DUDOSO).
# C(50,2) = 1225 pares en la FASE 1.
SEEDS: List[int] = [
    42, 123, 7, 2024, 999, 314, 271, 555, 888, 1337,
    100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
    11, 22, 33, 44, 55, 66, 77, 88, 99, 111,
    222, 333, 444, 5555, 6666, 7777, 13, 17, 19, 23,
    29, 31, 37, 41, 43, 47, 53, 59, 61, 67,
]

# ============================================================================
# Descarga de índices 
# ============================================================================

def init_earth_engine(project_id: str = PROJECT_ID) -> None:
    """Inicializa Earth Engine, autenticando si hace falta.

    En Colab la autenticacion es interactiva: imprime un enlace, se pega el
    codigo y las credenciales quedan en /root/.config/earthengine, que se borra
    al cerrar la sesion. Por eso hay que reautenticar en cada sesion nueva de
    Colab, aunque la carpeta de Drive persista.
    """
    if not HAS_EE:
        raise RuntimeError(
            "earthengine-api no esta instalado. En Colab:\n"
            "  !pip -q install earthengine-api geemap")
    try:
        ee.Initialize(project=project_id)
        return
    except Exception as e:
        print(f"  EE sin inicializar ({type(e).__name__}); autenticando...")
    try:
        if en_colab():
            # auth_mode="notebook" evita depender de un navegador local, que en
            # Colab no existe. Las versiones nuevas de la libreria ya lo eligen
            # solas, pero pedirlo explicito no molesta y arregla las viejas.
            ee.Authenticate(auth_mode="notebook")
        else:
            ee.Authenticate()
    except TypeError:
        ee.Authenticate()          # firma vieja, sin auth_mode
    ee.Initialize(project=project_id)
    print(f"  Earth Engine listo (proyecto {project_id})")

def obtener_tile_mas_frecuente(coleccion) -> Optional[str]:
    tiles = coleccion.aggregate_array("MGRS_TILE").getInfo()
    if not tiles:
        return None
    conteo: dict = {}
    for t in tiles:
        conteo[t] = conteo.get(t, 0) + 1
    return max(conteo, key=conteo.get)

def calcular_area_interes(coords: List[float], tamano: float):
    return ee.Geometry.Point(coords).buffer(tamano / 2).bounds()

def mask_clouds_qa60_scl(image: ee.Image) -> ee.Image:
    qa = image.select("QA60")
    qa_mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    scl = image.select("SCL")
    scl_mask = scl.expression(
        "(b == 4) || (b == 5) || (b == 6) || (b == 7) ? 1 : 0",
        {"b": scl},
    )
    return image.updateMask(qa_mask.And(scl_mask))

def compute_12_indices(image: ee.Image) -> ee.Image:
    b2  = image.select("B2").divide(S2_SCALE)     # Blue
    b3  = image.select("B3").divide(S2_SCALE)     # Green
    b4  = image.select("B4").divide(S2_SCALE)     # Red
    b5  = image.select("B5").divide(S2_SCALE)     # Red edge 705
    b6  = image.select("B6").divide(S2_SCALE)     # Red edge 740 (PSRI)
    b7  = image.select("B7").divide(S2_SCALE)     # Red edge 783
    b8  = image.select("B8").divide(S2_SCALE)     # NIR ancha, 833 nm, 10 m
    b8a = image.select("B8A").divide(S2_SCALE)    # NIR estrecha, 865 nm, 20 m
    b11 = image.select("B11").divide(S2_SCALE)    # SWIR 1610, 20 m

    eps = 1e-6

    # 1. NDVI
    ndvi = b8.subtract(b4).divide(b8.add(b4).add(eps)).rename("NDVI")
    # Denominadores protegidos para los índices de inversos.
    b3s = b3.max(REFL_MIN)
    b5s = b5.max(REFL_MIN)

    # 2. MARI (Modified Anthocyanin Reflectance Index, Gitelson 2001)
    #    MARI = (1/R550 - 1/R700) * R780  ->  (1/B3 - 1/B5) * B7
    #    Antes: (B7-B5)/(B7+B5), que es NDRE — ni usaba B3 ni era MARI.
    mari = b3s.pow(-1).subtract(b5s.pow(-1)).multiply(b7).rename("MARI")
    # 3. ARI (Anthocyanin Reflectance Index, Gitelson 2001)
    #    ARI = 1/R550 - 1/R700  ->  1/B3 - 1/B5
    #    Antes: (1/B5 - 1/B7)*B8 — bandas equivocadas (era una forma
    #    tipo MARI sobre el par red-edge, no ARI).
    ari = b3s.pow(-1).subtract(b5s.pow(-1)).rename("ARI")
    # 4. EVI
    evi = b8.subtract(b4).multiply(2.5).divide(
        b8.add(b4.multiply(6)).subtract(b2.multiply(7.5)).add(1).add(eps)
    ).rename("EVI")
    # 5. EVI2
    evi2 = b8.subtract(b4).multiply(2.5).divide(
        b8.add(b4.multiply(2.4)).add(1).add(eps)
    ).rename("EVI2")
    # 6. NDWI (McFeeters)
    ndwi = b3.subtract(b8).divide(b3.add(b8).add(eps)).rename("NDWI")
    # 7. NDMI (Normalized Difference Moisture Index)
    #    Con B8A (865 nm) en vez de B8 (833 nm). Dos motivos:
    #      a) B8A y B11 son ambas nativas de 20 m. B8 es de 10 m, así que
    #         mezclarla con B11 metía dos resoluciones en un mismo índice.
    #         La formulación estándar de NDMI en Sentinel-2 usa B8A.
    #      b) B8A es estrecha (20 nm) y evita la absorción de vapor de agua
    #         de 940 nm que sí toca la cola de B8 (115 nm de ancho), justo lo
    #         que importa en un índice de humedad.
    #    Efecto lateral buscado: NDMI y NDII dejan de ser el mismo canal.
    #    NDII se queda en B8 (tabla de la presentación) y NDMI pasa a B8A.
    ndmi = b8a.subtract(b11).divide(b8a.add(b11).add(eps)).rename("NDMI")
    # 8. CHL_REDEDGE (Chlorophyll Index red-edge, Gitelson)
    #    CI_rededge = R783/R705 - 1  ->  B7/B5 - 1
    #    Antes: (B8A-B5)/(B8A+B5), otra vez NDRE — por eso correlacionaba
    #    r=0.99 con el "MARI" viejo: eran el mismo índice.
    chl = b7.divide(b5s).subtract(1).rename("CHL_REDEDGE")
    # 9. NDII (Normalized Difference Infrared Index, Hardisky 1983)
    #     Con B8 (833 nm), según la tabla de bandas de la presentación.
    #     Ya NO es idéntico a NDMI: ese usa B8A (865 nm). Siguen midiendo lo
    #     mismo (contraste NIR-SWIR) y correlacionarán alto (~0.97), pero son
    #     dos canales distintos, no una copia bit a bit.
    ndii = b8.subtract(b11).divide(b8.add(b11).add(eps)).rename("NDII")
    # 10. SAVI (L=0.5)
    savi = b8.subtract(b4).multiply(1.5).divide(
        b8.add(b4).add(0.5).add(eps)
    ).rename("SAVI")
    # 11. PSRI (Plant Senescence Reflectance Index, Merzlyak 1999)
    #     PSRI = (R680 - R500) / R750  ->  (B4 - B2) / B6
    #     Antes: (B4-B3)/B7 — B3 en vez de B2 y B7 en vez de B6.
    psri = b4.subtract(b2).divide(b6.max(REFL_MIN)).rename("PSRI")
    # 12. KNDVI (kernelized NDVI, Camps-Valls 2021)
    #     Con el kernel gaussiano y sigma = 0.5*(NIR+RED), kNDVI se
    #     reduce a tanh(NDVI^2). El factor 2 que había no corresponde a
    #     ninguna formulación: comprimía el índice hacia 1.
    kndvi = ndvi.pow(2).tanh().rename("KNDVI")

    bandas = [ndvi, mari, ari, evi, evi2, ndwi,
              ndmi, chl, ndii, savi, psri, kndvi]
    # Recorte POR ÍNDICE (ver INDEX_CLAMP): un clamp único a [-1,1] saturaba
    # ARI, EVI, EVI2 y SAVI, que no están acotados a ese rango.
    bandas = [b.clamp(*INDEX_CLAMP[nm]) for b, nm in zip(bandas, INDEX_NAMES)]
    return ee.Image.cat(bandas)

def calcular_dimensiones_pixel(aoi, scale: int, coords: List[float]) -> Tuple[int, int]:
    import math
    bounds = aoi.bounds().coordinates().getInfo()[0]
    xs = [p[0] for p in bounds]
    ys = [p[1] for p in bounds]
    lat = coords[1]
    width_m = (max(xs) - min(xs)) * 111_320.0 * math.cos(math.radians(lat))
    height_m = (max(ys) - min(ys)) * 111_320.0
    w_px = int(np.ceil(width_m / scale))
    h_px = int(np.ceil(height_m / scale))
    if w_px % 2 == 1: w_px += 1
    if h_px % 2 == 1: h_px += 1
    return w_px, h_px

def descargar_indices(img: ee.Image, aoi, scale: int) -> Optional[np.ndarray]:
    for intento in range(1, MAX_RETRIES + 1):
        try:
            arr = geemap.ee_to_numpy(img, region=aoi, scale=scale)
            if arr is None or arr.size == 0:
                raise RuntimeError("Array vacío.")
            # arr ya viene como (H, W, 12)
            if arr.ndim != 3 or arr.shape[2] != 12:
                raise RuntimeError(f"Shape inesperado: {arr.shape}")
            return arr.astype(np.float32)
        except Exception as e:
            wait = (2 ** intento) + random.uniform(0, 1)
            print(f"   ↳ Intento {intento}/{MAX_RETRIES}: {e}. Reintento en {wait:.1f}s")
            time.sleep(wait)
    return None

def descargar_rgb(img: ee.Image, aoi, scale: int) -> Optional[np.ndarray]:
    for intento in range(1, MAX_RETRIES + 1):
        try:
            arr = geemap.ee_to_numpy(
                img.select(["B4", "B3", "B2"]), region=aoi, scale=scale
            )
            if arr is None or arr.size == 0:
                raise RuntimeError("Array vacío.")
            return arr
        except Exception as e:
            wait = (2 ** intento) + random.uniform(0, 1)
            time.sleep(wait)
    return None

def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if rgb.ndim == 2:
        return rgb
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

def align_ecc(template_gray: np.ndarray, target_gray: np.ndarray, max_iter: int = 5000, eps: float = 1e-9) -> Tuple[np.ndarray, bool]:
    if template_gray.shape != target_gray.shape:
        target_gray = cv2.resize(target_gray,
                                  (template_gray.shape[1], template_gray.shape[0]))
    # phaseCorrelate para inicialización rápida
    dx, dy = 0.0, 0.0
    try:
        shift, _ = cv2.phaseCorrelate(np.float64(template_gray), np.float64(target_gray))
        dx, dy = float(shift[0]), float(shift[1])
    except Exception:
        pass

    if abs(dx) < 0.5 and abs(dy) < 0.5:
        return np.eye(2, 3, dtype=np.float32), True

    warp = np.array([[1.0, 0.0, dx],
                     [0.0, 1.0, dy]], dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps)
    try:
        _, warp = cv2.findTransformECC(template_gray, target_gray, warp,
                                        cv2.MOTION_TRANSLATION, criteria)
        return warp, True
    except cv2.error:
        return warp, False

def aplicar_warp(img: np.ndarray, warp: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.warpAffine(img, warp, (img.shape[1], img.shape[0]),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)
    out = np.empty_like(img)
    for c in range(img.shape[2]):
        out[:, :, c] = cv2.warpAffine(img[:, :, c], warp,
                                       (img.shape[1], img.shape[0]),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REFLECT_101)
    return out

def descargar_y_alinear_12_indices() -> Optional[np.ndarray]:
    print(f"\n{'='*60}")
    print("DESCARGA DE 12 ÍNDICES ESPECTRALES — SENTINEL-2")
    print(f"{'='*60}")

    init_earth_engine()
    aoi = calcular_area_interes(COORDENADAS, TAMANO_AREA)

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(FECHA_INICIO, FECHA_FIN)
        # Nubosidad máxima de la escena. Subido de 2 a 3: admite más fechas
        # (más muestras de entrenamiento) a cambio de algo más de nube
        # residual, que el enmascarado QA60+SCL de abajo recorta igual.
        .filterMetadata("CLOUDY_PIXEL_PERCENTAGE", "less_than", 3)
        .map(mask_clouds_qa60_scl)
    )

    tile = obtener_tile_mas_frecuente(collection)
    if tile is None:
        print("No se encontró tile MGRS.")
        return None
    print(f"Tile más frecuente: {tile}")
    # sort() explícito: el orden de una ImageCollection no está garantizado,
    # y TODO el pipeline temporal (ventanas, predicción t+1) asume que el
    # eje 0 del stack es cronológico. Sin esto, el "forecasting" podría
    # estar entrenándose sobre frames desordenados.
    collection = (collection
                  .filterMetadata("MGRS_TILE", "equals", tile)
                  .sort("system:time_start"))
    count = collection.size().getInfo()
    if count == 0:
        return None

    # Fechas de adquisición, en el MISMO orden que toList(): sin ellas es
    # imposible saber si dos frames consecutivos del stack están separados
    # por 5 días o por 6 meses.
    fechas_millis = collection.aggregate_array("system:time_start").getInfo()
    if fechas_millis is None or len(fechas_millis) != count:
        print("   No se pudieron leer las fechas; stack sin ellas.")
        fechas_millis = None
    print(f"Total imágenes: {count}")

    w_px, h_px = calcular_dimensiones_pixel(aoi, RESOLUCION, COORDENADAS)
    print(f"Dimensiones estimadas: {w_px}×{h_px} px")

    images_list = collection.toList(count)

    ref_img = ee.Image(images_list.get(0))
    rgb_ref = descargar_rgb(ref_img, aoi, RESOLUCION)
    indices_ref = descargar_indices(compute_12_indices(ref_img), aoi, RESOLUCION)
    if rgb_ref is None or indices_ref is None:
        print("Error descargando referencia.")
        return None

    desired_shape = indices_ref.shape[:2]  # (H, W)
    print(f"Shape NDVI referencia: {indices_ref.shape}")

    rgb_ref_f = rgb_ref.astype(np.float32)
    ref_vmin = float(np.percentile(rgb_ref_f, 2))
    ref_vmax = float(np.percentile(rgb_ref_f, 98))
    rgb_ref_norm = np.clip(
        (rgb_ref_f - ref_vmin) / max(ref_vmax - ref_vmin, 1e-6) * 255.0,
        0, 255,
    ).astype(np.uint8)
    ref_gray = rgb_to_gray(rgb_ref_norm)

    print(f"\nDescargando y alineando {count} imágenes...")
    processed = [None] * count
    processed[0] = indices_ref
    _FALLOS_ECC: List[int] = []


    def procesar(index, img):
        try:
            rgb_arr = descargar_rgb(img, aoi, RESOLUCION)
            indices_arr = descargar_indices(compute_12_indices(img), aoi, RESOLUCION)
            if rgb_arr is None or indices_arr is None:
                return None, index
            rgb_norm = np.clip(
                (rgb_arr.astype(np.float32) - ref_vmin) /
                max(ref_vmax - ref_vmin, 1e-6) * 255.0,
                0, 255,
            ).astype(np.uint8)
            tgt_gray = rgb_to_gray(rgb_norm)
            if tgt_gray.shape != desired_shape:
                tgt_gray = cv2.resize(tgt_gray, (desired_shape[1], desired_shape[0]))
            warp, ok_ecc = align_ecc(ref_gray, tgt_gray)
            if not ok_ecc:
                # findTransformECC lanzo: se usa el warp de correlacion de
                # fase, que es peor. Antes este flag se descartaba con
                # "warp, _ =" y los frames mal alineados entraban al stack
                # sin dejar rastro.
                _FALLOS_ECC.append(index)
            if indices_arr.shape[:2] != desired_shape:
                indices_resized = np.empty((desired_shape[0], desired_shape[1], 12),
                                            dtype=np.float32)
                for c in range(12):
                    indices_resized[:, :, c] = cv2.resize(
                        indices_arr[:, :, c],
                        (desired_shape[1], desired_shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                indices_arr = indices_resized
            aligned_indices = aplicar_warp(indices_arr, warp)
            # Clamp por índice (mismos límites que en GEE); el warp bilineal
            # sólo interpola, no debería sacar valores de rango, pero se
            # asegura la coherencia con INDEX_CLAMP.
            aligned_indices = np.clip(aligned_indices,
                                      INDEX_CLAMP_LO, INDEX_CLAMP_HI)
            return aligned_indices, index
        except Exception as e:
            print(f"Error imagen {index}: {e}")
            return None, index

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as ex:
        futures = {
            ex.submit(procesar, i, ee.Image(images_list.get(i))): i
            for i in range(1, count)
        }
        for fut in as_completed(futures):
            arr, idx = fut.result()
            if arr is not None:
                processed[idx] = arr
                print(f"{idx}/{count}")

    # Se conserva el índice original de cada frame que sobrevivió, para
    # poder recortar el vector de fechas exactamente igual que el stack.
    if _FALLOS_ECC:
        print(f"\n  [!] ECC fallo en {len(_FALLOS_ECC)}/{count-1} frames "
              f"(se uso correlacion de fase como respaldo).")
        print(f"      indices: {sorted(_FALLOS_ECC)[:20]}"
              f"{' ...' if len(_FALLOS_ECC) > 20 else ''}")
        if len(_FALLOS_ECC) > 0.1 * max(count - 1, 1):
            print("      Mas del 10% de los frames sin alineacion fina: la "
                  "co-registracion no es fiable y las correlaciones entre "
                  "indices heredan ese error.")
    else:
        print("\n  ECC convergio en todos los frames.")

    valid_idx = [i for i, a in enumerate(processed) if a is not None]
    valid = [processed[i] for i in valid_idx]
    print(f"\nArrays válidos: {len(valid)}/{count}")
    if not valid:
        return None

    if fechas_millis is not None:
        fechas_validas = np.array([fechas_millis[i] for i in valid_idx],
                                  dtype=np.int64)
        gaps_d = np.diff(fechas_validas) / 86_400_000.0
        if gaps_d.size:
            print("")
            print(f"Muestreo temporal: {len(fechas_validas)} fechas entre "
                  f"{np.datetime64(int(fechas_validas[0]), 'ms')} y "
                  f"{np.datetime64(int(fechas_validas[-1]), 'ms')}")
            print(f"  Hueco entre imágenes consecutivas (días): "
                  f"mediana={np.median(gaps_d):.1f}  "
                  f"p90={np.percentile(gaps_d, 90):.1f}  "
                  f"máx={gaps_d.max():.1f}")
            print(f"  Huecos > {MAX_GAP_DAYS} d: "
                  f"{int((gaps_d > MAX_GAP_DAYS).sum())}/{gaps_d.size}")
    else:
        fechas_validas = None

    stack = np.stack(valid, axis=0).astype(np.float32)
    print(f"\nStack final: {stack.shape}, dtype={stack.dtype}, "
          f"range=[{stack.min():.3f}, {stack.max():.3f}]")

    # Reporte de saturación por índice: un canal con mucho porcentaje pegado
    # en su límite es un canal casi constante (fue exactamente lo que pasó
    # con ARI/EVI/EVI2/SAVI cuando faltaba dividir por S2_SCALE). Si aquí
    # aparece >5% en algún índice, ese índice NO aporta información y hay
    # que revisar su fórmula o ampliar su rango en INDEX_CLAMP.
    print("")
    print("Saturación por índice (fracción de píxeles en el límite):")
    hay_problema = False
    for _i, _nm in enumerate(INDEX_NAMES):
        _x = stack[..., _i]
        _lo, _hi = INDEX_CLAMP[_nm]
        _flo = float(np.mean(_x <= _lo + 1e-6))
        _fhi = float(np.mean(_x >= _hi - 1e-6))
        _marca = "  <-- REVISAR" if max(_flo, _fhi) > 0.05 else ""
        if _marca:
            hay_problema = True
        print(f"  {_nm:<12s} rango=[{_lo:>5.1f},{_hi:>4.1f}]  "
              f"en mínimo={_flo*100:5.2f}%  en máximo={_fhi*100:5.2f}%"
              f"  std={float(np.std(_x)):.4f}{_marca}")
    if hay_problema:
        print("  [!] Hay índices saturados >5%: revisar fórmula/escala "
              "antes de entrenar.")

    os.makedirs(DIR_INDICES, exist_ok=True)
    np.save(OUTPUT_NPY, stack)
    print(f"Arrays guardados en: {OUTPUT_NPY}")

    meta_kwargs = dict(
        index_names=np.array(INDEX_NAMES),
        coords=np.array(COORDENADAS),
        resolution=RESOLUCION,
        size_m=TAMANO_AREA,
        start_date=FECHA_INICIO,
        end_date=FECHA_FIN,
    )
    if fechas_validas is not None:
        # Épocas Unix en ms, una por frame del stack y en el mismo orden.
        meta_kwargs["dates_millis"] = fechas_validas
    np.savez(OUTPUT_METADATA, **meta_kwargs)
    print(f"Metadata: {OUTPUT_METADATA}")

    return stack

# ============================================================================
# ENTRENAMIENTO DEL CONVTRANSFORMER
# ============================================================================

def set_seed(seed: int) -> None:
    """Fija TODAS las fuentes de aleatoriedad para que una semilla sea reproducible.

    POR QUE EL RAZONAMIENTO ANTERIOR ERA INCORRECTO. El comentario que estaba
    aqui decia: "no necesitamos reproducibilidad bit-exacta entre semillas
    (buscamos que difieran), asi que determinism=True solo costaba velocidad".
    Eso confunde dos cosas distintas:

      - ENTRE semillas si queremos diferencias. Esa es la variabilidad que
        FASE 1 y el Paradigma 1 miden.
      - Para UNA semilla fija la corrida tiene que ser repetible, o no se
        puede atribuir ningun cambio a ninguna causa.

    Con cudnn.benchmark=True cuDNN elige el algoritmo de convolucion
    midiendolo en caliente, y la misma semilla puede tomar rutas distintas en
    corridas distintas. Efecto medido sobre cuatro corridas de la MISMA
    configuracion (resultados_pipeline{,(1),(2),(3)}.zip):

        rho_mean          0.351 · 0.522 · 0.539 · 0.386
        aristas robustas  2 · 0 · 6 · 1        <- el titular de la tesis
        ratio_H           0.9956 · 0.9930 · 0.9916 · 0.9953

    El numero de aristas robustas oscila entre 0 y 6 sin que cambie la
    configuracion. Mientras eso pase, NINGUN experimento es falsable: el
    efecto de cualquier cambio queda dentro de la deriva entre corridas.
    """
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if MODO_DETERMINISTA:
        # CUBLAS_WORKSPACE_CONFIG hace falta para que las GEMM de cuBLAS sean
        # deterministas; sin el, use_deterministic_algorithms lanza excepcion.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # warn_only=True: alguna op (p. ej. interpolate bicubic en backward)
        # puede no tener kernel determinista; se avisa en vez de reventar.
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True

def get_device() -> torch.device:
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        return torch.device("cuda")
    print("GPU no disponible, usando CPU")
    return torch.device("cpu")

device = get_device()

USE_MIXED_PRECISION = device.type == "cuda"
USE_AMP = device.type == "cuda"

if device.type == "cuda" and not MODO_DETERMINISTA:
    # TF32 acelera las GEMM usando 10 bits de mantisa en vez de 23. Solo se
    # activa con MODO_DETERMINISTA=False: reduce la precision de los matmul y
    # por tanto contribuye a que dos corridas de la misma semilla diverjan,
    # que es justo lo que este pipeline no puede permitirse.
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

def cargar_fechas_stack(n_esperado: int) -> Optional[np.ndarray]:
    """Fechas (época Unix en ms) alineadas con el eje 0 del stack, o None.

    Devuelve None —y el filtro de huecos queda desactivado— si el
    metadata.npz viene de una corrida anterior a este fix (sin
    'dates_millis') o si el número de fechas no coincide con el de frames.
    Preferible desactivar el filtro a emparejar fechas equivocadas.
    """
    if not os.path.exists(OUTPUT_METADATA):
        return None
    try:
        meta = np.load(OUTPUT_METADATA, allow_pickle=True)
        if "dates_millis" not in meta.files:
            print("  metadata.npz sin fechas (corrida previa al fix): "
                  "filtro de huecos temporales DESACTIVADO.")
            return None
        fechas = np.asarray(meta["dates_millis"], dtype=np.int64)
    except Exception as e:
        print(f"  No se pudieron leer las fechas ({e}): filtro desactivado.")
        return None
    if len(fechas) != n_esperado:
        print(f"  metadata tiene {len(fechas)} fechas y el stack "
              f"{n_esperado} frames: filtro DESACTIVADO "
              f"(re-ejecuta PARTE 1 para regenerarlo).")
        return None
    return fechas


def _make_sequences(arr: np.ndarray, seq_length: int,
                    dates_millis: Optional[np.ndarray] = None,
                    max_gap_days: int = MAX_GAP_DAYS,
                    etiqueta: str = ""):
    """Ventanas deslizantes de `seq_length` frames → frame siguiente.

    Si se pasan las fechas, descarta las ventanas que cruzan un hueco
    temporal mayor a `max_gap_days`: el modelo trata la ventana como
    equiespaciada, así que una ventana que salta de enero a junio le enseña
    una dinámica que no existe.
    """
    n_ventanas = max(len(arr) - seq_length, 0)
    usar_filtro = bool(dates_millis is not None and max_gap_days
                       and len(dates_millis) == len(arr))

    def _construir(filtrar: bool):
        X, Y, descartadas = [], [], 0
        for i in range(n_ventanas):
            if filtrar:
                ventana = dates_millis[i:i + seq_length + 1]
                if (np.diff(ventana).max() / 86_400_000.0) > max_gap_days:
                    descartadas += 1
                    continue
            X.append(arr[i:i + seq_length])
            Y.append(arr[i + seq_length])
        return X, Y, descartadas

    X, Y, descartadas = _construir(usar_filtro)
    if usar_filtro and not X and n_ventanas:
        # Filtrar todo deja el split vacío y revienta el DataLoader; eso es
        # peor que no filtrar. Se avisa y se reconstruye sin filtro.
        print(f"  {etiqueta}: TODAS las {n_ventanas} ventanas cruzan un "
              f"hueco > {max_gap_days} d; filtro ignorado "
              f"(sube MAX_GAP_DAYS).")
        X, Y, descartadas = _construir(False)
    elif descartadas:
        print(f"  {etiqueta}: {descartadas}/{n_ventanas} ventanas "
              f"descartadas por hueco > {max_gap_days} d "
              f"({len(X)} utilizables).")
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def reportar_baselines_enmascarado(frames_val: np.ndarray, mse_modelo: float,
                                   scaler=None) -> Dict[str, float]:
    """Baselines de la tarea enmascarada: reconstruir el índice j sin verlo.

    Los baselines correctos aquí NO son la persistencia (no hay tiempo en la
    tarea) sino los predictores estadísticos que usan los otros 11 índices:

      constante 0        — predice la media del canal (que es 0 tras estandarizar)
      regresión lineal   — mínimos cuadrados de cada índice sobre los otros 11,
                           ajustada en train y evaluada en val. ESTE es el
                           baseline decisivo: es exactamente lo que mide la
                           correlación de Pearson, y es el mismo objeto con el
                           que el Paradigma 5 compara el grafo de atención.
                           Si el modelo no le gana, su atención no aporta nada
                           sobre la estadística lineal entre índices.
    """
    F = np.asarray(frames_val, dtype=np.float64)          # (N, H, W, 12)
    n_idx = F.shape[-1]
    X = F.reshape(-1, n_idx)

    mse_cero = float(np.mean(X ** 2))

    # regresión lineal de cada canal sobre los otros 11, ajustada en el propio
    # split de validación con validación cruzada implícita muy simple: se usa
    # la solución de mínimos cuadrados del propio split, así que es una cota
    # OPTIMISTA para el baseline (le damos ventaja al baseline, no al modelo).
    # El baseline tiene que ver los MISMOS canales que el modelo. Con
    # MASCARA_K = k el modelo reconstruye j desde 12-k canales, no desde 11: dar
    # los 11 a la regresión la favorece en k-1 canales y volvería el "el modelo
    # no supera al lineal" un artefacto de la comparación, no un hallazgo.
    # Se promedian R sorteos de los k-1 distractores para no depender de uno.
    k_mask = max(1, min(int(MASCARA_K), n_idx))
    rng_bl = np.random.default_rng(SEMILLA_NULO)
    R = 1 if k_mask == 1 else 8
    errs = []
    for j in range(n_idx):
        otros_todos = [c for c in range(n_idx) if c != j]
        e_j = []
        for _ in range(R):
            if k_mask == 1:
                visibles = otros_todos
            else:
                ocultos = rng_bl.choice(len(otros_todos), k_mask - 1,
                                        replace=False)
                fuera = {otros_todos[int(t)] for t in ocultos}
                visibles = [c for c in otros_todos if c not in fuera]
            A = np.concatenate([X[:, visibles], np.ones((len(X), 1))], axis=1)
            coef, *_ = np.linalg.lstsq(A, X[:, j], rcond=None)
            e_j.append(float(np.mean((A @ coef - X[:, j]) ** 2)))
        errs.append(float(np.mean(e_j)))
    mse_lineal = float(np.mean(errs))

    sep = "  " + "-" * 62
    print("")
    print(sep)
    print(f"  BASELINES — TAREA ENMASCARADA   ({len(F)} frames × {n_idx} "
          f"índices · k = {max(1, min(int(MASCARA_K), n_idx))} ocultos)")
    print(sep)
    print(f"  {'':<34}{'MSE':>12}")
    print(f"  {'Modelo entrenado':<34}{float(mse_modelo):>12.4f}")
    etq_bl = (f"Regresión lineal ({n_idx - k_mask} vis.)" if k_mask > 1
              else f"Regresión lineal (otros {n_idx - 1})")
    print(f"  {etq_bl:<34}{mse_lineal:>12.4f}")
    print(f"  {'Constante 0 (media del canal)':<34}{mse_cero:>12.4f}")
    print("")
    if float(mse_modelo) < mse_lineal:
        print(f"  [ok] El modelo supera a la regresión lineal en "
              f"{100*(1-float(mse_modelo)/max(mse_lineal,1e-12)):.1f}%.")
        print(f"       Hay dependencia entre índices que la estadística lineal")
        print(f"       no captura, y la atención tiene algo que explicar.")
    else:
        razon = float(mse_modelo) / max(mse_lineal, 1e-12)
        print(f"  [!] El modelo NO supera a la regresión lineal ({razon:.2f}x peor).")
        print(f"      Toda la dependencia entre índices es lineal: la atención no")
        print(f"      puede aportar sobre Pearson.")
    print(f"\n  Por índice (MSE del baseline lineal, el más difícil de batir):")
    for j in range(n_idx):
        print(f"     {INDEX_NAMES[j]:<14}{errs[j]:>10.4f}")
    print(sep)
    print("")
    return dict(modelo=float(mse_modelo), lineal=mse_lineal, cero=mse_cero,
                k_mascara=k_mask, n_visibles=n_idx - k_mask,
                lineal_por_indice={INDEX_NAMES[j]: errs[j] for j in range(n_idx)})


def reportar_baselines(X_val: np.ndarray, Y_val: np.ndarray,
                       mse_modelo: float, scaler=None,
                       etiqueta: str = "") -> Dict[str, float]:
    """Compara el MSE del modelo contra baselines triviales.

    Un MSE "bajo" no significa nada por sí solo: está en unidades de
    StandardScaler, no de índice crudo, y no es comparable con el 0.007724
    de la presentación (que está en NDVI crudo). Además, si copiar el
    último frame de entrada (persistencia) ya iguala al modelo, el modelo
    no aprendió la dinámica y su mapa de atención no es interpretable como
    "relación entre índices": mide ruido de inicialización.
    """
    if Y_val is None or len(Y_val) == 0:
        print("  (sin muestras de validación: no hay baselines que calcular)")
        return {}

    Yd = Y_val.astype(np.float64)
    escala = None
    if scaler is not None and hasattr(scaler, "scale_"):
        s = np.asarray(scaler.scale_, dtype=np.float64)
        if s.size in (1, Yd.shape[-1]):
            escala = s

    def _mse(pred):
        e = pred.astype(np.float64) - Yd
        crudo = float(np.mean((e * escala) ** 2)) if escala is not None else None
        return float(np.mean(e ** 2)), crudo

    filas = [
        ("Persistencia (frame t)", _mse(X_val[:, -1])),
        ("Media de la ventana",    _mse(X_val.mean(axis=1))),
        ("Constante 0 (media glob.)", _mse(np.zeros_like(Yd))),
    ]
    mse_persist = filas[0][1][0]
    crudo_modelo = (float(mse_modelo) * float(np.mean(escala ** 2))
                    if escala is not None else None)

    sep = "  " + "-" * 62
    print("")
    print(sep)
    titulo = "BASELINES DE REFERENCIA"
    if etiqueta:
        titulo += " — " + etiqueta
    print(f"  {titulo}   ({len(Yd)} muestras de validación)")
    print(sep)
    print(f"  {'':<30}{'MSE escalado':>16}{'MSE crudo':>16}")
    c_mod = f"~{crudo_modelo:.6f}" if crudo_modelo is not None else "-"
    print(f"  {'Modelo entrenado':<30}{float(mse_modelo):>16.4f}{c_mod:>16}")
    for nombre, (esc, cru) in filas:
        c = f"{cru:.6f}" if cru is not None else "-"
        print(f"  {nombre:<30}{esc:>16.4f}{c:>16}")

    print("")
    if float(mse_modelo) >= mse_persist:
        razon = float(mse_modelo) / max(mse_persist, 1e-12)
        print(f"  [!] EL MODELO NO SUPERA A LA PERSISTENCIA ({razon:.2f}x peor).")
        print(f"      No aprendió la dinámica temporal: la atención extraída")
        print(f"      de este modelo NO es interpretable como relación entre")
        print(f"      índices. Revisar datos/arquitectura antes de seguir.")
    else:
        mejora = 100.0 * (1.0 - float(mse_modelo) / mse_persist)
        print(f"  [ok] El modelo supera a la persistencia en {mejora:.1f}%.")
    if crudo_modelo is not None:
        print(f"  (MSE crudo del modelo es aproximado; referencia slide 14 "
              f"= 0.007724 en NDVI crudo)")
    print(sep)
    print("")

    return {
        "modelo": float(mse_modelo),
        "persistencia": mse_persist,
        "media_ventana": filas[1][1][0],
        "constante_0": filas[2][1][0],
        "modelo_crudo_aprox": crudo_modelo,
        "persistencia_crudo": filas[0][1][1],
    }

def trocear_en_parches(X: np.ndarray, P: int) -> np.ndarray:
    """(T, H, W, C) -> (T*nh*nw, P, P, C) en rejilla NO solapada.

    Los pixeles sobrantes por el borde (H % P, W % P) se descartan: solapar
    parches meteria el mismo pixel en varias muestras y romperia la
    independencia entre ellas, que es justo lo que hace falta aqui.
    """
    T, H, W, C = X.shape
    nh, nw = H // P, W // P
    if nh == 0 or nw == 0:
        return X
    X = X[:, :nh * P, :nw * P, :]
    X = X.reshape(T, nh, P, nw, P, C).transpose(0, 1, 3, 2, 4, 5)
    return np.ascontiguousarray(X.reshape(-1, P, P, C))


def process_indices_data(stack: np.ndarray, seq_length: int = SEQ_LENGTH,
                         val_ratio: float = VAL_RATIOVAL,
                         dates_millis: Optional[np.ndarray] = None):
    n = len(stack)
    n_train = int((1.0 - val_ratio) * n)
    train_data = stack[:n_train]
    val_data = stack[n_train:]

    scaler = StandardScaler()
    train_resh = train_data.reshape(-1, train_data.shape[-1])
    scaler.fit(train_resh)
    train_scaled = scaler.transform(train_resh).reshape(train_data.shape).astype(np.float32)
    val_scaled = scaler.transform(
        val_data.reshape(-1, val_data.shape[-1])
    ).reshape(val_data.shape).astype(np.float32)

    # Las fechas se parten con el MISMO corte que los datos.
    f_tr = dates_millis[:n_train] if dates_millis is not None else None
    f_va = dates_millis[n_train:] if dates_millis is not None else None

    img_size = stack.shape[1:3]  # (H, W)

    if TAREA == "enmascarado":
        # No hay ventanas temporales: cada frame es independiente y de él salen
        # 12 muestras (una por índice oculto). Se devuelven los frames
        # estandarizados tal cual; IndicesEnmascaradosDataset hace el resto.
        if TOKEN_PARCHE and TOKEN_PARCHE > 0:
            n_fr_tr, n_fr_va = len(train_scaled), len(val_scaled)
            train_scaled = trocear_en_parches(train_scaled, TOKEN_PARCHE)
            val_scaled = trocear_en_parches(val_scaled, TOKEN_PARCHE)
            img_size = (TOKEN_PARCHE, TOKEN_PARCHE)
            print(f"  TAREA=enmascarado · TOKEN_PARCHE={TOKEN_PARCHE}")
            print(f"    train: {n_fr_tr} frames -> {len(train_scaled)} parches "
                  f"de {TOKEN_PARCHE}x{TOKEN_PARCHE}")
            print(f"    val:   {n_fr_va} frames -> {len(val_scaled)} parches")
        else:
            print(f"  TAREA=enmascarado · {len(train_scaled)} frames train × "
                  f"{NUM_INDICES} índices = {len(train_scaled)*NUM_INDICES} muestras")
            print(f"                     · {len(val_scaled)} frames val × "
                  f"{NUM_INDICES} = {len(val_scaled)*NUM_INDICES} muestras")
        return (train_scaled, None), (val_scaled, None), img_size, scaler

    X_train, Y_train = _make_sequences(train_scaled, seq_length, f_tr,
                                       etiqueta="train")
    X_val, Y_val = _make_sequences(val_scaled, seq_length, f_va,
                                   etiqueta="val")
    return (X_train, Y_train), (X_val, Y_val), img_size, scaler

class IndicesDataset(Dataset):
    def __init__(self, X, Y):
        self.X = np.ascontiguousarray(np.transpose(X, (0,4,1,2,3)))
        self.Y = np.ascontiguousarray(np.transpose(Y, (0,3,1,2)))
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float(), torch.from_numpy(self.Y[idx]).float()


_FAM_CACHE: Dict[str, torch.Tensor] = {}


def familia_de_indice() -> List[int]:
    """Índice de familia espectral (0..len-1) de cada uno de los NUM_INDICES.

    Sale de INDEX_FAMILIES, que a su vez sale de las fórmulas de
    compute_12_indices: qué bandas comparte cada índice. Un índice que no
    aparezca en ninguna familia queda en una familia propia.
    """
    fam_de = [-1] * NUM_INDICES
    for f, (_nombre, miembros) in enumerate(INDEX_FAMILIES.items()):
        for nm in miembros:
            if nm in INDEX_NAMES:
                fam_de[INDEX_NAMES.index(nm)] = f
    siguiente = len(INDEX_FAMILIES)
    for i, f in enumerate(fam_de):
        if f < 0:
            fam_de[i] = siguiente
            siguiente += 1
    return fam_de


def _mascara_familias(device) -> torch.Tensor:
    """(N, N) booleano: fila v oculta todos los índices de la familia de v."""
    clave = str(device)
    if clave in _FAM_CACHE:
        return _FAM_CACHE[clave]
    fam_de = torch.tensor(familia_de_indice(), device=device)
    fam = fam_de.unsqueeze(0) == fam_de.unsqueeze(1)     # (N, N)
    fam = fam | torch.eye(NUM_INDICES, dtype=torch.bool, device=device)
    _FAM_CACHE[clave] = fam
    return fam


def reportar_mascara() -> None:
    """Deja constancia de qué oculta la máscara y cuánto ve el modelo."""
    fam_de = familia_de_indice()
    nombres = list(INDEX_FAMILIES.keys())
    print(f"\n  MÁSCARA: modo={MASCARA_MODO!r}  k={MASCARA_K}  "
          f"lectura k={MASCARA_K_LECTURA}")
    if MASCARA_MODO == "familia":
        for f, nm in enumerate(nombres):
            miembros = [INDEX_NAMES[i] for i, g in enumerate(fam_de) if g == f]
            if miembros:
                print(f"    {nm:<14} oculta {len(miembros)}: "
                      f"{', '.join(miembros)}  "
                      f"(visibles {NUM_INDICES - len(miembros)})")
    else:
        n_extra = MASCARA_K - 1
        for f, nm in enumerate(nombres):
            tam = sum(1 for g in fam_de if g == f)
            hermanos = tam - 1
            if hermanos <= 0:
                continue
            # P(los n_extra distractores sean exactamente los hermanos)
            if n_extra >= hermanos:
                from math import comb
                p_todos = comb(NUM_INDICES - 1 - hermanos, n_extra - hermanos) \
                    / comb(NUM_INDICES - 1, n_extra)
            else:
                p_todos = 0.0
            esperados = n_extra * hermanos / (NUM_INDICES - 1)
            print(f"    {nm:<14} {hermanos} hermanos · "
                  f"P(todos ocultos)={p_todos:.4f} · "
                  f"esperados ocultos={esperados:.2f}")
        print("    Si P(todos ocultos) es baja, la tarea NO exige enrutar")
        print("    fuera de la familia: queda un colineal visible casi siempre.")


def mascara_multiple(B: int, N: int, k: int, device,
                    generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Patrón de ocultación para forward_enmascarado.

    Devuelve (B, N, N) booleano: `m[b, v, t]` es True si en la variante v de la
    muestra b el token t va oculto. La diagonal siempre está a True —la variante
    v oculta el índice v, que es el que hay que reconstruir— y además se
    sortean k-1 distractores distintos por (b, v).

    Con k=1 devuelve exactamente la identidad, que es el comportamiento
    histórico, así que la lectura de la matriz de atención no cambia.
    """
    k = max(1, min(int(k), N))
    ident = torch.eye(N, dtype=torch.bool, device=device)
    m = ident.unsqueeze(0).expand(B, N, N).clone()

    if MASCARA_MODO == "familia" and N == NUM_INDICES:
        # La variante v oculta la FAMILIA ESPECTRAL de v, no k-1 distractores
        # al azar. Determinista: no depende del sorteo, así que la máscara de
        # validación es idéntica en cada época sin necesitar generador.
        fam = _mascara_familias(device)          # (N, N) booleano, cacheado
        return fam.unsqueeze(0).expand(B, N, N).clone()

    if k == 1:
        return m
    # Se sortean k-1 columnas por fila evitando la diagonal: se puntúa cada
    # token al azar, se pone -inf en la diagonal (ya ocultada) y se toman los
    # k-1 mayores. Es un muestreo sin reemplazo vectorizado.
    puntajes = torch.rand(B, N, N, device=device, generator=generator)
    puntajes = puntajes.masked_fill(ident.unsqueeze(0), -1.0)
    extra = puntajes.topk(k - 1, dim=-1).indices                # (B, N, k-1)
    m.scatter_(-1, extra, True)
    return m


class FramesDataset(Dataset):
    """Devuelve el frame entero (12, H, W). Una muestra por fecha.

    Se usa con ConvTransformer.forward_enmascarado, que genera las 12 variantes
    enmascaradas internamente y en una sola pasada. Como la salida en la
    posición j ES la reconstrucción del índice j, la pérdida es simplemente
    MSE(salida, frame) sobre los 12 canales: no hace falta máscara.
    """

    def __init__(self, frames: np.ndarray):
        self.frames = np.ascontiguousarray(np.transpose(frames, (0, 3, 1, 2)))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        f = torch.from_numpy(self.frames[idx]).float()
        return f, f


class IndicesEnmascaradosDataset(Dataset):
    """Tarea enmascarada: oculta el índice j y pide reconstruirlo (cambio 2).

    De cada frame salen NUM_INDICES muestras, una por índice enmascarado. La
    entrada es el frame con el canal j puesto a cero; el objetivo es el frame
    completo, pero la pérdida se calcula SOLO en el canal j (de eso se encarga
    la máscara que devuelve el tercer elemento).

    Formas idénticas a IndicesDataset, así que el ConvTransformer no cambia:
      x    (12, seq, H, W)  — canal j en cero
      y    (12, H, W)       — frame completo
      mask (12,)            — one-hot en j
    """

    def __init__(self, frames: np.ndarray, n_indices: int = None):
        # frames: (N, H, W, 12) ya estandarizado
        self.frames = np.ascontiguousarray(np.transpose(frames, (0, 3, 1, 2)))
        self.n_idx = int(n_indices or self.frames.shape[1])
        self.n_frames = len(self.frames)

    def __len__(self):
        return self.n_frames * self.n_idx

    def __getitem__(self, idx):
        f, j = divmod(idx, self.n_idx)
        y = self.frames[f]                        # (12, H, W)
        x = y.copy()
        x[j] = 0.0                                # se oculta el índice j
        mask = np.zeros(self.n_idx, dtype=np.float32)
        mask[j] = 1.0
        return (torch.from_numpy(x).float().unsqueeze(1),   # (12, 1, H, W)
                torch.from_numpy(y).float(),
                torch.from_numpy(mask))


def mse_enmascarado(pred: torch.Tensor, target: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
    """MSE calculado solo en los canales marcados por `mask`.

    pred/target: (B, 12, H, W) · mask: (B, 12)
    """
    err2 = (pred - target) ** 2                       # (B, 12, H, W)
    m = mask.unsqueeze(-1).unsqueeze(-1)              # (B, 12, 1, 1)
    return (err2 * m).sum() / m.expand_as(err2).sum().clamp_min(1.0)


class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int):
        super().__init__()
        # Ancho intermedio proporcional a embed_dim (mínimo 32), no fijo en 32.
        # Con embed_dim=32 (valor viejo) esto era invisible porque 32==32; con
        # embed_dim=256 un intermedio fijo en 32 crearía un cuello de botella
        # 32→256 en una sola capa en vez de un ensanchamiento progresivo.
        mid = max(32, embed_dim // 2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

def normalizar_atencion(logits: torch.Tensor, modo: str = None) -> torch.Tensor:
    """Convierte los logits QKᵀ/√d en pesos de atención.

    POR QUÉ ESTO ES CONFIGURABLE. El softmax fuerza que cada fila sume 1. Si el
    token i no necesita mirar a ninguno de los otros, no puede "no atender":
    tiene que volcar esa masa obligatoria en alguna columna. Eso es un attention
    sink, y es la explicación mecánica de lo que medimos en este modelo: la
    matriz resulta casi de rango 1 (A ≈ d·I + 1·cᵀ, R² = 0.87 en la capa 1 y
    0.99 en la capa 2), o sea todas las filas iguales, con dos o tres columnas
    absorbiendo la masa y el índice que pregunta siendo irrelevante. Los
    sumideros incluso se mudaban (NDWI/PSRI → EVI → EVI/SAVI) al cambiar la
    arquitectura, lo que confirma que eran un artefacto y no una relación
    espectral.

    Las alternativas quitan esa obligación, así que una fila puede sumar mucho
    menos que 1 cuando el modelo decide no mezclar:

      "softmax"  — comportamiento original, fila estocástica por construcción.
      "sigmoid"  — σ(logit − log N) elementwise, sin normalizar la fila
                   (Ramapuram et al. 2024, arXiv:2409.04431). El sesgo −log N
                   deja la suma de fila en ~1 al inicio para no cambiar la
                   escala de partida, pero el modelo es libre de bajarla.
      "softpick" — softmax rectificado: relu(eˣ−1) / Σ|eˣ−1|
                   (arXiv:2504.20966). Mantiene el reparto relativo del softmax
                   pero permite que la fila entera valga cero.

    Ver también arXiv:2505.06708 (gated attention, sink-free) y
    arXiv:2504.02732 (ICLR 2025, cuándo y por qué emergen los sinks).

    AVISO: con sigmoid/softpick las filas ya NO suman 1. El resto del pipeline
    lo tolera (attention_rollout renormaliza al final y discretizar normaliza
    por fila antes de cortar), pero la suma de fila pasa a ser un dato
    interesante por sí misma: una fila que suma ~0 significa "este índice
    decidió no mirar a ninguno", que es información real y no un fallo.
    """
    modo = (modo or ATENCION_NORMALIZACION).lower()
    n = logits.size(-1)
    if modo == "softmax":
        return torch.softmax(logits, dim=-1)
    if modo == "sigmoid":
        return torch.sigmoid(logits - float(np.log(n)))
    if modo == "softpick":
        # OJO: aquí NO se puede restar el máximo como en el softmax estable.
        # softpick(x)_i = relu(e^x_i − 1) / Σ_j |e^x_j − 1|, y la esparsidad
        # viene del SIGNO del logit: los negativos dan e^x−1 < 0 y el relu los
        # manda a cero. Si se resta el máximo, todos los logits quedan ≤ 0, el
        # numerador se anula entero y la fila suma 0. Para la estabilidad se
        # acota el logit por arriba en su lugar.
        e = torch.exp(logits.clamp(max=20.0)) - 1.0
        num = torch.relu(e)
        den = e.abs().sum(dim=-1, keepdim=True) + 1e-6
        return num / den
    raise ValueError(f"ATENCION_NORMALIZACION no reconocida: {modo!r}")


class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """Capa post-norm que además expone los pesos de atención por cabeza.

    Calcula la atención a mano en vez de delegar en nn.MultiheadAttention,
    porque esa implementación aplica softmax internamente y no permite
    cambiarlo. Los pesos (Q, K, V, out_proj) siguen siendo los de self_attn, así
    que el número de parámetros no cambia.
    """

    def forward(self, src, src_mask=None, src_key_padding_mask=None,
                is_causal=False, **kwargs):
        sa = self.self_attn
        B, N, d = src.shape
        nh = sa.num_heads
        hd = d // nh

        q, k, v = F.linear(src, sa.in_proj_weight, sa.in_proj_bias).chunk(3, dim=-1)
        qh = q.reshape(B, N, nh, hd).permute(0, 2, 1, 3)
        kh = k.reshape(B, N, nh, hd).permute(0, 2, 1, 3)
        vh = v.reshape(B, N, nh, hd).permute(0, 2, 1, 3)

        logits = (qh @ kh.transpose(-2, -1)) / (hd ** 0.5)
        if src_mask is not None:
            logits = logits + src_mask
        attn_weights = normalizar_atencion(logits)          # (B, nh, N, N)

        ctx = (attn_weights @ vh).permute(0, 2, 1, 3).reshape(B, N, d)
        src2 = sa.out_proj(ctx)

        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        return src, attn_weights

class AttentionTransformerEncoder(nn.TransformerEncoder):
    def forward(self, src, *args, **kwargs):
        output = src
        all_attn = []
        for layer in self.layers:
            output, attn = layer(output, *args, **kwargs)
            all_attn.append(attn)
        return output, all_attn

class ConvTransformer(nn.Module):
    def __init__(self, num_indices: int = NUM_INDICES, seq_length: int = SEQ_LENGTH,
                 img_size: Tuple[int, int] = (50, 52),
                 embed_dim: int = 256, d_model: int = 128,
                 num_heads: int = 4, num_layers: int = 2,
                 dropout: float = 0.1,
                 n_registros: Optional[int] = None,
                 dim_feedforward: Optional[int] = None):
        # embed_dim=256 (antes 32, sin verificar): la presentación original
        # (slide 14) reporta el resultado "bueno" del compañero con
        # "embed_dim=256, 4 cabezales" — 8x más ancho que lo que veníamos
        # usando. Con 32 canales el encoder conv por índice puede no tener
        # capacidad para captar diferencias NO lineales entre índices (ej.
        # cómo satura NDVI vs EVI, que fue diseñado justo para no saturar
        # igual) — la colinealidad lineal (r=0.888 medida en los datos
        # crudos) no descarta que un encoder con más capacidad sí distinga
        # los índices por esas diferencias no lineales.
        super().__init__()
        self.num_tokens = num_indices       # 12 índices como tokens
        self.in_channels = seq_length       # seq_length como canales
        self.img_size = tuple(img_size)

        self.encoder = ConvEncoder(self.in_channels, embed_dim)
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, *self.img_size)
            dummy = self.encoder(dummy)
        H_enc, W_enc = dummy.shape[2], dummy.shape[3]
        self.enc_shape = (embed_dim, H_enc, W_enc)
        self.flatten_dim = embed_dim * H_enc * W_enc

        self.frame_proj = nn.Linear(self.flatten_dim, d_model)
        # LayerNorm sobre el token para que el pos_embed no quede sepultado por
        # una norma de contenido de ~108 (ver USAR_TOKEN_NORM en la SECCIÓN 0).
        self.token_norm = nn.LayerNorm(d_model) if USAR_TOKEN_NORM else nn.Identity()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_indices, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=POS_EMBED_STD)

        # Register tokens: absorben el sumidero para que el bloque token-token
        # quede interpretable. Con n_registros=0 el modelo es identico al de
        # antes, byte a byte en el camino de datos.
        self.n_registros = int(max(0, n_registros if n_registros is not None
                                   else N_REGISTROS))
        if self.n_registros > 0:
            self.registros = nn.Parameter(
                torch.zeros(1, self.n_registros, d_model))
            nn.init.trunc_normal_(self.registros, std=POS_EMBED_STD)
        else:
            self.registros = None

        encoder_layer = CustomTransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=(dim_feedforward if dim_feedforward is not None
                             else DIM_FEEDFORWARD),
            batch_first=True, dropout=dropout,
        )
        self.transformer = AttentionTransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pred_linear = nn.Linear(d_model, self.flatten_dim)
        # SIN nn.Tanh() al final: la salida está acotada a [-1,1] pero los
        # targets pasan por StandardScaler, y el 32% de ellos cae FUERA de ese
        # rango (NDVI estandarizado llega a -3.03 / +2.41). Con Tanh el modelo
        # es incapaz por construcción de acertar un tercio de los píxeles
        # (piso de MSE ~0.069 con un predictor perfecto) y además el gradiente
        # muere en la saturación. En el trabajo original el Tanh era coherente
        # porque el target era NDVI crudo en [0,1]; con estandarización, no.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
        )

        self.usar_residual = USAR_RESIDUAL

        # Captura de atención
        self.capture_attention = False
        self._attention_cache: List[List[np.ndarray]] = []
        # Masa de atención que se fue a los registros, por capa. Es el sumidero
        # medido: con registros deberia subir aqui y BAJAR en el bloque
        # token-token, que es lo que el analisis lee.
        self._masa_registros: List[float] = []
        self.ultima_masa_registros: Optional[float] = None

    def _con_registros(self, tok: torch.Tensor) -> torch.Tensor:
        """Concatena los R registros al final de la secuencia de tokens."""
        if self.n_registros == 0:
            return tok
        B = tok.size(0)
        return torch.cat([tok, self.registros.expand(B, -1, -1).to(tok.dtype)],
                         dim=1)

    def _sin_registros(self, salida: torch.Tensor) -> torch.Tensor:
        """Descarta los R registros: solo los tokens de indice van al decoder."""
        if self.n_registros == 0:
            return salida
        return salida[:, :self.num_tokens]

    def _recortar_atencion(self, attn: torch.Tensor) -> np.ndarray:
        """(B, heads, N+R, N+R) -> bloque token-token (B, heads, N, N) normalizado.

        La masa que los tokens dieron a los registros se retira y se registra
        aparte: el grafo se define sobre relaciones entre TOKENS, y Top-P
        necesita una fila que sume 1 para que el nucleo este bien definido.
        Sin renormalizar, un P=0.5 sobre una fila que suma 0.6 seleccionaria
        casi toda la fila y el operador dejaria de ser comparable entre
        modelos.
        """
        a = attn.detach()
        N = self.num_tokens
        if self.n_registros > 0:
            fuga = a[..., :N, N:].sum(dim=-1)          # (B, heads, N)
            self._masa_registros.append(float(fuga.mean().cpu()))
            a = a[..., :N, :N]
        s = a.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return (a / s).cpu().numpy()

    def masa_registros(self) -> Optional[float]:
        """Fracción media de atención que los tokens dieron a los registros.

        Es el sumidero medido: con registros deberia subir aqui y BAJAR en
        el bloque token-token, que es lo que el analisis lee. Sirve
        despues de drenar la cache con get_attention_weights().
        """
        if self._masa_registros:
            return float(np.mean(self._masa_registros))
        return getattr(self, "ultima_masa_registros", None)

    def forward(self, x):
        B = x.size(0)
        # Frame t, que es lo que se suma al final si hay residual.
        entrada = x[:, :, -1]                           # (B, 12, H, W)
        # Cada índice se procesa por separado: (B*12, seq, H, W) → ConvEncoder
        x = x.reshape(B * self.num_tokens, self.in_channels,
                      self.img_size[0], self.img_size[1])
        feat = self.encoder(x)                          # (B*12, E, H', W')
        feat = feat.reshape(B, self.num_tokens, -1)     # (B, 12, E*H'*W')
        feat = self.frame_proj(feat)                    # (B, 12, d_model)
        feat = self.token_norm(feat)                    # escala ~1 por dimensión
        feat = feat + self.pos_embed
        feat = self._con_registros(feat)                # (B, 12+R, d_model)

        trans_out, all_attn = self.transformer(feat)

        if self.capture_attention and not self.training:
            # Se guarda el bloque token-token (B, num_heads, 12, 12): los
            # registros no son indices y no entran en la hipotesis relacional.
            self._attention_cache.append(
                [self._recortar_atencion(a) for a in all_attn]
            )

        trans_out = self._sin_registros(trans_out)      # (B, 12, d_model)
        latent = self.pred_linear(trans_out)            # (B, 12, flatten_dim)
        latent = latent.reshape(B * self.num_tokens, *self.enc_shape)
        out = self.decoder(latent)                      # (B*12, 1, H, W)
        out = F.interpolate(out, size=self.img_size,
                            mode="bicubic", align_corners=False)
        out = out.reshape(B, self.num_tokens, *self.img_size)
        # y(t+1) = x(t) + delta: el modelo aprende el CAMBIO, no la imagen.
        return entrada + out if self.usar_residual else out

    def forward_enmascarado(self, frames: torch.Tensor,
                           mascara: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Las 12 variantes enmascaradas de cada frame, en UNA pasada.

        frames: (B, 12, H, W)  → salida (B, 12, H, W), donde la posición j es la
        reconstrucción del índice j habiéndolo ocultado.

        `mascara` (B, 12, 12) booleano dice qué tokens van ocultos en cada
        variante; None equivale a la identidad, o sea ocultar sólo el índice que
        se reconstruye (k=1). Con k>1 se ocultan además k-1 distractores, lo que
        impide que el modelo se apoye en un vecino colineal fijo. Ver MASCARA_K.
        La optimización sigue siendo exacta: un token oculto es siempre la misma
        imagen de ceros, así que su embedding se calcula una vez y se reparte.

        POR QUÉ EXISTE. La versión ingenua genera 12 muestras por frame y cada
        una vuelve a pasar los 12 canales por el ConvEncoder: 144 pasadas de
        encoder por frame, cuando 11 de los 12 tokens son IDÉNTICOS entre
        variantes y el token oculto es siempre la misma imagen de ceros. Aquí se
        codifica cada canal una sola vez (12 pasadas) más el canal en cero (1),
        y las 12 variantes se arman sustituyendo. Lo mismo en el decoder: solo
        se decodifica el token que hace falta, el j de la variante j, en vez de
        los 144. Encoder y decoder bajan de 144·F a 12·F pasadas.

        Es una optimización EXACTA, no una aproximación: el encoder se aplica
        token a token de forma independiente, y el LayerNorm del token también,
        así que sustituir antes o después da el mismo resultado. La equivalencia
        con el camino lento se comprueba en verificar_fixes.py.
        """
        B, N, H, W = frames.shape
        assert N == self.num_tokens, f"esperaba {self.num_tokens} canales, no {N}"

        # 1) encoder una vez por canal + una vez para el canal oculto (ceros)
        e = self.encoder(frames.reshape(B * N, 1, H, W)).reshape(B, N, -1)
        e_cero = self.encoder(frames.new_zeros(1, 1, H, W)).reshape(1, -1)

        # 2) proyección y normalización ANTES de expandir: 43264 → 128, así la
        #    expansión a las 12 variantes cuesta 128 floats por token y no 43264
        p = self.token_norm(self.frame_proj(e))                 # (B, N, d)
        p_cero = self.token_norm(self.frame_proj(e_cero))       # (1, d)

        idx = torch.arange(N, device=frames.device)
        tok = p.unsqueeze(1).expand(B, N, N, p.size(-1)).clone()  # (B, variante, token, d)
        if mascara is None:
            tok[:, idx, idx, :] = p_cero.to(tok.dtype)
        else:
            assert mascara.shape == (B, N, N), \
                f"mascara {tuple(mascara.shape)}, esperaba {(B, N, N)}"
            # La diagonal tiene que estar oculta: la variante j reconstruye j.
            m = mascara.to(torch.bool) | torch.eye(
                N, dtype=torch.bool, device=frames.device).unsqueeze(0)
            tok[m] = p_cero.to(tok.dtype)
        tok = tok + self.pos_embed                               # (1,1,N,d) por broadcast

        # 3) transformer sobre B·N secuencias de N (+R) tokens
        d = tok.size(-1)
        seq = self._con_registros(tok.reshape(B * N, N, d))
        trans_out, all_attn = self.transformer(seq)
        if self.capture_attention and not self.training:
            self._attention_cache.append(
                [self._recortar_atencion(a) for a in all_attn])

        # 4) solo se decodifica el token j de la variante j
        trans_out = self._sin_registros(trans_out).reshape(B, N, N, d)
        sel = trans_out[:, idx, idx, :]                          # (B, N, d)

        latent = self.pred_linear(sel).reshape(B * N, *self.enc_shape)
        out = self.decoder(latent)
        out = F.interpolate(out, size=self.img_size, mode="bicubic",
                            align_corners=False)
        out = out.reshape(B, N, H, W)
        # El residual suma el canal de entrada, que en la variante j es CERO,
        # así que no aporta nada: se mantiene por coherencia con forward().
        return out

    def forward_con_atencion(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """Igual que forward() pero IMPONIENDO la matriz de atención A.

        A: (N, N) fila-estocástica. Se usa en TODAS las capas y TODAS las
        cabezas, sustituyendo al softmax(QKᵀ/√d).

        Para qué: la Etapa 4 del informe (§8.2) define la métrica de fidelidad
        como ΔMSE(e) = MSE(Ã(e)) − MSE(A), lo que exige RE-EJECUTAR el modelo
        con la matriz ablacionada y medir el cambio en el error de PREDICCIÓN.
        Sin esta función solo se podía medir el cambio de entropía de la propia
        matriz, que es una propiedad algebraica sin relación con el modelo ni
        con los datos, y por eso el umbral AUC > 0.005 de la Tabla 7 nunca era
        comparable con lo que se calculaba.

        Operacionalización: "correr el modelo con el grafo G \\ {e}" se traduce
        en imponer la misma matriz 12×12 (el consenso ablacionado) a las 2 capas
        y las 4 cabezas. Es una elección, y hay que declararla: el modelo real
        tiene 8 matrices distintas, no una.

        Con registros activos hay una diferencia que declarar: aqui se impone
        toda la masa sobre los N tokens de indice, mientras el modelo entrenado
        pudo mandar parte a los registros. El MSE base sale por tanto peor que
        el real. No afecta al DeltaMSE, que es una diferencia entre dos
        sustituciones con el mismo sesgo, pero si al valor absoluto de mse_base.
        """
        B = x.size(0)
        entrada = x[:, :, -1]
        xr = x.reshape(B * self.num_tokens, self.in_channels, *self.img_size)
        feat = self.encoder(xr).reshape(B, self.num_tokens, -1)
        src = self.token_norm(self.frame_proj(feat)) + self.pos_embed

        d_model = src.size(-1)
        for layer in self.transformer.layers:
            sa = layer.self_attn
            nh = sa.num_heads
            hd = d_model // nh
            _, _, v = F.linear(src, sa.in_proj_weight, sa.in_proj_bias).chunk(3, dim=-1)
            vh = v.reshape(B, self.num_tokens, nh, hd).permute(0, 2, 1, 3)
            Aq = A.to(device=src.device, dtype=vh.dtype)
            Aq = Aq.expand(B, nh, self.num_tokens, self.num_tokens)
            o = sa.out_proj((Aq @ vh).permute(0, 2, 1, 3).reshape(B, self.num_tokens, d_model))
            src = layer.norm1(src + o)
            src = layer.norm2(src + layer.linear2(layer.activation(layer.linear1(src))))

        latent = self.pred_linear(src).reshape(B * self.num_tokens, *self.enc_shape)
        out = self.decoder(latent)
        out = F.interpolate(out, size=self.img_size, mode="bicubic", align_corners=False)
        out = out.reshape(B, self.num_tokens, *self.img_size)
        return entrada + out if self.usar_residual else out

    def get_attention_weights(self):
        cache = self._attention_cache
        self._attention_cache = []
        # La masa que se fue a los registros se guarda ANTES de limpiar:
        # el caller normalmente drena la cache y despues pregunta por
        # ella, y si se borrase aqui siempre leeria None.
        if self._masa_registros:
            self.ultima_masa_registros = float(np.mean(self._masa_registros))
        self._masa_registros = []
        return cache

def attention_rollout(attention_per_layer: List[np.ndarray], residual_fraction: float = 0.5) -> np.ndarray:
    rollout = np.eye(attention_per_layer[0].shape[-1], dtype=np.float32)
    for attn_layer in attention_per_layer:
        avg = attn_layer.mean(axis=0)
        avg = residual_fraction * np.eye(avg.shape[0], dtype=np.float32) + (1-residual_fraction)*avg
        rollout = avg @ rollout
    return rollout / np.maximum(rollout.sum(axis=1, keepdims=True), 1e-12)

def entropia_normalizada(A: np.ndarray, eps: float = 1e-12) -> float:
    """Entropía media por fila, dividida por ln(N). 1.0 = fila uniforme."""
    Pm = np.clip(np.asarray(A, dtype=np.float64), eps, None)
    Pm = Pm / Pm.sum(axis=-1, keepdims=True)
    return float((-(Pm * np.log(Pm)).sum(axis=-1) / np.log(A.shape[-1])).mean())


def r2_rango1(A: np.ndarray) -> float:
    """¿A = d·I + 1·cᵀ? R² del ajuste, evaluado fuera de la diagonal.

    R² → 1 significa que todas las filas comparten el mismo perfil: el
    índice que consulta no influye en a quién mira. En ese caso la matriz
    no describe "atención entre índices", solo columnas sumidero, y las
    aristas que se extraigan de ella son una relectura de qué columnas
    reciben más atención.
    """
    A = np.asarray(A, dtype=np.float64)
    N = A.shape[0]
    fuera = ~np.eye(N, dtype=bool)
    col = np.array([A[fuera[:, j], j].mean() for j in range(N)])
    recon = np.where(np.eye(N, dtype=bool), A, np.tile(col, (N, 1)))
    ss_tot = ((A[fuera] - A[fuera].mean()) ** 2).sum()
    ss_res = ((A - recon)[fuera] ** 2).sum()
    return float(1.0 - ss_res / max(ss_tot, 1e-18))


def compute_attention_for_batch(model: ConvTransformer,
                                sample_input: torch.Tensor
                                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve (rollout, por_capa, por_cabeza) en una sola pasada.

      rollout    (N, N)        — attention rollout; lo que consume el resto
                                 del pipeline (FASE 1, paradigmas 1-5).
      por_capa   (L, N, N)     — A1, A2, ... promediadas sobre cabezas y
                                 batch, SIN el residual del rollout.
      por_cabeza (L, H, N, N)  — lo mismo pero sin promediar cabezas.

    Se guardan las tres porque no dicen lo mismo: el rollout usa
    residual_fraction=0.5, y con L=2 eso equivale a
        rollout = 0.25·I + 0.25·A1 + 0.25·A2 + 0.25·A2A1,
    es decir que ~el 77% de la diagonal que aparece en el mapa final la
    inyecta la propia medición, no el modelo (medido sobre 50 semillas:
    diag(A1)=0.110, diag(A2)=0.099, uniforme=0.083, diag(rollout)=0.325).
    Promediar las cabezas cuesta otro +0.17 de entropía normalizada.
    """
    base = model.module if isinstance(model, nn.DataParallel) else model
    base.capture_attention = True
    base.eval()
    with torch.no_grad():
        _ = model(sample_input.to(device))
    cache = base.get_attention_weights()
    base.capture_attention = False

    if not cache:
        eye = np.eye(NUM_INDICES, dtype=np.float32)
        return eye, eye[None], eye[None, None]

    rollouts = []
    for sample_layers in cache:
        for b in range(sample_layers[0].shape[0]):
            rollouts.append(attention_rollout([l[b] for l in sample_layers]))
    rollout = np.mean(rollouts, axis=0)

    n_layers = len(cache[0])
    por_cabeza = np.stack([
        np.concatenate([c[l] for c in cache], axis=0).mean(axis=0)
        for l in range(n_layers)
    ])                                    # (L, H, N, N)
    por_capa = por_cabeza.mean(axis=1)    # (L, N, N)
    return (rollout.astype(np.float32),
            por_capa.astype(np.float32),
            por_cabeza.astype(np.float32))


def compute_rollout_for_batch(model: ConvTransformer, sample_input: torch.Tensor) -> np.ndarray:
    """Compatibilidad: solo el rollout."""
    return compute_attention_for_batch(model, sample_input)[0]


def reportar_capas_atencion(rollout: np.ndarray, por_capa: np.ndarray,
                            por_cabeza: np.ndarray, etiqueta: str = "") -> Dict:
    """Imprime A1, A2 y el rollout lado a lado y devuelve las métricas."""
    unif = 1.0 / rollout.shape[-1]
    print(f"\n  ATENCIÓN POR CAPA{(' — ' + etiqueta) if etiqueta else ''}")
    print(f"  normalización de la atención: {ATENCION_NORMALIZACION}")
    if ATENCION_NORMALIZACION != "softmax":
        sf = [float(por_capa[li].sum(axis=1).mean()) for li in range(por_capa.shape[0])]
        print(f"  suma de fila por capa: " + "  ".join(f"A{i+1}={v:.3f}" for i, v in enumerate(sf)))
        print(f"  (con softmax sería 1.000 por construcción; por debajo significa")
        print(f"   que el índice decidió NO mirar a los otros, que es información)")
    print(f"  {'':<26}{'diagonal':>10}{'H/lnN':>9}{'R2 rango-1':>12}")
    met = {}
    for li in range(por_capa.shape[0]):
        A = por_capa[li]
        d, h, r2 = float(np.diag(A).mean()), entropia_normalizada(A), r2_rango1(A)
        met[f"A{li+1}"] = dict(diag=d, entropia=h, r2_rango1=r2,
                               suma_fila=float(A.sum(axis=1).mean()))
        print(f"  {f'A{li+1} (cabezas promed.)':<26}{d:>10.4f}{h:>9.4f}{r2:>12.4f}")
        for hh in range(por_cabeza.shape[1]):
            Ah = por_cabeza[li, hh]
            print(f"  {f'   cabeza {hh}':<26}{np.diag(Ah).mean():>10.4f}"
                  f"{entropia_normalizada(Ah):>9.4f}{r2_rango1(Ah):>12.4f}")
    d, h, r2 = (float(np.diag(rollout).mean()),
                entropia_normalizada(rollout), r2_rango1(rollout))
    met["rollout"] = dict(diag=d, entropia=h, r2_rango1=r2)
    print(f"  {'rollout (residual=0.5)':<26}{d:>10.4f}{h:>9.4f}{r2:>12.4f}")
    print(f"  {'uniforme de referencia':<26}{unif:>10.4f}{1.0:>9.4f}{'—':>12}")
    if por_capa.shape[0] >= 2 and d > 1e-9:
        print(f"  → el término 0.25·I del rollout explica el "
              f"{0.25 / d * 100:.1f}% de su diagonal")
    return met

def plot_attention_map(attention_matrix: np.ndarray, labels: List[str], output_path: str, title: str = "Mapa de Atención (Rollout) — 12 Índices") -> None:
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(attention_matrix, cmap="YlOrRd", aspect="equal")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Índice consultado", fontsize=11)
    ax.set_ylabel("Índice que consulta", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)

    vmin, vmax = attention_matrix.min(), attention_matrix.max()
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = attention_matrix[i, j]
            color = "white" if val > (vmin + vmax) / 2 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Atención", fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Mapa de atención: {output_path}")

def _bytes_stack(*arrs: np.ndarray) -> int:
    """Bytes que ocuparán los frames en VRAM como float32."""
    return int(sum(int(np.prod(a.shape)) * 4 for a in arrs))


def _cabe_en_vram(*arrs: np.ndarray, fraccion: float = 0.10) -> bool:
    """¿El stack cabe en `fraccion` de la VRAM libre?

    Se mide la VRAM LIBRE, no la total: en la laptop el escritorio ya ocupa
    parte, y en el cluster puede haber otro job. El 10% deja el 90% para
    activaciones, que es lo que de verdad causa OOM (el forward enmascarado
    corre el encoder sobre B*12 imágenes, no B).
    """
    if not torch.cuda.is_available():
        return False
    try:
        libre, _total = torch.cuda.mem_get_info()
    except Exception:
        return False
    return _bytes_stack(*arrs) < libre * fraccion


class _LoaderVRAM:
    """Batches ya residentes en VRAM, con la interfaz de DataLoader.

    Devuelve (frame, frame) igual que FramesDataset, así que los bucles de
    entrenamiento y validación no cambian: el `.to(device)` de dentro es un
    no-op sobre un tensor que ya está en el dispositivo.
    """

    def __init__(self, frames: np.ndarray, batch_size: int, shuffle: bool,
                 dev: torch.device):
        self.t = torch.from_numpy(np.ascontiguousarray(
            np.transpose(frames, (0, 3, 1, 2)))).float().to(dev)
        self.bs = int(batch_size)
        self.shuffle = bool(shuffle)
        self.dev = dev

    @property
    def dataset(self):
        return self.t

    def __len__(self) -> int:
        return (self.t.size(0) + self.bs - 1) // self.bs

    def __iter__(self):
        n = self.t.size(0)
        idx = (torch.randperm(n, device=self.dev) if self.shuffle
               else torch.arange(n, device=self.dev))
        for i in range(0, n, self.bs):
            b = self.t[idx[i:i + self.bs]]
            yield b, b


def _lr_warmup_coseno(epoca: int, total: int, warmup: int,
                      piso: float = 0.05) -> float:
    """Multiplicador de LR: rampa lineal `warmup` épocas, luego coseno.

    POR QUÉ, y no ReduceLROnPlateau. En un Transformer los pesos Q/K se mueven
    caóticamente en las primeras épocas; sin calentamiento la matriz de
    atención colapsa en columnas sumidero antes de que el modelo aprenda nada,
    y ese colapso es justo el artefacto que este proyecto mide. Con rampa, las
    50 semillas parten del mismo régimen y la varianza entre ellas deja de
    mezclar "mala inicialización" con "estructura del fenómeno".
    """
    if epoca <= warmup:
        return float(epoca) / float(max(1, warmup))
    avance = float(epoca - warmup) / float(max(1, total - warmup))
    return max(piso, 0.5 * (1.0 + float(np.cos(np.pi * min(1.0, avance)))))


def train_convtransformer(stack: np.ndarray,
                           seq_length: int = SEQ_LENGTH,
                           total_epochs: int = TOTAL_EPOCHS,
                           batch_size: int = BATCH_SIZE,
                           lr: float = LR_CONVTRANSFORMER,
                           weight_decay: float = 1e-5,
                           patience: int = PATIENCE,
                           grad_clip: float = 1.0,
                           num_workers: int = N_WORKERS,
                           ckpt_path: str = CKPT_CONVTRANSFORMER,
                           dates_millis: Optional[np.ndarray] = None):
    print(f"\n{'='*60}")
    print(f"ENTRENAMIENTO CONVTRANSFORMER — 12 ÍNDICES")
    print(f"{'='*60}")

    (X_tr, Y_tr), (X_va, Y_va), img_size, scaler = process_indices_data(
        stack, seq_length=seq_length, dates_millis=dates_millis
    )
    print(f"Train: {X_tr.shape}  (N, seq, H, W, 12)")
    print(f"Val:   {X_va.shape}")
    print(f"img_size: {img_size}")

    enmascarado = (TAREA == "enmascarado")
    if enmascarado:
        # Una muestra por frame; las 12 variantes se generan dentro del modelo
        # con forward_enmascarado (12x menos pasadas de encoder y decoder).
        train_ds = FramesDataset(X_tr)
        val_ds = FramesDataset(X_va)
        print(f"  Camino rápido: {len(train_ds)} frames por época "
              f"(en vez de {len(train_ds)*NUM_INDICES} muestras sueltas)")
        reportar_mascara()
    else:
        train_ds = IndicesDataset(X_tr, Y_tr)
        val_ds = IndicesDataset(X_va, Y_va)

    # Dataset residente en VRAM si cabe: con N_WORKERS=0 (obligado en Windows)
    # el DataLoader convierte numpy->torch en el proceso principal en CADA
    # batch de CADA época, y con 186 frames de 12x52x51 eso es el cuello de
    # botella, no la GPU. Subir el stack entero una vez cuesta 24 MB y elimina
    # la copia host->device por batch. No cambia una sola cuenta: los mismos
    # tensores, el mismo orden de barajado, el mismo tamaño de batch.
    residente = (enmascarado and device.type == "cuda"
                 and _cabe_en_vram(X_tr, X_va))
    if residente:
        train_loader = _LoaderVRAM(X_tr, batch_size, True, device)
        val_loader = _LoaderVRAM(X_va, batch_size, False, device)
        num_workers = 0
        print(f"  Datos residentes en VRAM: "
              f"{_bytes_stack(X_tr, X_va) / 1024**2:.1f} MB (sin DataLoader)")
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=False,
            persistent_workers=True if num_workers > 0 else False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
        )
    print(f"Batch size: {batch_size}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = ConvTransformer(
        num_indices=NUM_INDICES,
        seq_length=seq_length,
        img_size=img_size,
        num_heads=4,
        num_layers=2,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros: {n_params:,} ({n_params/1e6:.2f}M)")

    # Sin weight decay en bias, LayerNorm y pos_embed: decaer estos hacia 0
    # aplana el softmax de atención (Q/K empujados a 0) y desestabiliza el
    # embedding posicional que distingue a los 12 índices como tokens.
    no_decay = ("bias", "norm", "pos_embed")
    decay_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)]
    no_decay_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and any(nd in n for nd in no_decay)]
    # fused=True fusiona la actualización de TODOS los parámetros en un solo
    # kernel CUDA en vez de lanzar uno por tensor. Misma matemática exacta,
    # 15-30% menos tiempo en el paso del optimizador. Los dos grupos de
    # weight_decay se conservan: decaer bias/norm/pos_embed hacia 0 aplana el
    # softmax de atención, que es exactamente el artefacto que aquí se mide.
    _fused = (device.type == "cuda")
    try:
        optimizer = optim.AdamW(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=lr, fused=_fused,
        )
    except (TypeError, RuntimeError):
        # PyTorch < 2.0 no acepta fused, y algunas builds lo rechazan.
        optimizer = optim.AdamW(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=lr,
        )
        _fused = False
    warmup_epochs = min(WARMUP_EPOCHS, max(1, total_epochs // 5))
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda ep: _lr_warmup_coseno(ep + 1, total_epochs, warmup_epochs),
    )
    mse_criterion = nn.MSELoss()

    amp_scaler = GradScaler('cuda', enabled=USE_AMP)

    best_val = float("inf")
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "lr": []}

    print(f"\nEntrenando en {device}...")
    if USE_MIXED_PRECISION:
        print(f"   Mixed precision (AMP): ACTIVADO")
    print(f"   Early stopping patience: {patience}")
    print()

    for epoch in range(1, total_epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        n_samples = 0
        n_skipped = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=USE_AMP):
                if enmascarado:
                    # Máscara nueva en cada batch: los k-1 distractores cambian,
                    # así que el modelo no puede memorizar qué vecino mirar.
                    msk = mascara_multiple(inputs.size(0), inputs.size(1),
                                           MASCARA_K, inputs.device)
                    out = model.forward_enmascarado(inputs, msk)
                else:
                    out = model(inputs)
                loss = mse_criterion(out, targets)

            if not torch.isfinite(loss):
                # Un NaN/Inf que entra a backward() corrompe los pesos para
                # siempre (a diferencia de un gradiente con Inf, que
                # clip_grad_norm_ sí puede neutralizar) — todas las épocas
                # siguientes quedarían en NaN sin recuperación posible.
                # Descartar el batch entero es más seguro que dejarlo pasar.
                n_skipped += 1
                continue

            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            amp_scaler.step(optimizer)
            amp_scaler.update()

            train_loss += loss.item() * inputs.size(0)
            n_samples += inputs.size(0)

        if n_skipped:
            print(f"  ⚠️  {n_skipped} batch(es) con loss no finito descartados esta época")
        train_loss /= max(n_samples, 1)

        model.eval()
        val_loss = 0.0
        n_val = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with autocast("cuda", enabled=USE_AMP):
                    if enmascarado:
                        # En validación la máscara es DETERMINISTA (generador con
                        # semilla fija): si cambiara en cada época, el val_loss
                        # mezclaría la mejora del modelo con la suerte del
                        # sorteo y el early stopping pararía por ruido.
                        g = torch.Generator(device=inputs.device)
                        g.manual_seed(SEMILLA_NULO + n_val)
                        msk = mascara_multiple(inputs.size(0), inputs.size(1),
                                               MASCARA_K, inputs.device,
                                               generator=g)
                        out = model.forward_enmascarado(inputs, msk)
                    else:
                        out = model(inputs)
                    val_loss += mse_criterion(out, targets).item() * inputs.size(0)
                n_val += inputs.size(0)

        val_loss /= max(n_val, 1)
        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(lr_now)

        epoch_time = time.time() - epoch_start

        improvement = ""
        if val_loss < best_val - 1e-7:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
            improvement = "guardado"
        elif epoch <= warmup_epochs:
            # Durante el calentamiento el LR es una fracción del objetivo y el
            # val_loss casi no mejora. Contar esas épocas dispararía el early
            # stopping antes de que el warmup termine, y las 50 semillas
            # entrenarían 6 épocas cada una.
            improvement = " (warmup)"
        else:
            epochs_no_improve += 1

        print(f"Epoch {epoch:3d}/{total_epochs} "
              f"| train MSE: {train_loss:.6f} "
              f"| val MSE: {val_loss:.6f} "
              f"| lr: {lr_now:.2e} "
              f"| {epoch_time:.1f}s{improvement}")

        if epochs_no_improve >= patience:
            print(f"\nDetención temprana en la época {epoch} (sin mejora en {patience} épocas)")
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"\nEntrenamiento finalizado. Mejor val MSE: {best_val:.6f}")
    print(f"   Modelo guardado: {ckpt_path}")
    _plot_training_history(history, ckpt_path.replace(".pth", "_history.png"))

    # Sin baseline, "val MSE = 0.27" no dice si el modelo aprendió algo: el
    # número está en unidades de StandardScaler y no es comparable con el
    # 0.007724 (NDVI crudo) de la presentación. Se compara contra
    # persistencia y media de ventana sobre el MISMO set de validación.
    if enmascarado:
        reportar_baselines_enmascarado(X_va, best_val, scaler=scaler)
    else:
        reportar_baselines(X_va, Y_va, best_val, scaler=scaler,
                           etiqueta="ConvTransformer 12 índices")

    # persistent_workers=True deja subprocesos vivos hasta que el loader se
    # recolecta; en Windows (spawn) eso no es determinista y con 50 semillas
    # seguidas acumula procesos → OpenBLAS memory allocation failed.
    # Cerrar explícito aquí evita el leak.
    if num_workers > 0:
        train_loader._iterator = None
        val_loader._iterator = None
    del train_loader, val_loader
    gc.collect()

    return model, (X_va, Y_va), scaler, best_val

def _plot_training_history(history: dict, output_path: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], "b-", label="Train MSE", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "r-", label="Val MSE", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title("Curva de aprendizaje", fontweight="bold")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["lr"], "g-", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning Rate")
    ax2.set_title("Learning Rate Schedule", fontweight="bold")
    ax2.set_yscale("log")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Curva de aprendizaje: {output_path}")

# ============================================================================
# NOTA: aqui vivia el NDVITemporalTransformer (atencion temporal entre frames
# de UN solo indice, el diagrama verde de la slide 16). ELIMINADO: el trabajo
# se centra exclusivamente en el ConvTransformer de atencion ENTRE INDICES
# (diagrama rojo). Se fueron con el: process_single_index_temporal_data,
# TemporalIndexDataset, NDVITemporalTransformer, train_temporal_transformer
# y main_parte_temporal.
# ============================================================================

# ============================================================================
# PARTE 2b · ENTRENAMIENTO POR SEMILLA (genera attention_seed_*.npy)
#
# Reentrena el ConvTransformer COMPLETO por semilla —init de pesos + orden de
# batches— porque es lo que hace comparables las M realizaciones bajo la
# hipótesis "mismo fenómeno, distinto ruido de entrenamiento" que FASE 1 y el
# Paradigma 1 necesitan contrastar.
# ============================================================================

def entrenar_y_extraer_attention_seed(stack: np.ndarray, seed: int,
                                      seq_length: int = SEQ_LENGTH,
                                      total_epochs: int = TOTAL_EPOCHS,
                                      batch_size: int = BATCH_SIZE,
                                      lr: float = LR_CONVTRANSFORMER,
                                      patience: int = PATIENCE,
                                      num_workers: int = N_WORKERS,
                                      dates_millis: Optional[np.ndarray] = None
                                      ) -> np.ndarray:
    """Entrena un ConvTransformer con `seed`.

    Devuelve (rollout, por_capa, por_cabeza, best_val) — ver
    compute_attention_for_batch para qué es cada una.
    """
    set_seed(seed)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"model_seed_{seed}.pth")
    model, (X_val, Y_val), _, best_val = train_convtransformer(
        stack, seq_length=seq_length, total_epochs=total_epochs,
        batch_size=batch_size, lr=lr, patience=patience,
        num_workers=num_workers, ckpt_path=ckpt_path,
        dates_millis=dates_millis,
    )
    if TAREA == "enmascarado":
        # Se extrae con forward_enmascarado, igual que en entrenamiento: cada
        # frame produce las 12 variantes y por tanto 12 matrices de atención,
        # una por índice oculto. Con LECTURA_DIAGONAL se conserva de cada
        # variante v solo la fila v — la del token oculto, que es el que el
        # decoder reconstruye. Ver LECTURA_DIAGONAL.
        base = model.module if isinstance(model, nn.DataParallel) else model
        base.capture_attention = True
        base.eval()
        N_tok = base.num_tokens
        n_disp = len(X_val)
        n_fr = n_disp if N_FRAMES_LECTURA is None else min(int(N_FRAMES_LECTURA),
                                                           n_disp)
        lote = max(1, int(FRAMES_POR_LOTE_LECTURA))
        print(f"  lectura de atención: {n_fr}/{n_disp} frames de validación · "
              f"diagonal={LECTURA_DIAGONAL} · k_lectura={MASCARA_K_LECTURA}")
        rolls: List[np.ndarray] = []
        cabezas: List[np.ndarray] = []      # cada elemento: (L, H, N, N)
        with torch.no_grad():
            for ini in range(0, n_fr, lote):
                fr = torch.from_numpy(np.ascontiguousarray(np.transpose(
                    X_val[ini:ini + lote], (0, 3, 1, 2)))).float().to(device)
                # La matriz se lee con k = MASCARA_K_LECTURA (por defecto 1), no
                # con el k del entrenamiento. Con k>1 la fila j tendría k-1
                # columnas apagadas al azar y A dependería del sorteo; con k=1
                # la fila j es "de quién depende j teniendo los otros 11
                # disponibles", que es la cantidad que el framework interpreta.
                msk_lec = (None if MASCARA_K_LECTURA <= 1 else
                           mascara_multiple(fr.size(0), fr.size(1),
                                            MASCARA_K_LECTURA, fr.device))
                _ = base.forward_enmascarado(fr, msk_lec)
                for capas in base.get_attention_weights():
                    n_var = capas[0].shape[0]       # frames_del_lote · N
                    assert n_var % N_tok == 0, \
                        f"cache con {n_var} filas, no múltiplo de {N_tok}"
                    # tok se aplanó como (frame, variante) → b = f*N + v
                    if LECTURA_DIAGONAL:
                        for f in range(n_var // N_tok):
                            R = np.stack([
                                attention_rollout(
                                    [l[f * N_tok + v] for l in capas])[v]
                                for v in range(N_tok)])
                            rolls.append(
                                R / np.maximum(R.sum(axis=1, keepdims=True), 1e-12))
                            cabezas.append(np.stack([
                                np.stack([capas[l][f * N_tok + v, :, v, :]
                                          for v in range(N_tok)], axis=1)
                                for l in range(len(capas))]))
                    else:
                        for b in range(n_var):
                            rolls.append(attention_rollout([l[b] for l in capas]))
                            cabezas.append(np.stack([capas[l][b]
                                                     for l in range(len(capas))]))
        base.capture_attention = False
        rollout = np.mean(rolls, axis=0).astype(np.float32)
        por_cabeza = np.mean(np.stack(cabezas), axis=0).astype(np.float32)
        por_capa = por_cabeza.mean(axis=1).astype(np.float32)
        reportar_capas_atencion(rollout, por_capa, por_cabeza,
                                etiqueta=f"semilla {seed}")
        return rollout, por_capa, por_cabeza, float(best_val)
    else:
        sample_input = torch.from_numpy(
            np.ascontiguousarray(np.transpose(X_val[:16], (0, 4, 1, 2, 3)))
        ).float()
    rollout, por_capa, por_cabeza = compute_attention_for_batch(model, sample_input)
    reportar_capas_atencion(rollout, por_capa, por_cabeza,
                            etiqueta=f"semilla {seed}")
    return rollout, por_capa, por_cabeza, float(best_val)


def _resumen_capas_multisemilla(seeds: List[int]) -> None:
    """Agrega A1, A2 y el rollout sobre todas las semillas disponibles."""
    rolls, capas = [], []
    for s in seeds:
        pr = os.path.join(DIR_MATRICES, f"attention_seed_{s}.npy")
        pc = os.path.join(DIR_MATRICES, f"attention_capas_seed_{s}.npy")
        if os.path.exists(pr) and os.path.exists(pc):
            try:
                rolls.append(np.load(pr)); capas.append(np.load(pc))
            except Exception:
                pass
    if not capas:
        return
    C = np.stack(capas)          # (M, L, N, N)
    R = np.stack(rolls)          # (M, N, N)
    M, L = C.shape[0], C.shape[1]

    print(f"\n{'='*72}")
    print(f" ATENCIÓN POR CAPA — AGREGADO SOBRE {M} SEMILLAS")
    print(f"{'='*72}")
    print(f"  {'':<26}{'diagonal':>18}{'H/lnN':>16}{'R2 rango-1':>16}")
    filas = [(f"A{li+1}", C[:, li]) for li in range(L)] + [("rollout", R)]
    for nombre, X in filas:
        d = np.array([np.diag(x).mean() for x in X])
        h = np.array([entropia_normalizada(x) for x in X])
        r = np.array([r2_rango1(x) for x in X])
        print(f"  {nombre:<26}{d.mean():>10.4f}±{d.std():<7.4f}"
              f"{h.mean():>8.4f}±{h.std():<7.4f}{r.mean():>8.4f}±{r.std():<7.4f}")
    print(f"  {'uniforme de referencia':<26}{1.0/C.shape[-1]:>10.4f}"
          f"{'':<8}{1.0:>8.4f}")
    print("\n  Cómo leerlo:")
    print("   · diagonal ≈ 1/N en A1/A2 → el modelo NO aprende la")
    print("     identidad; si el rollout la tiene alta, la pone su")
    print("     residual_fraction, no el modelo.")
    print("   · R2 rango-1 → 1 → todas las filas iguales: no hay relación")
    print("     entre índices, solo columnas sumidero.")


def _spearman_filas(A: np.ndarray, B: np.ndarray) -> float:
    """Spearman medio por fila entre dos matrices, ignorando la diagonal.

    Mismo criterio que usa la FASE 1, para que los números sean comparables.
    """
    N = A.shape[0]
    rs = []
    for i in range(N):
        m = np.ones(N, dtype=bool); m[i] = False
        a, b = A[i][m], B[i][m]
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            continue
        r = stats.spearmanr(a, b).statistic
        if np.isfinite(r):
            rs.append(r)
    return float(np.mean(rs)) if rs else float("nan")


def _estabilidad_media(mats: List[np.ndarray], max_pares: int = 300) -> float:
    """Spearman medio sobre pares de realizaciones."""
    M = len(mats)
    pares = [(i, j) for i in range(M) for j in range(i + 1, M)]
    if max_pares and len(pares) > max_pares:
        rng = np.random.default_rng(SEMILLA_NULO)
        pares = [pares[k] for k in rng.choice(len(pares), max_pares, replace=False)]
    vs = [_spearman_filas(mats[i], mats[j]) for i, j in pares]
    vs = [v for v in vs if np.isfinite(v)]
    return float(np.mean(vs)) if vs else float("nan")


def _emparejar_cabezas(cabezas: np.ndarray) -> np.ndarray:
    """cabezas: (M, H, N, N) de UNA capa → (M, H, N, N) con las cabezas
    alineadas contra la semilla 0.

    Necesario porque la atención multi-cabeza es invariante a permutación de
    cabezas: la cabeza 0 de una semilla no tiene por qué corresponder a la
    cabeza 0 de otra. Sin emparejar, comparar "la cabeza 0" entre semillas
    mide ruido de etiquetado.
    """
    from scipy.optimize import linear_sum_assignment
    H = cabezas.shape[1]

    def _alinear(ref):
        out = []
        for Hm in cabezas:
            coste = np.array([[np.abs(ref[a] - Hm[b]).mean() for b in range(H)]
                              for a in range(H)])
            _, col = linear_sum_assignment(coste)
            out.append(Hm[col])
        return np.stack(out)

    # Emparejamiento iterativo contra la MEDIA, no contra una semilla concreta.
    # Con una referencia arbitraria el resultado depende de cuál se elija: en
    # pruebas, cambiar la semilla de referencia movia rho de 0.41 a 0.51 y
    # cambiaba que cabeza salia "la mejor". Tres pasadas contra la media
    # convergen y quitan esa dependencia.
    ref = cabezas[0]
    alineadas = _alinear(ref)
    for _ in range(3):
        nueva_ref = alineadas.mean(axis=0)
        previas = alineadas
        alineadas = _alinear(nueva_ref)
        if np.array_equal(previas, alineadas):
            break
    return alineadas


def calidad_agrupamiento_cabezas(emparejadas: np.ndarray) -> Dict[str, float]:
    """¿Las cabezas forman grupos reproducibles entre semillas?

    emparejadas: (M, H, N, N) ya alineadas. Se compara, para cada cabeza,
    su distancia al centroide de su ranura contra la del centroide ajeno
    más cercano (silueta). Si las cabezas de distintas semillas no se
    parecen entre sí más de lo que se parecen a cualquier otra, el
    emparejamiento es arbitrario y "la cabeza k" no significa nada.
    """
    M, H = emparejadas.shape[0], emparejadas.shape[1]
    cent = emparejadas.mean(axis=0)
    propio, ajeno = [], []
    for m in range(M):
        for h in range(H):
            ds = [float(np.abs(emparejadas[m, h] - cent[k]).mean()) for k in range(H)]
            propio.append(ds[h])
            ajeno.append(min(ds[k] for k in range(H) if k != h))
    propio, ajeno = np.array(propio), np.array(ajeno)
    sil = float(np.mean((ajeno - propio) / np.maximum(ajeno, propio)))
    return {"silueta": sil,
            "ratio_ajeno_propio": float(ajeno.mean() / max(propio.mean(), 1e-12)),
            "frac_mal_asignadas": float((ajeno < propio).mean())}


def preparar_matrices_atencion(seeds: List[int] = SEEDS,
                               fuente: str = None) -> Optional[str]:
    """Materializa en attention_seed_*.npy la fuente elegida (FUENTE_ATENCION).

    Los 4 cargadores del pipeline leen attention_seed_*.npy, así que en vez de
    tocarlos se reescribe ese fichero con la matriz elegida. El rollout NUNCA
    se pierde: se respalda una sola vez como attention_rollout_seed_*.npy, y
    A1/A2/cabezas siguen en sus propios ficheros.

    Es idempotente y reversible: con FUENTE_ATENCION="rollout" se restaura.
    """
    fuente = fuente or FUENTE_ATENCION
    validas = ("rollout", "producto", "capa1", "capa2", "cabeza_estable")
    if fuente not in validas:
        print(f"FUENTE_ATENCION={fuente!r} no válida; usando 'rollout'.")
        fuente = "rollout"

    def _p(plantilla, s):
        return os.path.join(DIR_MATRICES, plantilla.format(s=s))

    # Respaldo del rollout (una sola vez, antes de sobrescribir nada).
    for s in seeds:
        act, bak = _p("attention_seed_{s}.npy", s), _p("attention_rollout_seed_{s}.npy", s)
        if os.path.exists(act) and not os.path.exists(bak):
            np.save(bak, np.load(act))

    print(f"\n{'='*72}")
    print(f" FUENTE DE LA MATRIZ DE ATENCIÓN: {fuente}")
    print(f"{'='*72}")

    if fuente == "rollout":
        n_rest = 0
        for s in seeds:
            bak = _p("attention_rollout_seed_{s}.npy", s)
            if os.path.exists(bak):
                np.save(_p("attention_seed_{s}.npy", s), np.load(bak))
                n_rest += 1
        print(f"  Restauradas {n_rest}/{len(seeds)} matrices de rollout.")
        print("  (comportamiento original del pipeline)")
        _escribir_manifiesto(fuente, None, None)
        return fuente

    # Fuentes derivadas: necesitan los ficheros por capa / por cabeza.
    plantilla = ("attention_cabezas_seed_{s}.npy" if fuente == "cabeza_estable"
                 else "attention_capas_seed_{s}.npy")
    faltan = [s for s in seeds if not os.path.exists(_p(plantilla, s))]
    if faltan:
        print(f"  Faltan {len(faltan)}/{len(seeds)} ficheros {plantilla.format(s='*')}.")
        print("  Se mantiene el rollout. Re-ejecuta la PARTE 2b para generarlos.")
        return "rollout"

    datos = np.stack([np.load(_p(plantilla, s)) for s in seeds])

    if fuente == "producto":
        # Mismo encadenado de capas que el rollout pero con
        # residual_fraction = 0: A_L · ... · A_1, sin la identidad.
        elegidas = []
        for m in range(datos.shape[0]):
            Acum = datos[m, 0]
            for li in range(1, datos.shape[1]):
                Acum = datos[m, li] @ Acum
            elegidas.append(Acum)
        elegidas = np.stack(elegidas)
        detalle = f"producto de {datos.shape[1]} capas, sin residual"
        cabeza = None
    elif fuente in ("capa1", "capa2"):
        li = 0 if fuente == "capa1" else 1
        if li >= datos.shape[1]:
            print(f"  No hay capa {li + 1}; se mantiene el rollout.")
            return "rollout"
        elegidas = datos[:, li]
        detalle = f"A{li + 1} (promedio de cabezas)"
        cabeza = None
    else:
        if CAPA_CABEZA >= datos.shape[1]:
            print(f"  CAPA_CABEZA={CAPA_CABEZA} fuera de rango; se mantiene el rollout.")
            return "rollout"
        emparejadas = _emparejar_cabezas(datos[:, CAPA_CABEZA])
        n_cab = emparejadas.shape[1]
        print(f"  Emparejando {n_cab} cabezas de A{CAPA_CABEZA + 1} "
              f"entre {len(seeds)} semillas (asignación húngara)...")
        cal = calidad_agrupamiento_cabezas(emparejadas)
        print(f"  Calidad del agrupamiento: silueta={cal['silueta']:.3f}  "
              f"ajeno/propio={cal['ratio_ajeno_propio']:.2f}x  "
              f"mal asignadas={cal['frac_mal_asignadas']*100:.0f}%")
        if cal["silueta"] < SILUETA_MINIMA:
            print(f"  [!] Silueta {cal['silueta']:.3f} < {SILUETA_MINIMA}: las "
                  f"cabezas NO forman grupos reproducibles entre semillas.")
            print("      Emparejarlas es arbitrario: 'la cabeza k' no designa")
            print("      nada estable, y su rho depende del orden de las semillas.")
            print("      Se mantiene el rollout. Usa FUENTE_ATENCION='producto'.")
            return "rollout"
        rhos = [_estabilidad_media([emparejadas[m, h] for m in range(len(seeds))])
                for h in range(n_cab)]
        print(f"  {'cabeza':>8}{'rho entre semillas':>22}{'R2 rango-1':>14}")
        for h in range(n_cab):
            Mh = emparejadas[:, h].mean(0)
            print(f"  {h:>8d}{rhos[h]:>22.4f}{r2_rango1(Mh):>14.4f}")
        cabeza = CABEZA_FORZADA if CABEZA_FORZADA is not None else int(np.nanargmax(rhos))
        elegidas = emparejadas[:, cabeza]
        detalle = (f"cabeza {cabeza} de A{CAPA_CABEZA + 1}, "
                   f"rho={rhos[cabeza]:.4f}")
        print(f"  → elegida la cabeza {cabeza}"
              f"{' (forzada)' if CABEZA_FORZADA is not None else ' (la más estable)'}")

    # Comparativa contra el rollout, para que quede constancia del cambio.
    rolls = [np.load(_p("attention_rollout_seed_{s}.npy", s)) for s in seeds
             if os.path.exists(_p("attention_rollout_seed_{s}.npy", s))]
    if rolls:
        print(f"\n  {'':<26}{'rho':>10}{'diagonal':>11}{'R2 rango-1':>13}")
        for nom, mats in [("rollout (antes)", rolls),
                          (detalle, list(elegidas))]:
            Mm = np.mean(mats, axis=0)
            print(f"  {nom:<26}{_estabilidad_media(list(mats)):>10.4f}"
                  f"{np.diag(Mm).mean():>11.4f}{r2_rango1(Mm):>13.4f}")

    for s, A in zip(seeds, elegidas):
        M = np.asarray(A, dtype=np.float32)
        if RENORMALIZAR_FILAS:
            M = M / np.maximum(M.sum(axis=1, keepdims=True), 1e-12)
        np.save(_p("attention_seed_{s}.npy", s), M)
    print(f"\n  Escritas {len(seeds)} matrices en attention_seed_*.npy")
    print("  (el rollout sigue en attention_rollout_seed_*.npy;")
    print("   para volver atrás: FUENTE_ATENCION = 'rollout')")
    _escribir_manifiesto(fuente, cabeza, detalle)
    return fuente


def _escribir_manifiesto(fuente: str, cabeza: Optional[int],
                         detalle: Optional[str]) -> None:
    """Deja constancia en disco de qué contiene attention_seed_*.npy.

    Sin esto es imposible saber, mirando resultados/, si vienen del rollout
    o de una cabeza — los ficheros se llaman igual.
    """
    ruta = os.path.join(DIR_MATRICES, "fuente_atencion.json")
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump({"fuente": fuente, "capa": CAPA_CABEZA,
                   "cabeza": cabeza, "detalle": detalle}, f, indent=2,
                  ensure_ascii=False)


def main_parte2_multiseed(seeds: List[int] = SEEDS,
                          seq_length: int = SEQ_LENGTH,
                          total_epochs: int = TOTAL_EPOCHS,
                          batch_size: int = BATCH_SIZE,
                          lr: float = LR_CONVTRANSFORMER,
                          patience: int = PATIENCE,
                          num_workers: int = N_WORKERS,
                          forzar_reentrenamiento: bool = False) -> None:
    """Entrena M = len(seeds) ConvTransformers y guarda attention_seed_{seed}.npy.

    Reanudable: si `attention_seed_{seed}.npy` ya existe con la forma correcta
    (NUM_INDICES, NUM_INDICES), esa semilla se salta. Con
    `forzar_reentrenamiento=True` se ignora el cache y se reentrena todo — útil
    si se sospecha que las matrices existentes vienen de otra arquitectura o de
    una corrida vieja (como la que causó el mismatch 6x6 vs 12x12 en FASE 1).
    """
    if not os.path.exists(OUTPUT_NPY):
        print(f"No se encontró {OUTPUT_NPY}. Corre PARTE 1 primero.")
        return
    stack = np.load(OUTPUT_NPY)
    print(f"Stack cargado: {stack.shape}  (N_dates, H, W, 12)")
    fechas = cargar_fechas_stack(len(stack))

    os.makedirs(DIR_MATRICES, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"PARTE 2b · ENTRENAMIENTO POR SEMILLA ({len(seeds)} semillas)")
    print(f"{'='*60}")

    pendientes = []
    for s in seeds:
        out_path = os.path.join(DIR_MATRICES, f"attention_seed_{s}.npy")
        capas_path = os.path.join(DIR_MATRICES, f"attention_capas_seed_{s}.npy")
        ya_existe = False
        # Se exige también el fichero por capa: una semilla que solo tenga
        # el rollout viene de una corrida anterior a este cambio y hay que
        # rehacerla, o A1/A2 quedarían sin generar para esa semilla.
        if (not forzar_reentrenamiento and os.path.exists(out_path)
                and os.path.exists(capas_path)):
            try:
                ya_existe = np.load(out_path).shape == (NUM_INDICES, NUM_INDICES)
            except Exception:
                ya_existe = False
        if not ya_existe:
            pendientes.append(s)

    print(f"Semillas ya calculadas: {len(seeds) - len(pendientes)}/{len(seeds)}")
    print(f"Semillas pendientes:    {len(pendientes)}/{len(seeds)}")

    val_losses_path = os.path.join(DIR_MATRICES, "val_losses.json")
    if os.path.exists(val_losses_path):
        with open(val_losses_path) as f:
            val_losses_por_seed = {int(k): v for k, v in json.load(f).items()}
    else:
        val_losses_por_seed = {}

    for idx, seed in enumerate(pendientes, 1):
        print(f"\n{'─'*60}")
        print(f"Semilla {seed}  ({idx}/{len(pendientes)} pendientes)")
        print(f"{'─'*60}")
        rollout, por_capa, por_cabeza, best_val = entrenar_y_extraer_attention_seed(
            stack, seed, seq_length=seq_length, total_epochs=total_epochs,
            batch_size=batch_size, lr=lr, patience=patience, num_workers=num_workers,
            dates_millis=fechas,
        )
        out_path = os.path.join(DIR_MATRICES, f"attention_seed_{seed}.npy")
        np.save(out_path, rollout)
        # A1, A2 ... por separado (promediadas sobre cabezas) y sin promediar.
        np.save(os.path.join(DIR_MATRICES, f"attention_capas_seed_{seed}.npy"),
                por_capa)
        np.save(os.path.join(DIR_MATRICES, f"attention_cabezas_seed_{seed}.npy"),
                por_cabeza)
        print(f"  Guardado: {out_path}  shape={rollout.shape}  "
              f"range=[{rollout.min():.4f}, {rollout.max():.4f}]")
        print(f"            attention_capas_seed_{seed}.npy    "
              f"shape={por_capa.shape}")
        print(f"            attention_cabezas_seed_{seed}.npy  "
              f"shape={por_cabeza.shape}")
        # Necesario para el Paradigma 4 (I(G;Y)): antes no se guardaba en
        # ningún lado y el análisis de información mutua quedaba
        # permanentemente deshabilitado (val_loss constante = 0 para todas
        # las semillas, sin variabilidad que medir).
        val_losses_por_seed[seed] = best_val
        with open(val_losses_path, "w") as f:
            json.dump(val_losses_por_seed, f, indent=2)

    _resumen_capas_multisemilla(seeds)

    # Agregación Rashomon: se ejecuta aquí porque ya están las 50 matrices y los
    # val_losses, y no requiere nada mas.
    try:
        mats_r = cargar_matrices_cacheadas(seeds)
        if mats_r:
            calcular_eta(mats_r, verbose=False)
            rset = conjunto_rashomon(val_losses_por_seed)
            print(f"\n  Conjunto Rashomon (val_loss ≤ (1+{RASHOMON_EPSILON})·mejor): "
                  f"{len(rset)}/{len(seeds)} semillas")
            idx = [i for i, s in enumerate(seeds) if s in set(rset)] or list(range(len(mats_r)))
            res_r = importancia_rashomon([mats_r[i] for i in idx],
                                         [seeds[i] for i in idx])
            reportar_rashomon(res_r)
            with open(os.path.join(DIR_MATRICES, "rashomon.json"), "w",
                      encoding="utf-8") as f:
                json.dump(res_r, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"\n  Análisis Rashomon saltado: {type(e).__name__}: {e}")

    print(f"\n{'='*60}")
    print(f"PARTE 2b completa: {len(seeds)}/{len(seeds)} matrices en {DIR_MATRICES}")
    print(f"{'='*60}")


# ============================================================================
# FASE 1 · ESTABILIDAD CONTINUA DE LA ATENCIÓN
# (Etapa 1 del informe · Informe_Framework_Multicriterio_XAI_v2.pdf, §4-§5 ·
#  entregable "Métricas Lp, D_JS, H_i, ρ, Wasserstein")
#
# Responde a: ¿las matrices A^(m) de las M semillas representan el mismo
# fenómeno ANTES de discretizar? Si aquí ya divergen, cualquier grafo extraído
# en la PARTE 3 mide ruido, y las métricas posteriores no son interpretables.
#
# Convenciones de promediado (§4.2 del informe):
#   · métricas de similitud (Lp, ρ, τ, D_JS, Wasserstein) → promedio sobre los
#     C(M,2) pares
#   · métricas de dispersión (H, σ²)                      → promedio sobre las
#     M muestras
#   · métricas por fila                                   → se reporta media Y
#     mínimo (el mínimo es el cuello de botella del lema de estabilidad H3)
# ============================================================================

def _filas_candidatas(A: np.ndarray, drop_diagonal: bool = FASE1_DROP_DIAGONAL) -> np.ndarray:
    """Devuelve las filas de A como distribuciones sobre los destinos candidatos.

    El grafo extraído no admite bucles, así que la diagonal se elimina y la fila
    se RE-NORMALIZA sobre los N-1 candidatos restantes. Esto importa mucho con
    attention rollout: el término residual (0.5·I) infla la diagonal y, sin
    quitarla, la entropía y la D_JS quedan dominadas por una entrada que el
    operador de corte nunca va a seleccionar.
    """
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    if drop_diagonal:
        R = np.empty((n, n - 1), dtype=np.float64)
        for i in range(n):
            R[i] = np.delete(A[i], i)
    else:
        R = A.copy()
    R = np.clip(R, 0.0, None)
    s = R.sum(axis=1, keepdims=True)
    return np.where(s > 0.0, R / np.maximum(s, 1e-300), 1.0 / R.shape[1])


def normas_matriciales(A: np.ndarray, B: np.ndarray) -> Dict[str, float]:
    """Normas L1, L2 (Frobenius) y L_inf inducida entre dos matrices de atención (§5.1).

      · L1 total = Σ_ij |ΔA_ij| ∈ [0, 2N]  (el informe dice [0,2]: eso solo vale
        para la versión PROMEDIADA POR FILA, L1/N, que es la que se reporta como
        `l1_row` y sí vive en [0,2]).
      · L2 = ||ΔA||_F ∈ [0, √(2N)]  → para N=12, máx ≈ 4.899.
      · L_inf inducida = máx_i Σ_j |ΔA_ij| ∈ [0, 2]. Es la norma que se compara
        con el margen Δ_K en el lema H3 de la etapa 3.
    """
    D = np.abs(np.asarray(A, dtype=np.float64) - np.asarray(B, dtype=np.float64))
    n = D.shape[0]
    return {
        "l1_total": float(D.sum()),
        "l1_row": float(D.sum() / n),                 # ∈ [0, 2]
        "l2": float(np.sqrt(np.square(D).sum())),     # ∈ [0, √(2N)]
        "linf": float(D.sum(axis=1).max()),           # ∈ [0, 2]  (norma inducida)
    }


def divergencia_js_por_fila(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """D_JS entre las filas homólogas de dos matrices ya normalizadas (§5.3).

        D_JS(p‖q) = ½·D_KL(p‖m) + ½·D_KL(q‖m),   m = ½(p+q)

    Se usa `scipy.special.rel_entr`, que define 0·log(0/·) = 0, en lugar del
    truco `log(x + ε)` del informe: ese ε introduce un sesgo negativo en la
    divergencia y puede devolver valores fuera de [0, ln 2].
    UNIDADES: nats (logaritmo natural) → rango [0, ln 2] ≈ [0, 0.6931].
    """
    M = 0.5 * (P + Q)
    kl_pm = special.rel_entr(P, M).sum(axis=1)
    kl_qm = special.rel_entr(Q, M).sum(axis=1)
    return np.clip(0.5 * (kl_pm + kl_qm), 0.0, np.log(2.0))


def correlaciones_de_rango(A: np.ndarray, B: np.ndarray,
                           drop_diagonal: bool = FASE1_DROP_DIAGONAL
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """ρ de Spearman y τ de Kendall entre filas homólogas (§5.2).

    Se usan las implementaciones de scipy en vez de la fórmula cerrada
    1 - 6Σd²/(N(N²-1)) del informe: esa fórmula SOLO es válida sin empates, y
    las matrices de atención suelen tener entradas casi idénticas en la cola.
    Filas constantes → NaN (se propagan y se agregan con nanmean).
    """
    A = np.asarray(A, dtype=np.float64); B = np.asarray(B, dtype=np.float64)
    n = A.shape[0]
    rhos = np.full(n, np.nan); taus = np.full(n, np.nan)
    for i in range(n):
        a = np.delete(A[i], i) if drop_diagonal else A[i]
        b = np.delete(B[i], i) if drop_diagonal else B[i]
        if np.ptp(a) == 0.0 or np.ptp(b) == 0.0:
            continue
        rhos[i] = stats.spearmanr(a, b).statistic
        taus[i] = stats.kendalltau(a, b).statistic
    return rhos, taus


def wasserstein_por_fila(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """W_p (Earth Mover's Distance, 1D) entre filas homólogas ya normalizadas (§20.2).

    A diferencia de la D_JS, que trata las distribuciones como cantidades
    "estáticas", Wasserstein mide el esfuerzo de transporte: dos filas que
    difieren en una entrada ADYACENTE (posición j vs j+1) tienen W_p pequeño,
    mientras que dos que difieren en entradas lejanas tienen W_p grande, aunque
    la D_JS sea la misma. Se usa el orden de los índices como soporte 1D
    (posiciones 0..N-2 sobre los candidatos), vía `scipy.stats.wasserstein_distance`
    con las filas como pesos de esa distribución discreta.
    """
    n, m = P.shape
    support = np.arange(m, dtype=np.float64)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = stats.wasserstein_distance(support, support, u_weights=P[i], v_weights=Q[i])
    return out


def entropias_por_fila(A: np.ndarray,
                       drop_diagonal: bool = FASE1_DROP_DIAGONAL) -> Dict[str, np.ndarray]:
    """H_i de Shannon por fila (nats) y su ratio respecto al máximo ln(N_cand) (§5.4).

    r = H_i / ln(N_cand):  r > 0.95 → fila casi uniforme, el Top-P será
    esencialmente aleatorio;  r < 0.7 → fila afilada, márgenes suficientes.
    """
    P = _filas_candidatas(A, drop_diagonal)
    n_cand = P.shape[1]
    H = -special.xlogy(P, P).sum(axis=1)
    return {"H": H, "ratio": H / np.log(n_cand), "H_max": float(np.log(n_cand))}


def estadisticos_por_columna(A: np.ndarray) -> Dict[str, float]:
    """Entropía por columna y concentración máxima → detección de attention sinks (§5.4).

    Ā_col_j = (1/N)·Σ_i A_ij. Como A es fila-estocástica, Σ_j Ā_col_j = 1 exacto,
    así que ya es una distribución de probabilidad y no hace falta renormalizar.
    """
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    col = A.mean(axis=0)
    col = col / max(col.sum(), 1e-300)
    H_col = float(-special.xlogy(col, col).sum())
    return {
        "H_col": H_col,
        "H_col_ratio": H_col / float(np.log(n)),
        "C_col_max": float(col.max()),
        "col_mass": col,
        "sink_idx": int(np.argmax(col)),
    }


def varianza_por_posicion(matrices: List[np.ndarray]) -> Dict[str, np.ndarray]:
    """σ²_ij y CV_ij de cada entrada a lo largo de las M semillas (§5.5, ddof=1)."""
    stack = np.stack([np.asarray(m, dtype=np.float64) for m in matrices])
    mean = stack.mean(axis=0)
    var = stack.var(axis=0, ddof=1) if stack.shape[0] > 1 else np.zeros_like(mean)
    sd = np.sqrt(var)
    cv = sd / (mean + 1e-12)
    return {"mean": mean, "var": var, "std": sd, "cv": cv}


def _pares_de_semillas(M: int, max_pares: int = FASE1_MAX_PARES) -> List[Tuple[int, int]]:
    pares = list(itertools.combinations(range(M), 2))
    if max_pares and len(pares) > max_pares:
        rng = np.random.default_rng(SEMILLA_NULO)
        sel = rng.choice(len(pares), size=max_pares, replace=False)
        pares = [pares[k] for k in sorted(sel)]
    return pares


def analizar_estabilidad_continua(matrices: List[np.ndarray],
                                  labels: List[str],
                                  seeds: List[int],
                                  drop_diagonal: bool = FASE1_DROP_DIAGONAL,
                                  verbose: bool = True) -> Dict:
    """Ejecuta la Etapa 1 completa sobre las M matrices de atención."""
    M = len(matrices)
    n = matrices[0].shape[0]
    pares = _pares_de_semillas(M)

    # --- comprobación previa: ¿son fila-estocásticas? -----------------------
    sumas = np.array([np.asarray(A, dtype=np.float64).sum(axis=1) for A in matrices])
    desvio_estocastico = float(np.abs(sumas - 1.0).max())

    filas_norm = [_filas_candidatas(A, drop_diagonal) for A in matrices]

    l1_tot, l1_row, l2s, linfs = [], [], [], []
    linf_cand, eps_cand = [], []
    djs_pares, rho_pares, tau_pares, wass_pares = [], [], [], []
    djs_por_fila_acum = np.zeros(n); rho_por_fila_acum = np.zeros(n)
    tau_por_fila_acum = np.zeros(n); wass_por_fila_acum = np.zeros(n)
    cuenta_fila = np.zeros(n)

    for (a, b) in pares:
        nm = normas_matriciales(matrices[a], matrices[b])
        l1_tot.append(nm["l1_total"]); l1_row.append(nm["l1_row"])
        l2s.append(nm["l2"]);          linfs.append(nm["linf"])

        # Perturbación medida EN EL ESPACIO DONDE OCURRE EL CORTE: las filas ya
        # renormalizadas sobre los N-1 candidatos. Comparar el margen Δ (que vive
        # en ese espacio) con una L_inf calculada sobre la matriz cruda mezcla
        # dos escalas distintas y hace que el lema H3 nunca se cumpla.
        Dc = np.abs(filas_norm[a] - filas_norm[b])
        linf_cand.append(float(Dc.sum(axis=1).max()))   # L_inf inducida, candidatos
        eps_cand.append(float(Dc.max()))                # ε por entrada

        djs = divergencia_js_por_fila(filas_norm[a], filas_norm[b])
        djs_pares.append(float(djs.mean()))
        djs_por_fila_acum += djs

        rhos, taus = correlaciones_de_rango(matrices[a], matrices[b], drop_diagonal)
        rho_pares.append(float(np.nanmean(rhos)))
        tau_pares.append(float(np.nanmean(taus)))
        rho_por_fila_acum += np.nan_to_num(rhos)
        tau_por_fila_acum += np.nan_to_num(taus)

        wass = wasserstein_por_fila(filas_norm[a], filas_norm[b])
        wass_pares.append(float(wass.mean()))
        wass_por_fila_acum += wass

        cuenta_fila += 1.0

    djs_por_fila = djs_por_fila_acum / np.maximum(cuenta_fila, 1)
    rho_por_fila = rho_por_fila_acum / np.maximum(cuenta_fila, 1)
    tau_por_fila = tau_por_fila_acum / np.maximum(cuenta_fila, 1)
    wass_por_fila = wass_por_fila_acum / np.maximum(cuenta_fila, 1)

    # --- dispersión: se promedia sobre las M muestras, no sobre pares -------
    ent = [entropias_por_fila(A, drop_diagonal) for A in matrices]
    H_all = np.stack([e["H"] for e in ent])            # (M, n)
    ratio_all = np.stack([e["ratio"] for e in ent])
    col = [estadisticos_por_columna(A) for A in matrices]
    var_pos = varianza_por_posicion(matrices)

    # --- máscara fuera de la diagonal para resumir CV -----------------------
    off = ~np.eye(n, dtype=bool)
    cv_off = var_pos["cv"][off]

    res = {
        "n_seeds": M, "n_nodes": n, "n_pairs": len(pares),
        "labels": list(labels), "seeds": list(seeds),
        "drop_diagonal": bool(drop_diagonal),
        "desvio_estocastico": desvio_estocastico,
        # normas
        "l1_total_mean": float(np.mean(l1_tot)), "l1_total_max": float(np.max(l1_tot)),
        "l1_row_mean": float(np.mean(l1_row)),   "l1_row_max": float(np.max(l1_row)),
        "l2_mean": float(np.mean(l2s)),          "l2_max": float(np.max(l2s)),
        "linf_mean": float(np.mean(linfs)),      "linf_max": float(np.max(linfs)),
        "l2_max_teorico": float(np.sqrt(2 * n)),
        # divergencia y rangos
        "djs_mean": float(np.mean(djs_pares)),   "djs_max": float(np.max(djs_pares)),
        "djs_por_fila": djs_por_fila,
        "rho_mean": float(np.mean(rho_pares)),   "rho_min": float(np.min(rho_pares)),
        "rho_por_fila": rho_por_fila,
        "tau_mean": float(np.mean(tau_pares)),   "tau_min": float(np.min(tau_pares)),
        "tau_por_fila": tau_por_fila,
        # Wasserstein (§20.2, Tabla 7 etapa 1)
        "wasserstein_mean": float(np.mean(wass_pares)), "wasserstein_max": float(np.max(wass_pares)),
        "wasserstein_por_fila": wass_por_fila,
        # entropías
        "H_mean": float(H_all.mean()), "H_max_teorico": float(ent[0]["H_max"]),
        "ratio_H_mean": float(ratio_all.mean()),
        "ratio_H_por_fila": ratio_all.mean(axis=0),
        "ratio_H_max_fila": float(ratio_all.mean(axis=0).max()),
        "H_col_mean": float(np.mean([c["H_col"] for c in col])),
        "H_col_ratio_mean": float(np.mean([c["H_col_ratio"] for c in col])),
        "C_col_max_mean": float(np.mean([c["C_col_max"] for c in col])),
        "col_mass_mean": np.mean([c["col_mass"] for c in col], axis=0),
        "sink_counter": Counter([c["sink_idx"] for c in col]),
        # varianza por posición
        "mean_matrix": var_pos["mean"], "std_matrix": var_pos["std"],
        "cv_matrix": var_pos["cv"],
        "cv_off_mean": float(np.mean(cv_off)),
        "cv_off_frac_estable": float(np.mean(cv_off < 0.5)),
        "cv_off_frac_inestable": float(np.mean(cv_off > 1.0)),
        # ---- cota de Lipschitz para la etapa 3 (lema H3) -------------------
        # CORRECCIÓN: el informe (§5.1) pide "Δ_K > 2·L_inf". Eso cuenta el
        # factor 2 dos veces. El lema real es Δ_K > 2ε, con ε = máxima variación
        # POR ENTRADA; y para dos distribuciones vale ε ≤ ½·‖Δfila‖₁ = ½·L_inf
        # (desigualdad de variación total). Luego Δ_K > L_inf ya es suficiente,
        # y usar 2·L_inf es una cota gratuitamente conservadora que declara
        # inestables filas que no lo son. Aquí se reporta ε directamente.
        "linf_cand_mean": float(np.mean(linf_cand)),
        "linf_cand_max": float(np.max(linf_cand)),
        "eps_mean": float(np.mean(eps_cand)),
        "eps_max": float(np.max(eps_cand)),
        "cota_h3_media": float(2.0 * np.mean(eps_cand)),
        "cota_h3_peor_caso": float(2.0 * np.max(eps_cand)),
    }

    # Veredicto con los umbrales fijos de la Tabla 7, sin calibrar. El caller
    # (main_parte3) lo recalcula con calibrar_bateria_fase1 y deja los dos, para
    # que se vea qué métricas aprobaban sólo porque la atención es plana.
    # No se calibra aquí dentro: calibrar_bateria_fase1 vuelve a llamar a esta
    # misma función sobre cada nulo, y sería recursión infinita.
    res["veredicto_tabla7"] = _veredicto_fase1(res)
    res["veredicto"] = dict(res["veredicto_tabla7"])
    if verbose:
        _reporte_fase1(res, labels)
    return res


def _clasificar(valor: float, aprueba: float, falla: float, mayor_es_mejor: bool) -> str:
    if mayor_es_mejor:
        if valor >= aprueba: return "APRUEBA"
        if valor <= falla:   return "FALLA"
    else:
        if valor <= aprueba: return "APRUEBA"
        if valor >= falla:   return "FALLA"
    return "DUDOSO"


# Métricas de la FASE 1 con la clave que devuelve analizar_estabilidad_continua.
# `mayor_mejor` fija la dirección en la que se gana, y se usa para contar las
# excedencias del nulo en el sentido correcto.
METRICAS_FASE1: List[Tuple[str, str, bool]] = [
    ("D_JS",          "djs_mean",          False),
    ("Spearman",      "rho_mean",          True),
    ("Kendall",       "tau_mean",          True),
    ("Entropia_fila", "ratio_H_mean",      False),
    ("Wasserstein",   "wasserstein_mean",  False),
    ("L_inf",         "linf_mean",         False),
    ("CV_entrada",    "cv_off_mean",       False),
    ("Entropia_col",  "H_col_ratio_mean",  True),
    ("Sinks",         "C_col_max_mean",    False),
]


def calibrar_bateria_fase1(matrices: List[np.ndarray], observado: Dict,
                           n_nulos: int = None) -> Dict:
    """Las 9 métricas de la Tabla 7 contra el nulo UNIFORME.

    POR QUÉ. Cinco de las nueve son DISTANCIAS entre matrices (D_JS,
    Wasserstein, L_inf, CV por entrada) o dispersión de columna (Entropía_col,
    Sinks). Dos matrices planas están cerca entre sí y no tienen sumideros, así
    que a esas cinco las satisface la planitud y no la estructura. Medido el
    31/08 sobre las 20 matrices de Pencahue: el nulo uniforme APRUEBA D_JS,
    Wasserstein, CV_entrada, Entropía_col y Sinks, y en L_inf puntúa MEJOR que
    los datos reales (0.144 contra 0.315). Sólo Spearman y Kendall discriminan
    (0.554 y 0.446 reales contra -0.004 y -0.004 del nulo), porque son
    correlaciones de rango y no se pueden ganar aplanando.

    Sin esta calibración, un modelo de atención completamente plana saca ocho
    de nueve APRUEBA en la Tabla 7. Con ella, cada métrica tiene que ganarle a
    la planitud antes de contar como evidencia.

    Devuelve por métrica: observado, media/desviación del nulo, p empírico y
    `discrimina` (True si le gana al nulo con p < P_MAX_FASE1).
    """
    n_nulos = N_NULOS_FASE1 if n_nulos is None else int(n_nulos)
    rng = np.random.default_rng(SEMILLA_NULO)
    muestras: Dict[str, List[float]] = {c: [] for _, c, _ in METRICAS_FASE1}
    for _ in range(n_nulos):
        nul = generar_nulo_atencion(matrices, "uniforme", rng)
        rn = analizar_estabilidad_continua(nul, INDEX_NAMES,
                                           list(range(len(nul))),
                                           verbose=False)
        for _, clave, _ in METRICAS_FASE1:
            v = rn.get(clave)
            if v is not None and np.isfinite(v):
                muestras[clave].append(float(v))

    res: Dict[str, Dict] = {}
    for etq, clave, mayor_mejor in METRICAS_FASE1:
        vals = np.array(muestras[clave], dtype=float)
        o = observado.get(clave)
        if o is None or not np.isfinite(o) or vals.size == 0:
            continue
        o = float(o)
        n_ex = int((vals >= o).sum() if mayor_mejor else (vals <= o).sum())
        p_emp = (n_ex + 1) / (vals.size + 1)
        res[etq] = dict(
            clave=clave, observado=o, mayor_mejor=bool(mayor_mejor),
            nulo_media=float(vals.mean()), nulo_std=float(vals.std()),
            p_empirico=float(p_emp), n_excedencias=n_ex,
            n_nulos_validos=int(vals.size),
            discrimina=bool(p_emp < P_MAX_FASE1),
        )
    res["_n_nulos"] = n_nulos
    res["_nulo"] = "uniforme"
    return res


def reportar_bateria_fase1(cal: Dict) -> None:
    if not cal:
        return
    print("\n" + "=" * 88)
    print(f" BATERÍA DE LA FASE 1 CONTRA EL NULO UNIFORME "
          f"({cal.get('_n_nulos', '?')} realizaciones)")
    print(" Filas planas con el mismo ruido y cero estructura. Si una métrica no")
    print(" le gana, esa métrica no mide estructura: la satisface la planitud.")
    print("=" * 88)
    print(f"  {'métrica':<16}{'observado':>11}{'nulo mu':>11}{'nulo sd':>10}"
          f"{'p emp':>8}   discrimina")
    for etq, _, _ in METRICAS_FASE1:
        d = cal.get(etq)
        if not d:
            continue
        veredicto = "si" if d["discrimina"] else "NO - la pasa la planitud"
        print(f"  {etq:<16}{d['observado']:>11.4f}{d['nulo_media']:>11.4f}"
              f"{d['nulo_std']:>10.4f}{d['p_empirico']:>8.3f}   {veredicto}")
    n_d = sum(1 for e, _, _ in METRICAS_FASE1
              if cal.get(e, {}).get("discrimina"))
    print(f"  -> {n_d} de {len(METRICAS_FASE1)} métricas de la Tabla 7 "
          f"miden algo que la planitud no explique")
    print("=" * 88)


def _veredicto_fase1(r: Dict,
                     calibracion: Optional[Dict] = None) -> Dict[str, str]:
    """Compara cada métrica con los umbrales de la Tabla 7 (etapa 1).

    Con `calibracion` (de calibrar_bateria_fase1), una métrica que no le gane
    al nulo uniforme NO puede decir APRUEBA: baja a DUDOSO. Sin eso, una
    atención plana saca ocho de nueve aprobados.
    """
    u = UMBRALES_FASE1
    v = {
        "D_JS":        _clasificar(r["djs_mean"],          *u["djs"],         mayor_es_mejor=False),
        "Spearman":    _clasificar(r["rho_mean"],          *u["rho"],         mayor_es_mejor=True),
        "Kendall":     _clasificar(r["tau_mean"],          *u["tau"],         mayor_es_mejor=True),
        "Entropia_fila": _clasificar(r["ratio_H_mean"],    *u["ratio_H"],     mayor_es_mejor=False),
        "Wasserstein": _clasificar(r["wasserstein_mean"],  *u["wasserstein"], mayor_es_mejor=False),
        "L_inf":       _clasificar(r["linf_mean"],         *u["linf"],        mayor_es_mejor=False),
        "CV_entrada":  _clasificar(r["cv_off_mean"],       *u["cv"],          mayor_es_mejor=False),
        "Entropia_col": _clasificar(r["H_col_ratio_mean"], *u["hcol"],        mayor_es_mejor=True),
        "Sinks":       _clasificar(r["C_col_max_mean"],    *u["cmax"],        mayor_es_mejor=False),
    }
    # Una métrica que no le gana al nulo uniforme no puede aprobar: su APRUEBA
    # lo produce la planitud de la atención, no la estructura. Se degrada a
    # DUDOSO y se deja constancia en la clave _degradadas.
    # Una metrica que no le gana a su nulo queda EXCLUIDA del veredicto, no
    # degradada. Degradarla a DUDOSO la deja contando como evidencia parcial, y
    # no lo es: su valor lo produce la planitud de la atencion. Medido: cuatro
    # de las nueve puntuan MEJOR sobre ruido sin estructura que sobre los datos
    # reales (L_inf 0.239 nulo vs 0.315 real). Ver certificar_metricas.py.
    excluidas = []
    if calibracion:
        for etq, _, _ in METRICAS_FASE1:
            d = calibracion.get(etq)
            if d and not d.get("discrimina"):
                v[etq] = "NO_EVALUABLE"
                excluidas.append(etq)
    degradadas = excluidas

    # Compuerta dura: si la entropía por fila FALLA, la atención es plana y el
    # resto de la batería mide el ruido de una matriz sin forma. Aprobar la
    # etapa en ese caso es lo que dejaba pasar un modelo que no atiende a nada.
    plana = v.get("Entropia_fila") == "FALLA"

    dims = [x for k, x in v.items()
            if k != "GLOBAL" and not k.startswith("_")]
    n_falla = sum(1 for x in dims if x == "FALLA")
    n_duda = sum(1 for x in dims if x == "DUDOSO")
    n_aprueba = sum(1 for x in dims if x == "APRUEBA")
    n_excl = sum(1 for x in dims if x == "NO_EVALUABLE")

    # La compuerta de planitud SI se mantiene: si la atencion es uniforme, el
    # resto de la bateria describe el ruido de una matriz sin forma. Es la
    # unica condicion necesaria del eslabon, y no agrega criterios
    # incompatibles.
    if plana:
        v["GLOBAL"] = "ATENCION PLANA"
    elif not EMITIR_VEREDICTO_GLOBAL:
        # Caracterizacion, no aprobado. Agregar criterios que no pueden
        # cumplirse a la vez produce un numero que no significa nada.
        v["GLOBAL"] = (str(n_aprueba) + " aprueban / " + str(n_duda)
                       + " dudosas / " + str(n_falla) + " fallan / "
                       + str(n_excl) + " no evaluables de " + str(len(dims)))
    elif n_falla == 0 and n_duda <= 1:
        v["GLOBAL"] = "APRUEBA"
    elif n_falla >= 2:
        v["GLOBAL"] = "FALLA"
    else:
        v["GLOBAL"] = "DUDOSO"
    v["_conteo"] = dict(aprueba=n_aprueba, dudoso=n_duda, falla=n_falla,
                        no_evaluable=n_excl, total=len(dims))
    if degradadas:
        v["_excluidas_por_nulo"] = ", ".join(degradadas)
    if plana:
        v["_motivo_global"] = ("entropía por fila en FALLA: la atención es "
                               "plana y el resto de la batería no es "
                               "interpretable")
    return v


def _reporte_fase1(r: Dict, labels: List[str]) -> None:
    n = r["n_nodes"]
    print(f"\n{'═'*90}")
    print(f" FASE 1 · ESTABILIDAD CONTINUA DE LA ATENCIÓN (Etapa 1)")
    print(f" M = {r['n_seeds']} semillas · N = {n} nodos · {r['n_pairs']} pares comparados")
    print(f" Diagonal excluida: {r['drop_diagonal']}  (el grafo no admite bucles)")
    print(f"{'═'*90}")

    if r["desvio_estocastico"] > 1e-4:
        print(f"\n  AVISO: las filas no suman 1 exactamente "
              f"(desvío máx = {r['desvio_estocastico']:.2e}). "
              f"Las cotas de L1/L_inf y la D_JS asumen fila-estocasticidad.")

    print(f"\n[1] NORMAS MATRICIALES (promedio sobre los C(M,2) pares)")
    print(f"  L1 total       = {r['l1_total_mean']:.4f}  (máx par {r['l1_total_max']:.4f})   rango [0, {2*n}]")
    print(f"  L1 por fila    = {r['l1_row_mean']:.4f}  (máx par {r['l1_row_max']:.4f})   rango [0, 2]   ← comparable con el informe")
    print(f"  L2 (Frobenius) = {r['l2_mean']:.4f}  (máx par {r['l2_max']:.4f})   rango [0, {r['l2_max_teorico']:.3f}]")
    print(f"  L_inf inducida = {r['linf_mean']:.4f}  (máx par {r['linf_max']:.4f})   rango [0, 2]")
    print(f"\n  Sobre las filas renormalizadas a los N-1 candidatos (espacio del corte):")
    print(f"  L_inf candidatos = {r['linf_cand_mean']:.4f}  (máx par {r['linf_cand_max']:.4f})")
    print(f"  ε por entrada    = {r['eps_mean']:.4f}  (peor par {r['eps_max']:.4f})")
    print(f"  → lema H3: el Top-K se preserva si Δ_K > 2ε = {r['cota_h3_media']:.4f} "
          f"(peor caso {r['cota_h3_peor_caso']:.4f})")

    print(f"\n[2] DIVERGENCIA DE JENSEN-SHANNON (nats, rango [0, {np.log(2):.4f}])")
    print(f"  D_JS media = {r['djs_mean']:.4f}   máx entre pares = {r['djs_max']:.4f}")
    peor = int(np.argmax(r["djs_por_fila"]))
    print(f"  Fila más inestable: {labels[peor]} (D_JS = {r['djs_por_fila'][peor]:.4f})")

    print(f"\n[3] CORRELACIÓN DE RANGOS (¿se preserva la jerarquía de destinos?)")
    print(f"  Spearman ρ = {r['rho_mean']:.4f}  (peor par {r['rho_min']:.4f})")
    print(f"  Kendall  τ = {r['tau_mean']:.4f}  (peor par {r['tau_min']:.4f})")
    peor_rho = int(np.argmin(r["rho_por_fila"]))
    print(f"  Fila con jerarquía menos estable: {labels[peor_rho]} (ρ = {r['rho_por_fila'][peor_rho]:.4f})")

    print(f"\n[4] DISTANCIA DE WASSERSTEIN (Earth Mover's Distance, §20.2)")
    print(f"  W_p media = {r['wasserstein_mean']:.4f}   máx entre pares = {r['wasserstein_max']:.4f}")
    peor_w = int(np.argmax(r["wasserstein_por_fila"]))
    print(f"  Fila con mayor costo de transporte: {labels[peor_w]} (W_p = {r['wasserstein_por_fila'][peor_w]:.4f})")

    print(f"\n[5] ENTROPÍA POR FILA (¿hay destinos diferenciados?)")
    print(f"  H media = {r['H_mean']:.4f} nats de un máximo de {r['H_max_teorico']:.4f}")
    print(f"  ratio r = H/H_max = {r['ratio_H_mean']:.4f}   (r>0.95 → Top-P casi aleatorio)")
    planas = [labels[i] for i in range(n) if r["ratio_H_por_fila"][i] > 0.95]
    print(f"  Filas casi uniformes (r>0.95): {planas if planas else 'ninguna'}")

    print(f"\n[6] ENTROPÍA POR COLUMNA (detección de attention sinks)")
    print(f"  H_col/ln N = {r['H_col_ratio_mean']:.4f}   (>0.90 sano, <0.70 hay sinks)")
    print(f"  Concentración máx por columna = {r['C_col_max_mean']:.4f}   (debe ser < 0.20)")
    top_cols = np.argsort(-r["col_mass_mean"])[:3]
    print(f"  Columnas que más atención reciben: "
          + ", ".join(f"{labels[j]} ({r['col_mass_mean'][j]:.3f})" for j in top_cols))

    print(f"\n[7] VARIANZA POR POSICIÓN (diagnóstico granular, fuera de la diagonal)")
    print(f"  CV medio = {r['cv_off_mean']:.4f}")
    print(f"  Entradas estables (CV<0.5): {r['cv_off_frac_estable']*100:.1f}%   "
          f"inestables (CV>1.0): {r['cv_off_frac_inestable']*100:.1f}%")

    print(f"\n[8] VEREDICTO DE LA ETAPA 1 (umbrales de la Tabla 7)")
    for k, v in r["veredicto"].items():
        if k == "GLOBAL":
            continue
        print(f"  {k:<15s} {v}")
    print(f"  {'─'*40}")
    print(f"  {'GLOBAL':<15s} {r['veredicto']['GLOBAL']}")
    if r["veredicto"]["GLOBAL"] == "FALLA":
        print(f"  → La atención continua NO es estable entre semillas. El grafo que")
        print(f"    extraiga la PARTE 3 será inestable por construcción y sus")
        print(f"    p-valores describirán el ruido del operador, no el fenómeno.")
    elif r["veredicto"]["GLOBAL"] == "DUDOSO":
        print(f"  → Estabilidad marginal: se puede avanzar a la etapa 2, pero hay que")
        print(f"    reportar las métricas límite junto con los resultados.")
    else:
        print(f"  → Se puede avanzar a la discretización (etapa 2) y al paradigma 1.")


def plot_fase1(r: Dict, labels: List[str], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    n = r["n_nodes"]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    ax = axes[0, 0]
    im = ax.imshow(r["mean_matrix"], cmap="YlOrRd", aspect="equal")
    ax.set_title("Ā: atención media sobre M semillas", fontweight="bold")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[0, 1]
    cv_plot = np.array(r["cv_matrix"], dtype=float).copy()
    np.fill_diagonal(cv_plot, np.nan)
    im = ax.imshow(cv_plot, cmap="RdYlGn_r", vmin=0, vmax=1.5, aspect="equal")
    ax.set_title("CV_ij entre semillas\n(verde<0.5 estable · rojo>1.0 ruido)", fontweight="bold")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[0, 2]
    ax.bar(range(n), r["col_mass_mean"], color="#e67e22", alpha=0.85)
    ax.axhline(1.0 / n, color="green", ls="--", label=f"uniforme (1/N = {1/n:.3f})")
    ax.axhline(0.20, color="red", ls="--", label="umbral de sink (0.20)")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("masa de atención recibida")
    ax.set_title(f"Ā_col: attention sinks\nH_col/lnN = {r['H_col_ratio_mean']:.3f}", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 0]
    ax.barh(range(n), r["djs_por_fila"], color="#8e44ad", alpha=0.85)
    ax.axvline(0.10, color="green", ls="--", label="aprueba (<0.10)")
    ax.axvline(0.30, color="red", ls="--", label="falla (>0.30)")
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("D_JS media entre pares (nats)")
    ax.set_title("Estabilidad continua por fila", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="x")

    ax = axes[1, 1]
    ax.barh(range(n), r["rho_por_fila"], color="#2980b9", alpha=0.85)
    ax.axvline(0.8, color="green", ls="--", label="aprueba (>0.8)")
    ax.axvline(0.5, color="red", ls="--", label="falla (<0.5)")
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis(); ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel("Spearman ρ medio")
    ax.set_title("Preservación de jerarquía por fila", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="x")

    ax = axes[1, 2]
    ax.barh(range(n), r["ratio_H_por_fila"], color="#16a085", alpha=0.85)
    ax.axvline(0.90, color="green", ls="--", label="aprueba (<0.90)")
    ax.axvline(0.95, color="red", ls="--", label="falla (>0.95)")
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis(); ax.set_xlim(0, 1.05)
    ax.set_xlabel("r = H_i / ln(N-1)")
    ax.set_title("¿Hay destinos diferenciados?", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="x")

    plt.suptitle(f"FASE 1 · Estabilidad continua de la atención — "
                 f"{r['n_seeds']} semillas — veredicto: {r['veredicto']['GLOBAL']}",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fase1_estabilidad_continua.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figura: {output_dir}/fase1_estabilidad_continua.png")


def guardar_fase1(r: Dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    np.savez(
        os.path.join(output_dir, "fase1_estabilidad_continua.npz"),
        mean_matrix=r["mean_matrix"], std_matrix=r["std_matrix"], cv_matrix=r["cv_matrix"],
        djs_por_fila=r["djs_por_fila"], rho_por_fila=r["rho_por_fila"],
        tau_por_fila=r["tau_por_fila"], ratio_H_por_fila=r["ratio_H_por_fila"],
        wasserstein_por_fila=r["wasserstein_por_fila"],
        col_mass_mean=r["col_mass_mean"], labels=np.array(r["labels"]),
        seeds=np.array(r["seeds"]),
    )
    resumen = {k: v for k, v in r.items()
               if isinstance(v, (int, float, str, bool, list, dict))
               and not isinstance(v, Counter)}
    resumen["sink_counter"] = {str(k): int(v) for k, v in r["sink_counter"].items()}
    with open(os.path.join(output_dir, "fase1_resumen.json"), "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Datos:  {output_dir}/fase1_estabilidad_continua.npz")
    print(f"  Resumen: {output_dir}/fase1_resumen.json")


def cargar_matrices_cacheadas(seeds: List[int] = SEEDS,
                              n_indices: int = NUM_INDICES) -> Optional[List[np.ndarray]]:
    """Busca attention_seed_*.npy en las carpetas conocidas (SEARCH_DIRS).

    CORRECCIÓN: antes solo comprobaba que existieran los 50 archivos, sin
    validar su forma. Si en `SEARCH_DIRS` queda un cache de una corrida
    anterior con OTRO número de índices (p.ej. 6x6 de una versión vieja del
    pipeline), se cargaba igual y el mismatch reventaba más abajo, en un
    `ax.set_xticklabels` de un plot, con un traceback que no dice nada del
    problema real. Ahora se valida que cada matriz sea (n_indices, n_indices)
    antes de aceptar el directorio; si no calza, se avisa y se sigue buscando.
    """
    for d in SEARCH_DIRS:
        if not all(os.path.exists(os.path.join(d, f"attention_seed_{s}.npy")) for s in seeds):
            continue
        mats = [np.load(os.path.join(d, f"attention_seed_{s}.npy")) for s in seeds]
        malas = [m.shape for m in mats if m.shape != (n_indices, n_indices)]
        if malas:
            print(f"AVISO: cache en {d} tiene forma {malas[0]}, se esperaba "
                  f"({n_indices}, {n_indices}) — probablemente de una corrida "
                  f"vieja con otro nº de índices. Se ignora y se sigue buscando.")
            continue
        print(f"Matrices de atención encontradas en: {d}")
        return mats
    return None


def main_fase1() -> Optional[Dict]:
    matrices = cargar_matrices_cacheadas(SEEDS)
    if matrices is None:
        print("No se encontraron las matrices attention_seed_*.npy.")
        print(f"  Buscadas para {len(SEEDS)} semillas en:")
        for d in SEARCH_DIRS:
            print(f"    · {d}")
        return None
    r = analizar_estabilidad_continua(matrices, INDEX_NAMES, SEEDS)
    plot_fase1(r, INDEX_NAMES, DIR_FASE1)
    guardar_fase1(r, DIR_FASE1)
    return r


# ============================================================================
# TOP-P
# ============================================================================

def _select_top_p(row_values: np.ndarray,
                  p_threshold: float,
                  eta: float = 0.0) -> Tuple[np.ndarray, int, float]:
    """Top-P por fila con margen mínimo η opcional (§6.1, §7.1 del informe).

    Devuelve (índices seleccionados, k_eff, margen en el corte).

    CORRECCIONES respecto a la versión anterior:
      1. El criterio de parada es K = mín{k : Σ_{j≤k} p_(j) ≥ P} (§6.1). Antes
         se usaba `cum > P`, que en caso de igualdad exacta seleccionaba un
         destino de más.
      2. Se devuelve el margen Δ = p_(K) - p_(K+1), necesario para la etapa 3.
      3. Con η > 0 se implementa el operador adaptativo Top-P_η (§7.1): si el
         margen es menor que η se absorbe el siguiente elemento, de modo que
         Δ̃ ≥ η por construcción y no hay rank-flipping entre semillas.
    """
    n = len(row_values)
    if n == 0:
        return np.array([], dtype=int), 0, 0.0
    abs_vals = np.abs(row_values).astype(np.float64)
    sorted_order = np.argsort(-abs_vals, kind="stable")
    total = float(abs_vals.sum())
    if total > 0.0:
        probs = abs_vals[sorted_order] / total
    else:
        probs = np.ones(n, dtype=np.float64) / n
    cum_probs = np.cumsum(probs)
    mask = cum_probs >= (p_threshold - 1e-12)
    k_eff = int(np.argmax(mask)) + 1 if mask.any() else n
    k_eff = int(np.clip(k_eff, 1, n))

    def _margen(k: int) -> float:
        return float(probs[k - 1] - probs[k]) if k < n else 0.0

    if eta > 0.0:
        while k_eff < n and _margen(k_eff) < eta:
            k_eff += 1

    return sorted_order[:k_eff], k_eff, _margen(k_eff)


# ---------------------------------------------------------------------------
# DISCRETIZACIÓN ÚNICA DEL FRAMEWORK
#
# Todas las etapas (FASE 1, paradigmas 1 a 5, triangulación, PARTE 12) obtienen
# su grafo de aquí. No queda ningún Top-K en el pipeline.
# ---------------------------------------------------------------------------

def p_efectivo(p: Optional[float] = None) -> float:
    """P a usar. Prioridad: argumento explícito > P* del Paradigma 1 > constante.

    Leer el P* que eligió el Paradigma 1 es lo que garantiza que los cinco
    paradigmas discreticen con el MISMO valor. Si el JSON aún no existe (por
    ejemplo al correr el Paradigma 3 antes del 1), cae en P_DISCRETIZACION.
    """
    if p is not None:
        return float(p)
    ruta = os.path.join(DIR_P1, "paradigm1_results.json")
    if os.path.exists(ruta):
        try:
            with open(ruta, encoding="utf-8") as f:
                d = json.load(f)
            for clave in ("p_optimo", "top_p", "P_optimo"):
                if d.get(clave) is not None:
                    return float(d[clave])
        except Exception:
            pass
    return float(P_DISCRETIZACION)


def calcular_eta(matrices: List[np.ndarray], drop_diagonal: bool = True,
                 verbose: bool = True) -> float:
    """Fija ETA_MARGEN a partir de la distribución REAL de márgenes.

    El margen es Δ = p_(k) - p_(k+1) entre valores consecutivos de una fila ya
    normalizada. Con η constante en 0.01 y márgenes reales del orden de 0.006,
    el bucle de absorción de _select_top_p se comía la fila entera: k llegaba a
    10.6 de 11, P_azar = k/(N-1) subía a 0.97 y ninguna arista podía ser
    significativa. Anclar η a un percentil de los márgenes observados hace que
    el operador absorba solo los cortes de verdad ambiguos.
    """
    global ETA_MARGEN
    if ETA_MODO == "fijo" or not matrices:
        ETA_MARGEN = float(ETA_FIJO)
        if verbose:
            print(f"  η = {ETA_MARGEN:.6f} (modo fijo)")
        return ETA_MARGEN

    margenes = []
    for A in matrices:
        A = np.asarray(A, dtype=np.float64)
        n = A.shape[0]
        for i in range(n):
            fila = np.delete(np.abs(A[i]), i) if drop_diagonal else np.abs(A[i])
            v = np.sort(fila)[::-1]
            s = float(v.sum())
            probs = v / s if s > 0 else np.ones_like(v) / len(v)
            if len(probs) > 1:
                margenes.append(-np.diff(probs))      # p_(k) - p_(k+1) ≥ 0
    if not margenes:
        ETA_MARGEN = float(ETA_FIJO)
        return ETA_MARGEN

    margenes = np.concatenate(margenes)
    ETA_MARGEN = float(np.percentile(margenes, ETA_PERCENTIL))
    if verbose:
        print(f"  η adaptativo: percentil {ETA_PERCENTIL:.0f} de "
              f"{len(margenes)} márgenes observados → η = {ETA_MARGEN:.6f}")
        print(f"     percentiles de los márgenes: " + "  ".join(
            f"p{q}={np.percentile(margenes, q):.5f}" for q in (10, 25, 50, 75, 90)))
        print(f"     (η fijo anterior = {ETA_FIJO}; con él se absorbía el "
              f"{np.mean(margenes < ETA_FIJO)*100:.0f}% de los cortes)")
    return ETA_MARGEN


def discretizar(A: np.ndarray, p: Optional[float] = None,
                eta: Optional[float] = None) -> Set[Tuple[int, int]]:
    """A → conjunto de aristas (i, j) por Top-P_η, fila a fila, sin diagonal.

    Sustituye al antiguo topk_set(A, k). Devuelve exactamente el mismo tipo
    (set de tuplas de enteros), así que los sitios que lo usaban solo pierden
    el argumento K. La diagonal se excluye siempre: el grafo no tiene bucles y
    es la misma convención que usa top_p_mask_batch y FASE1_DROP_DIAGONAL.
    """
    p = p_efectivo(p)
    eta = ETA_MARGEN if eta is None else float(eta)
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    aristas: Set[Tuple[int, int]] = set()
    for i in range(n):
        cols = [j for j in range(n) if j != i]
        sel, _, _ = _select_top_p(A[i][cols], p, eta=eta)
        for s in sel:
            aristas.add((int(i), int(cols[int(s)])))
    return aristas


def generar_nulo_atencion(matrices: List[np.ndarray], tipo: str,
                          rng: np.random.Generator) -> List[np.ndarray]:
    """Ensemble nulo con el mismo tamaño y forma que `matrices`.

    Dos nulos, que responden a preguntas distintas:

    tipo="sin_estructura" — se permutan las entradas fuera de la diagonal DENTRO
        de cada fila, independientemente por matriz. Conserva exactamente el
        conjunto de valores de cada fila (así que entropía por fila, suma de
        fila y CV quedan idénticos) pero destruye QUÉ columna recibe qué valor.
        Responde: ¿hay algo reproducible entre modelos, o cualquier acuerdo es
        del azar?

    tipo="solo_columnas" — cada fila se regenera como el perfil de columna
        agregado más ruido, con la dispersión observada. Conserva la estructura
        de columnas (los sumideros) pero borra lo específico de cada fila.
        Responde: ¿hay MÁS que el perfil de columna? Es el nulo que importa
        aquí, porque medimos que la matriz era casi de rango 1 (A ≈ d·I + 1·cᵀ):
        si la métrica observada no supera a este nulo, lo único reproducible es
        el perfil de columna y no hay relación índice-a-índice.

    tipo="uniforme" — filas PLANAS fuera de la diagonal (todas las entradas
        (i,j) con i!=j valen lo mismo en esperanza) más el ruido observado,
        renormalizando la fila. Cero estructura, ni de fila ni de columna.
        Responde: ¿la métrica mide estructura, o la satisface la propia
        planitud de la atención?
        POR QUÉ HACÍA FALTA. Medido el 31/08 con las 20 matrices de Pencahue,
        este nulo APRUEBA cinco de los nueve umbrales de la Tabla 7 (D_JS,
        Wasserstein, CV_entrada, Entropía_col y Sinks) y en L_inf puntúa MEJOR
        que los datos reales (0.144 contra 0.315). No es casualidad: esas cinco
        son distancias entre matrices, y dos matrices planas están cerca entre
        sí. Sólo Spearman y Kendall distinguen (0.554 y 0.446 reales contra
        -0.004 y -0.004 del nulo), porque son correlaciones de rango y no se
        pueden ganar aplanando.
    """
    A = np.stack([np.asarray(m, dtype=np.float64) for m in matrices])
    M, N, _ = A.shape
    fuera = ~np.eye(N, dtype=bool)

    if tipo == "sin_estructura":
        out = []
        for m in range(M):
            B = A[m].copy()
            for i in range(N):
                cols = np.array([j for j in range(N) if j != i])
                B[i, cols] = A[m, i, rng.permutation(cols)]
            out.append(B)
        return out

    if tipo == "solo_columnas":
        # perfil de columna agregado (fuera de la diagonal) y dispersión residual
        col = np.array([A[:, fuera[:, j], j].mean() for j in range(N)])
        resid = np.array([[A[m, i, j] - col[j]
                           for j in range(N) for i in range(N) if i != j]
                          for m in range(M)])
        s = float(resid.std())
        out = []
        for m in range(M):
            B = A[m].copy()
            for i in range(N):
                cols = np.array([j for j in range(N) if j != i])
                v = col[cols] + rng.normal(0.0, s, size=len(cols))
                B[i, cols] = np.clip(v, 0.0, None)
            out.append(B)
        return out

    if tipo == "uniforme":
        # media fuera de la diagonal y CV entre modelos, para reproducir el
        # nivel de ruido sin reproducir NADA de la estructura
        base = float(A[:, fuera].mean())
        vals = np.stack([m[fuera] for m in A])
        cv = float(vals.std(0).mean() / max(abs(vals.mean()), 1e-12))
        out = []
        for m in range(M):
            B = np.full((N, N), base)
            B *= 1.0 + cv * rng.standard_normal((N, N))
            B = np.clip(B, 1e-9, None)
            np.fill_diagonal(B, np.diag(A[m]))
            # el rollout es fila-estocástico; se conserva esa propiedad para
            # que D_JS y las cotas de L1/L_inf sigan siendo comparables
            sf = A[m].sum(1)
            B = B / B.sum(1, keepdims=True) * sf[:, None]
            out.append(B)
        return out

    raise ValueError(f"tipo de nulo no reconocido: {tipo!r}")


def calibrar_metricas_estabilidad(matrices: List[np.ndarray],
                                  observado: Dict[str, float],
                                  n_nulos: int = None,
                                  max_pares: int = None) -> Dict:
    """Compara las métricas de estabilidad observadas contra los dos nulos.

    `observado` debe traer las claves que se quieran calibrar, con los mismos
    nombres que devuelve analizar_estabilidad_continua (rho_mean, tau_mean,
    djs_mean...). Devuelve, por métrica, la media y desviación del nulo, el
    Z-score y el percentil del observado dentro del nulo.
    """
    n_nulos = N_NULOS_UMBRAL if n_nulos is None else int(n_nulos)
    max_pares = MAX_PARES_CALIBRACION if max_pares is None else int(max_pares)
    rng = np.random.default_rng(SEMILLA_NULO)
    M = len(matrices)
    pares_todos = [(i, j) for i in range(M) for j in range(i + 1, M)]

    def _metricas(mats, pares):
        rho, tau, djs = [], [], []
        for a, b in pares:
            A, B = np.asarray(mats[a], float), np.asarray(mats[b], float)
            n = A.shape[0]
            for i in range(n):
                msk = np.ones(n, dtype=bool); msk[i] = False
                x, y = A[i][msk], B[i][msk]
                if np.std(x) > 1e-12 and np.std(y) > 1e-12:
                    r = stats.spearmanr(x, y).statistic
                    t = stats.kendalltau(x, y).statistic
                    if np.isfinite(r): rho.append(r)
                    if np.isfinite(t): tau.append(t)
                p = np.clip(x, 1e-12, None); p /= p.sum()
                q = np.clip(y, 1e-12, None); q /= q.sum()
                mm = 0.5 * (p + q)
                djs.append(float(0.5 * np.sum(p * np.log(p / mm))
                                 + 0.5 * np.sum(q * np.log(q / mm))))
        return dict(rho_mean=float(np.mean(rho)) if rho else np.nan,
                    tau_mean=float(np.mean(tau)) if tau else np.nan,
                    djs_mean=float(np.mean(djs)) if djs else np.nan)

    pares = pares_todos
    if max_pares and len(pares) > max_pares:
        idx = rng.choice(len(pares), max_pares, replace=False)
        pares = [pares_todos[k] for k in idx]

    obs_recalc = _metricas(matrices, pares)
    res = {}
    for tipo in ("sin_estructura", "solo_columnas", "uniforme"):
        muestras = {k: [] for k in obs_recalc}
        for _ in range(n_nulos):
            nul = generar_nulo_atencion(matrices, tipo, rng)
            mm = _metricas(nul, pares)
            for k, v in mm.items():
                if np.isfinite(v):
                    muestras[k].append(v)
        res[tipo] = {}
        for k, vals in muestras.items():
            if not vals:
                continue
            v = np.array(vals)
            # El observado que se compara es el RECALCULADO sobre los mismos
            # `pares` y con el mismo código que los nulos. Antes se prefería el
            # valor de la FASE 1, que usa los C(M,2)=1225 pares completos: eso
            # comparaba un promedio de 1225 pares contra nulos de 120, y las dos
            # cifras difieren (0.2887 vs 0.3145 en la corrida del 31/08). Aunque
            # el sesgo era conservador, no es la misma cantidad.
            o = float(obs_recalc.get(k, observado.get(k, np.nan)))
            alto = (k != "djs_mean")            # D_JS: mejor es MÁS BAJO
            n_ex = int((v >= o).sum() if alto else (v <= o).sum())
            res[tipo][k] = dict(
                observado=o,
                observado_fase1=float(observado.get(k, np.nan)),
                nulo_media=float(v.mean()), nulo_std=float(v.std()),
                z=float((o - v.mean()) / max(v.std(), 1e-12)),
                percentil=float((v < o).mean() * 100.0),
                # Razón observado/nulo: el tamaño de efecto citable.
                razon=float(o / v.mean()) if abs(v.mean()) > 1e-12 else float("nan"),
                # p empírico de una cola con corrección de continuidad. Con 40
                # nulos y cero excedencias el mínimo alcanzable es 1/41=0.024:
                # ése es el límite real de resolución, no el Z de dos cifras.
                p_empirico=float((n_ex + 1) / (len(v) + 1)),
                n_excedencias=n_ex, n_nulos_validos=int(len(v)),
            )
    res["_observado_recalculado"] = obs_recalc
    res["_n_pares"] = len(pares)
    res["_n_nulos"] = n_nulos
    return res


def reportar_calibracion(cal: Dict) -> None:
    if not cal:
        return
    ETQ = {"rho_mean": "Spearman medio por fila",
           "tau_mean": "Kendall medio por fila",
           "djs_mean": "D_JS media entre pares"}
    MEJOR_ALTO = {"rho_mean": True, "tau_mean": True, "djs_mean": False}
    print(f"\n{'='*82}")
    print(f" UMBRALES CALIBRADOS CONTRA MODELO NULO")
    print(f" {cal['_n_nulos']} realizaciones nulas · {cal['_n_pares']} pares por realización")
    print(f"{'='*82}")
    print("  Nulo 'sin_estructura' : permuta las columnas dentro de cada fila.")
    print("     Conserva los valores de la fila, borra a QUIÉN se atiende.")
    print("  Nulo 'solo_columnas'  : regenera cada fila desde el perfil de")
    print("     columna agregado. Conserva los sumideros, borra lo específico")
    print("     de cada fila. ES EL QUE IMPORTA: si el observado no lo supera,")
    print("     lo único reproducible es el perfil de columna.")
    for tipo in ("sin_estructura", "solo_columnas", "uniforme"):
        if tipo not in cal:
            continue
        print(f"\n  --- nulo: {tipo} ---")
        print(f"  {'métrica':<26}{'observado':>11}{'nulo μ':>10}{'razón':>8}"
              f"{'p emp':>8}{'Z':>9}   veredicto")
        for k, d in cal[tipo].items():
            # El veredicto va por el p EMPÍRICO, no por el Z. El Z está
            # inflado: la métrica es un promedio de cientos de filas y eso
            # encoge la σ del nulo. El p empírico cuenta excedencias.
            p_emp = d.get("p_empirico", float("nan"))
            supera = p_emp < 0.05
            ver = ("SUPERA AL NULO" if supera else
                   "indistinguible" if p_emp > 0.20 else "dudoso")
            raz = d.get("razon", float("nan"))
            # Si la media del nulo roza el cero la razón se dispara y no
            # significa nada (el nulo sin_estructura da ρ≈0.0003 → 900x).
            praz = (f"{raz:>8.2f}" if abs(d["nulo_media"]) > 0.02
                    else f"{'>>':>8}")
            print(f"  {ETQ.get(k, k):<26}{d['observado']:>11.4f}"
                  f"{d['nulo_media']:>10.4f}{praz}"
                  f"{p_emp:>8.3f}{d['z']:>+9.2f}   {ver}")
            of1 = d.get("observado_fase1")
            if of1 is not None and np.isfinite(of1) and \
                    abs(of1 - d["observado"]) > 1e-6:
                print(f"  {'':<26}(FASE 1 sobre todos los pares: "
                      f"{of1:.4f}; aquí se usa el recalculado sobre los "
                      f"mismos {cal['_n_pares']} pares del nulo)")
    print(f"\n  CÓMO LEER ESTO, Y QUÉ NO LEER.")
    print(f"  El Z sale enorme (decenas) porque cada métrica es un promedio de")
    print(f"  {cal['_n_pares']} pares × N filas, y promediar encoge la desviación del")
    print(f"  nulo. NO reportes el Z como si fuera un tamaño de efecto: el número")
    print(f"  que vale es la RAZÓN observado/nulo, y el signo de la comparación.")
    sc = cal.get("solo_columnas", {})
    if "rho_mean" in sc:
        d = sc["rho_mean"]
        raz = d.get("razon") or d["observado"] / max(abs(d["nulo_media"]), 1e-9)
        p_emp = d.get("p_empirico", float("nan"))
        print(f"\n  Spearman observado {d['observado']:.3f} vs "
              f"{d['nulo_media']:.3f} del nulo de solo-columnas = "
              f"{raz:.1f}x, p empírico {p_emp:.3f} "
              f"({d.get('n_excedencias', '?')} de "
              f"{d.get('n_nulos_validos', '?')} nulos lo igualan).")
        print(f"  Eso es lo que hay que citar: hay estructura fila-específica")
        print(f"  reproducible, o sea relación entre índices y no solo sumideros.")
        print(f"  Y es una prueba más exigente que el umbral fijo de la Tabla 7,")
        print(f"  que con ρ<0.5 daba FALLA sin preguntarse contra qué.")
        print(f"  CUIDADO CON LA RAZÓN: entre dos entrenamientos distintos")
        print(f"  del 31/08 pasó de 5.2x (0.404/0.078) a {raz:.1f}x "
              f"({d['observado']:.3f}/{d['nulo_media']:.3f}).")
        print(f"  El signo y el p aguantan; la MAGNITUD no es estable, así")
        print(f"  que en la tesis va el p empírico y no el múltiplo.")
    print(f"{'='*82}")


def conjunto_rashomon(val_losses: Dict[int, float],
                      epsilon: float = None) -> List[int]:
    """Semillas cuyo error está dentro de (1+ε) del mejor: el conjunto Rashomon.

    Un conjunto Rashomon es el grupo de modelos casi igual de buenos: los que
    caen dentro de (1+ε) del mejor error. Las 50 semillas correlacionan
    r = 0.9955 entre predicciones, así que funcionalmente son intercambiables.

    OJO CON ε. El CoV del MSE de validación medido el 31/08 es 0.077, no el
    0.0096 que decía antes este docstring. Con ε=0.05 el conjunto se quedaba
    en 4 de 50 semillas, y sobre 4 modelos el IC al 90% de importancia_rashomon
    es un percentil de cuatro puntos: sin resolución. Por eso, si el conjunto
    sale más pequeño que M_MIN se ensancha ε hasta alcanzarlo, y se avisa. Con
    los datos del 31/08: ε=0.05 → 4, ε=0.10 → 14, ε=0.15 → 25, ε=0.20 → 40.
    """
    epsilon = RASHOMON_EPSILON if epsilon is None else float(epsilon)
    if not val_losses:
        return []
    mejor = min(val_losses.values())

    def _sel(eps):
        return sorted(int(k) for k, v in val_losses.items()
                      if v <= mejor * (1.0 + eps))

    sel = _sel(epsilon)
    m_min = min(RASHOMON_M_MIN, len(val_losses))
    if len(sel) < m_min:
        eps = epsilon
        while len(sel) < m_min and eps < 2.0:
            eps += 0.01
            sel = _sel(eps)
        print(f"  Conjunto Rashomon: ε={epsilon:.2f} sólo daba "
              f"{len(_sel(epsilon))} de {len(val_losses)} modelos; se ensancha "
              f"a ε={eps:.2f} para llegar a {len(sel)} y que el IC del "
              f"percentil tenga sentido.")
    return sel


def importancia_rashomon(matrices: List[np.ndarray], seeds: List[int],
                         p: Optional[float] = None,
                         nivel: float = 0.90) -> Dict:
    """Importancia de cada arista AGREGADA sobre el conjunto Rashomon.

    POR QUÉ ESTO EN LUGAR DE LA FRECUENCIA. El framework mide hoy si la matriz
    de atención es ESTABLE entre semillas (FASE 1: Spearman, Kendall) y si una
    arista APARECE con frecuencia sobre el azar (Paradigma 1: Φ(e) contra un
    nulo). Ambas cosas tratan la variabilidad entre modelos como ruido a
    eliminar. Pero aquí medimos que los 50 modelos son funcionalmente
    equivalentes (r = 0.9955) y sin embargo sus atenciones discrepan
    (ρ_role = 0.247). Eso no es ruido de estimación: es el efecto Rashomon —
    modelos equivalentes admiten explicaciones distintas.

    La respuesta de la literatura (Rashomon Importance Distribution,
    arXiv:2309.13775, NeurIPS) no es elegir un modelo ni exigir que todos
    coincidan, sino reportar la DISTRIBUCIÓN de la importancia sobre el
    conjunto, con su incertidumbre. Una arista con percentil alto y estrecho es
    una conclusión robusta; una con percentil ancho es genuinamente ambigua, y
    decirlo es más honesto que descartarla por "inestable".

    Para cada arista devuelve:
      peso_*     — estadísticos del valor de atención a_ij sobre el conjunto
      pct_*      — percentil de la arista DENTRO de su fila (invariante a la
                   escala de cada modelo, que es lo que hace comparables las
                   semillas entre sí)
      ic_pct     — intervalo de confianza empírico del percentil
      robusta    — el IC completo por encima de RASHOMON_PCT_MIN
    """
    if not matrices:
        return {}
    A = np.stack([np.asarray(m, dtype=np.float64) for m in matrices])   # (M,N,N)
    M, N, _ = A.shape
    lo_q, hi_q = (1 - nivel) / 2 * 100, (1 + nivel) / 2 * 100

    # percentil de cada arista dentro de su fila, por modelo
    pct = np.zeros_like(A)
    for m in range(M):
        for i in range(N):
            cols = [j for j in range(N) if j != i]
            v = A[m, i, cols]
            orden = np.argsort(np.argsort(v))          # rango 0..len-1
            pct[m, i, cols] = orden / max(len(cols) - 1, 1)

    sel = np.stack([np.isin(np.arange(N * N).reshape(N, N),
                            [i * N + j for (i, j) in discretizar(A[m], p)])
                    for m in range(M)])                # (M,N,N) bool

    aristas = []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            w, q = A[:, i, j], pct[:, i, j]
            ic = (float(np.percentile(q, lo_q)), float(np.percentile(q, hi_q)))
            aristas.append(dict(
                i=i, j=j, label_i=INDEX_NAMES[i], label_j=INDEX_NAMES[j],
                peso_media=float(w.mean()), peso_std=float(w.std()),
                pct_media=float(q.mean()), pct_std=float(q.std()),
                ic_pct=[ic[0], ic[1]], ancho_ic=float(ic[1] - ic[0]),
                phi=float(sel[:, i, j].mean()),
                robusta=bool(ic[0] >= RASHOMON_PCT_MIN),
            ))
    aristas.sort(key=lambda r: -r["pct_media"])
    return dict(M=M, seeds=list(seeds), nivel=nivel,
                pct_min=RASHOMON_PCT_MIN, aristas=aristas,
                n_robustas=int(sum(1 for r in aristas if r["robusta"])))


def reportar_rashomon(res: Dict, top: int = 15) -> None:
    if not res:
        print("  (sin matrices para el análisis Rashomon)")
        return
    print(f"\n{'='*82}")
    print(f" IMPORTANCIA AGREGADA SOBRE EL CONJUNTO RASHOMON  ({res['M']} modelos)")
    print(f"{'='*82}")
    print(f"  En vez de exigir que las {res['M']} atenciones coincidan, se reporta la")
    print(f"  distribución del percentil de cada arista dentro de su fila, con IC")
    print(f"  al {res['nivel']*100:.0f}%. Robusta = IC completo por encima de "
          f"{res['pct_min']:.2f}.")
    print(f"\n  {'arista':<28}{'pct medio':>11}{'IC':>18}{'ancho':>8}{'Φ':>7}  robusta")
    for r in res["aristas"][:top]:
        ic = f"[{r['ic_pct'][0]:.2f}, {r['ic_pct'][1]:.2f}]"
        print(f"  {r['label_i']+'→'+r['label_j']:<28}{r['pct_media']:>11.3f}"
              f"{ic:>18}{r['ancho_ic']:>8.2f}{r['phi']:>7.2f}"
              f"  {'sí' if r['robusta'] else 'no'}")
    print(f"\n  Aristas robustas: {res['n_robustas']}/{len(res['aristas'])}")
    anchos = np.array([r["ancho_ic"] for r in res["aristas"]])
    print(f"  Ancho medio del IC del percentil: {anchos.mean():.3f}")
    print(f"  (IC ancho = la arista es genuinamente ambigua en el conjunto")
    print(f"   Rashomon, no 'inestable por poco M')")
    print(f"{'='*82}")


def tamano_medio_discretizacion(matrices: List[np.ndarray],
                                p: Optional[float] = None) -> float:
    """Aristas medias por fila tras discretizar. Para reportar la densidad."""
    if not matrices:
        return 0.0
    n = matrices[0].shape[0]
    return float(np.mean([len(discretizar(A, p)) / n for A in matrices]))


@dataclass
class FrameworkAnalysis:
    seed: int; matrix: np.ndarray; top_p: float
    svd_sigma1: float; svd_sigma2: float; svd_sigma3: float
    effective_rank: int; condition_number: float; nuclear_norm: float
    trace_A: float; trace_A2: float; trace_A3: float
    entropy: float; symmetry_ratio: float; spectral_gap: float
    laplacian_eigvals: List[float]
    n_edges: int; density: float; reciprocity: float
    pagerank_top3: List[str]; hubs: List[str]; authorities: List[str]
    communities: List[List[str]]; modularity: float
    node_roles: Dict[str, str]
    hypothesis_edges: List[Tuple[str, str, float]]
    hypothesis_signature: frozenset

def run_framework_pipeline(matrix: np.ndarray, labels: List[str],
                           seed: int, top_p: float) -> FrameworkAnalysis:
    n = len(labels); A = matrix.copy().astype(np.float64)

    U, S, Vt = np.linalg.svd(A)
    sigma1, sigma2, sigma3 = float(S[0]), float(S[1]) if len(S)>1 else 0., float(S[2]) if len(S)>2 else 0.
    energy = S**2; cumul = np.cumsum(energy)/max(energy.sum(), 1e-12)
    effective_rank = int(np.searchsorted(cumul, 0.95)+1)
    condition_number = float(S[0]/max(S[-1], 1e-12))
    nuclear_norm = float(S.sum())
    A2, A3 = A@A, A@A@A
    trace_A, trace_A2, trace_A3 = float(np.trace(A)), float(np.trace(A2)), float(np.trace(A3))
    # CORRECCIÓN: entropía por fila (§5.4). Antes se sumaba sobre TODA la matriz,
    # lo que da un valor en [0, N·lnN] y no es la H_i del informe. Ahora se
    # promedia sobre las filas → rango [0, ln N], comparable con los umbrales.
    P = A / np.maximum(A.sum(axis=1, keepdims=True), 1e-12)
    entropy = float(np.mean(-special.xlogy(P, P).sum(axis=1)))
    S_sym, K_anti = (A+A.T)/2, (A-A.T)/2
    symmetry_ratio = float(np.linalg.norm(K_anti,'fro')/max(np.linalg.norm(S_sym,'fro'),1e-12))

    # CORRECCIÓN: el Laplaciano del informe es L = D - A y su λ₂ (conectividad
    # algebraica) solo está definido para grafos no dirigidos. Con A asimétrica,
    # `np.eye(n) - A` tiene autovalores complejos y quedarse con la parte real
    # tras ordenar no es la brecha espectral de nada. Se simetriza primero,
    # se usa el grado real como D y eigvalsh (espectro real garantizado, λ₁=0).
    A_sym = (A + A.T) / 2.0
    L = np.diag(A_sym.sum(axis=1)) - A_sym
    laplacian_eigvals = np.linalg.eigvalsh(L).tolist()
    spectral_gap = float(laplacian_eigvals[1]) if len(laplacian_eigvals)>1 else 0.
    top_edges = []
    top_p_sizes_per_row: List[int] = []
    top_p_margins_per_row: List[float] = []
    for i in range(n):
        candidate_mask = np.ones(n, dtype=bool)
        candidate_mask[i] = False
        candidate_vals = A[i, candidate_mask].astype(np.float64)
        candidate_idx = np.where(candidate_mask)[0]

        selected_local, k_eff, margen = _select_top_p(candidate_vals, top_p, eta=ETA_MARGEN)
        selected_j = candidate_idx[selected_local]

        top_p_sizes_per_row.append(int(k_eff))
        top_p_margins_per_row.append(float(margen))
        for j in selected_j:
            top_edges.append((i, int(j), float(A[i, int(j)])))

    # CORRECCIÓN: la clave era solo `seed`, así que cada P sobrescribía los
    # tamaños del P anterior y lo que se guardaba al final en el .npz eran
    # siempre los del último P del barrido. Ahora la clave es (seed, P).
    GLOBAL_TOP_P_SIZES[(seed, float(top_p))] = top_p_sizes_per_row

    G = nx.DiGraph()
    for i in range(n): G.add_node(i, label=labels[i])
    for u,v,w in top_edges: G.add_edge(u,v,weight=w)
    density = G.number_of_edges()/max(n*(n-1),1)
    try: reciprocity = float(nx.reciprocity(G))
    except: reciprocity = 0.0

    try:
        pr = nx.pagerank(G, weight="weight")
        pagerank_top3 = [labels[i] for i,_ in sorted(pr.items(), key=lambda x:-x[1])[:3]]
    except: pagerank_top3 = []
    try:
        hubs_d, auth_d = nx.hits(G, weight="weight")
        hubs = [labels[i] for i,_ in sorted(hubs_d.items(), key=lambda x:-x[1])[:3]]
        authorities = [labels[i] for i,_ in sorted(auth_d.items(), key=lambda x:-x[1])[:3]]
    except: hubs, authorities = [], []
    try:
        undirected = G.to_undirected()
        communities = list(nx.community.louvain_communities(undirected, weight="weight", seed=42))
        communities = [[labels[nd] for nd in c] for c in communities]
        modularity = float(nx.community.modularity(undirected, communities, weight="weight"))
    except: communities, modularity = [], 0.0

    in_deg = dict(G.in_degree(weight="weight"))
    out_deg = dict(G.out_degree(weight="weight"))
    try: betw = nx.betweenness_centrality(G, weight="weight")
    except: betw = {i:0 for i in range(n)}
    in_t = np.mean(list(in_deg.values())) if in_deg else 0
    out_t = np.mean(list(out_deg.values())) if out_deg else 0
    betw_t = np.mean(list(betw.values())) if betw else 0
    node_roles = {}
    for i in range(n):
        ih = in_deg.get(i,0)>in_t; oh = out_deg.get(i,0)>out_t; ib = betw.get(i,0)>betw_t
        if ih and oh: node_roles[labels[i]] = "hub_source"
        elif ih: node_roles[labels[i]] = "hub"
        elif oh: node_roles[labels[i]] = "source"
        elif ib: node_roles[labels[i]] = "bridge"
        else: node_roles[labels[i]] = "isolated"

    hypothesis_edges = [(labels[u], labels[v], w) for u,v,w in top_edges]
    hypothesis_signature = frozenset((u,v) for u,v,_ in top_edges)

    return FrameworkAnalysis(
        seed=seed, matrix=matrix, top_p=top_p,
        svd_sigma1=sigma1, svd_sigma2=sigma2, svd_sigma3=sigma3,
        effective_rank=effective_rank, condition_number=condition_number,
        nuclear_norm=nuclear_norm, trace_A=trace_A, trace_A2=trace_A2, trace_A3=trace_A3,
        entropy=entropy, symmetry_ratio=symmetry_ratio, spectral_gap=spectral_gap,
        laplacian_eigvals=laplacian_eigvals,
        n_edges=G.number_of_edges(), density=density, reciprocity=reciprocity,
        pagerank_top3=pagerank_top3, hubs=hubs, authorities=authorities,
        communities=communities, modularity=modularity, node_roles=node_roles,
        hypothesis_edges=hypothesis_edges, hypothesis_signature=hypothesis_signature)

def compute_jaccard_matrix(analyses: List[FrameworkAnalysis]) -> np.ndarray:
    n = len(analyses)
    sims = []
    for i in range(n):
        for j in range(i+1, n):
            s1, s2 = analyses[i].hypothesis_signature, analyses[j].hypothesis_signature
            union = len(s1|s2)
            sims.append(len(s1&s2)/union if union > 0 else 0.)
    return np.array(sims)

def compute_edge_frequency(analyses: List[FrameworkAnalysis], labels: List[str]) -> Counter:
    freq = Counter()
    for a in analyses:
        for edge in a.hypothesis_signature:
            freq[edge] += 1
    return freq

def compute_core_periphery_decomposition(edge_freq: Counter, n_models: int,
                                          labels: List[str],
                                          stability_thresh: float = UMBRAL_DE_ESTABILIDAD):
    core = {(u,v): f for (u,v), f in edge_freq.items() if f/n_models >= stability_thresh}
    periphery = {(u,v): f for (u,v), f in edge_freq.items() if f/n_models < stability_thresh}
    return core, periphery

def _avg_top_p_size(seeds: List[int], top_p: float) -> float:
    """Tamaño medio del top por fila para un P concreto (clave (seed, P))."""
    all_sizes: List[int] = []
    for s in seeds:
        sizes = GLOBAL_TOP_P_SIZES.get((s, float(top_p)))
        if sizes:
            all_sizes.extend(sizes)
    if not all_sizes:
        return 1.0
    return float(np.mean(all_sizes))

# ============================================================================
# PARADIGMA 1 · PROBABILÍSTICO  (Etapas 3 y 4 del informe · §7-§8)
#
#   Pregunta: "¿SE REPITE?" — ¿la persistencia Φ(e) de una arista es explicable
#   por azar, o excede lo que produciría el operador de corte sobre una matriz
#   sin estructura?
#
#   Cadena implementada:
#     (a) margen del corte Δ y operador adaptativo Top-P_η
#     (b) persistencia Φ(e) sobre M semillas
#     (c) modelo nulo analítico + test exacto (Poisson-binomial ⊃ binomial)
#     (d) corrección por comparaciones múltiples: Bonferroni y Benjamini-Hochberg
#     (e) intervalo de confianza exacto de Clopper-Pearson para Φ(e)
#     (f) test de permutaciones (no paramétrico) + control FWER tipo max-T
#     (g) Z-score frente al grafo aleatorio con out-grado restringido
#     (h) grafo de consenso + compacidad (sparsity, cobertura)
# ============================================================================

# ----------------------------------------------------------------------------
# (a) Discretización vectorizada con margen
# ----------------------------------------------------------------------------

def probs_candidatas_batch(A_batch: np.ndarray) -> np.ndarray:
    """(..., N, N) → probabilidades sobre los N-1 candidatos (diagonal a 0).

    Es la versión vectorizada, y numéricamente idéntica, de `_select_top_p`:
    se elimina la diagonal y se renormaliza la fila sobre los destinos posibles.
    """
    X = np.abs(np.asarray(A_batch, dtype=np.float64)).copy()
    n = X.shape[-1]
    idx = np.arange(n)
    X[..., idx, idx] = 0.0
    tot = X.sum(axis=-1, keepdims=True)
    P = np.where(tot > 0.0, X / np.maximum(tot, 1e-300), 1.0 / (n - 1))
    P[..., idx, idx] = 0.0
    return P


def top_p_mask_batch(A_batch: np.ndarray, p_threshold: float,
                     eta: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aplica Top-P (y la expansión por margen η) a un lote de matrices.

    Devuelve (mask, k_eff, margen):
      mask   (..., N, N) bool  — aristas seleccionadas, diagonal siempre False
      k_eff  (..., N)    int   — tamaño del top de cada fila
      margen (..., N)    float — Δ = p_(k) - p_(k+1) en el punto de corte

    CORRECCIÓN vs. el código original: el criterio de parada es
    K_i = mín{k : Σ_{j≤k} p_(j) ≥ P}  (definición del informe, §6.1).
    El código previo usaba `cum > P`, que en el caso de igualdad exacta añade
    un destino de más.

    Con η > 0 se implementa el operador adaptativo Top-P_η (§7.1): si el margen
    en el corte es menor que η, se incorpora el elemento siguiente y se repite.
    Así el margen efectivo cumple Δ̃ ≥ η por construcción y desaparece la
    inestabilidad por rank-flipping.
    """
    P = probs_candidatas_batch(A_batch)
    n = P.shape[-1]
    idx = np.arange(n)

    P_sort = P.copy()
    P_sort[..., idx, idx] = -1.0                     # la diagonal nunca compite
    order = np.argsort(-P_sort, axis=-1, kind="stable")
    Ps = np.take_along_axis(P_sort, order, axis=-1)

    cum = np.cumsum(np.clip(Ps, 0.0, None), axis=-1)
    reach = cum >= (p_threshold - 1e-12)
    k = np.where(reach.any(axis=-1), np.argmax(reach, axis=-1) + 1, n - 1)
    k = np.clip(k, 1, n - 1)

    gaps = Ps[..., :-1] - Ps[..., 1:]                # gaps[t] = p_(t+1) - p_(t+2)
    if eta and eta > 0.0:
        for _ in range(n):
            g = np.take_along_axis(gaps, (k - 1)[..., None], axis=-1)[..., 0]
            crece = (g < eta) & (k < n - 1)
            if not np.any(crece):
                break
            k = np.where(crece, k + 1, k)

    margen = np.take_along_axis(gaps, np.clip(k - 1, 0, n - 2)[..., None], axis=-1)[..., 0]
    margen = np.where(k >= n - 1, 0.0, margen)

    ranks = np.argsort(order, axis=-1)
    mask = ranks < k[..., None]
    mask[..., idx, idx] = False
    return mask, k.astype(int), margen


# ----------------------------------------------------------------------------
# (c) Modelo nulo analítico y test exacto
# ----------------------------------------------------------------------------

def poisson_binomial_pmf(p: np.ndarray) -> np.ndarray:
    """PMF exacta de X = Σ_m Bernoulli(p_m) por convolución (DP en O(M²))."""
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    pmf = np.zeros(p.size + 1, dtype=np.float64)
    pmf[0] = 1.0
    for pi in p:
        pmf[1:] = pmf[1:] * (1.0 - pi) + pmf[:-1] * pi
        pmf[0] *= (1.0 - pi)
    return pmf


def poisson_binomial_sf(pmf: np.ndarray, x_obs: int) -> float:
    """P(X ≥ x_obs) a partir de la PMF (cola superior, test de una cola)."""
    if x_obs <= 0:
        return 1.0
    if x_obs >= pmf.size:
        return 0.0
    return float(np.clip(pmf[x_obs:].sum(), 0.0, 1.0))


def p_azar_por_fila(k_eff: np.ndarray, n_nodes: int) -> np.ndarray:
    """P_azar de cada (fila, semilla): probabilidad de selección bajo H0.

    CORRECCIÓN IMPORTANTE respecto al informe (§7.2). El informe fija
    P_azar = K/(N-1), válido solo para Top-K rígido, donde toda fila selecciona
    exactamente K destinos. Con Top-P el tamaño del top es ADAPTATIVO: k varía
    por fila y por semilla, así que la probabilidad nula es

        p_i^(m) = k_i^(m) / (N - 1)

    y el número de apariciones X_ij = Σ_m Bernoulli(p_i^(m)) es Poisson-binomial,
    NO binomial. Usar Binomial(M, K/(N-1)) con un K promedio subestima o
    sobreestima la cola según la dispersión de los k_i^(m), e infla los falsos
    positivos justo en las filas de atención plana (que son las que más se
    expanden). Cuando todos los k son iguales, la Poisson-binomial colapsa
    exactamente en la binomial del informe.

    k_eff: (M, N) → devuelve (M, N)
    """
    return np.clip(np.asarray(k_eff, dtype=np.float64) / float(n_nodes - 1), 0.0, 1.0)


# ----------------------------------------------------------------------------
# (d) Correcciones por comparaciones múltiples
# ----------------------------------------------------------------------------

def bonferroni(pvals: np.ndarray, alpha: float) -> Tuple[np.ndarray, float]:
    """FWER ≤ alpha. Devuelve (rechazadas, alpha_corregido)."""
    p = np.asarray(pvals, dtype=np.float64)
    alpha_corr = alpha / max(p.size, 1)
    return p <= alpha_corr, float(alpha_corr)


def benjamini_hochberg_bh(pvals: np.ndarray, alpha: float
                          ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Procedimiento step-up de BH sobre un array de p-valores.

    Se llama `_bh` (y no `benjamini_hochberg`) para no chocar con la función
    homónima de la sección "Paradigma 1" (más abajo, opera sobre un dict de
    EdgeStat con Top-K rígido; queda deprecada por esta PARTE 3).

    CORRECCIÓN respecto al informe (§7.2): el informe dice "retener la arista k
    si p_(k) ≤ (k/N_tests)·α", aplicado arista por arista. El procedimiento BH
    correcto es step-up: se busca el MAYOR k que cumple la desigualdad y se
    rechazan TODAS las hipótesis de rango ≤ k, incluidas las que individualmente
    no la cumplen. Aplicar la regla arista por arista pierde potencia y ya no
    controla la FDR al nivel nominal.
    """
    p = np.asarray(pvals, dtype=np.float64)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool), np.zeros(0), 0.0
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    umbrales = (np.arange(1, m + 1) / m) * alpha
    cumple = ranked <= umbrales
    rechazadas = np.zeros(m, dtype=bool)
    umbral_efectivo = 0.0
    if cumple.any():
        kmax = int(np.max(np.where(cumple)[0]))
        rechazadas[order[: kmax + 1]] = True
        umbral_efectivo = float(umbrales[kmax])
    q_ord = np.minimum.accumulate(
        (ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    q = np.empty(m, dtype=np.float64)
    q[order] = np.clip(q_ord, 0.0, 1.0)
    return rechazadas, q, umbral_efectivo


# ----------------------------------------------------------------------------
# (d-bis) e-valores secuenciales + e-BH — LINEAMIENTO_NUEVOS_PARADIGMAS.md §4
#
# Reemplazo/complemento de Bonferroni+BH: el p-valor binomial de arriba fija
# M (n° de semillas) de antemano y deja de ser válido si se agrega más
# muestra después de mirar el resultado (justo lo que pasó al pasar de N=20
# a N=50 en este proyecto: los p-valores de la corrida chica no eran válidos
# para decidir si valía la pena agrandar la muestra). Un e-valor es un
# estadístico de test que SIGUE siendo válido en cualquier punto de parada
# (Ville's inequality / test-by-betting): en vez de fijar M, se acumula un
# producto de factores de apuesta, uno por semilla nueva, y se puede mirar
# el resultado y decidir seguir sin invalidar el control de error.
#
# Construcción: plug-in secuencial de razón de verosimilitud (variante
# "prequential"/Kelly de test-by-betting; ver Ramdas et al. sobre e-valores y
# Wang & Ramdas 2022 para el procedimiento e-BH que agrega esto abajo). En
# cada semilla nueva, se apuesta con la frecuencia observada hasta ahora
# (acotada para nunca apostar por debajo del nulo, evita apuestas
# degeneradas con historia corta), y se multiplica el e-valor acumulado por
# el cociente de verosimilitud Bernoulli(apuesta)/Bernoulli(nulo) de la
# observación nueva. Es UNA construcción válida de e-valor, no
# necesariamente la misma función de apuesta óptima del paper de Wang,
# Dandapanthula y Ramdas (2025) — esa requiere su condición causal específica
# que no se verificó acá. La agregación e-BH de abajo sí es la exacta de
# Wang & Ramdas (2022): ordenar descendente, rechazar el mayor prefijo con
# e_(k) >= n_tests/(alpha*k).
# ----------------------------------------------------------------------------

def secuencia_evalor_bernoulli(x_seq: np.ndarray, p_null_seq: np.ndarray
                               ) -> np.ndarray:
    """E-valor acumulado tras cada observación de una secuencia Bernoulli.

    x_seq: (M,) 0/1, si la arista estuvo presente en el modelo m.
    p_null_seq: (M,) probabilidad del nulo para ese modelo/fila (p_null[:,i]
    de run_paradigma1 — varía por modelo porque k_eff varía por modelo).
    Devuelve (M,) con el e-valor acumulado hasta la observación t — válido
    parar en cualquier t, el último elemento es el e-valor final.
    """
    m = len(x_seq)
    e = np.empty(m, dtype=np.float64)
    log_e = 0.0
    exitos = 0.0
    for t in range(m):
        p0 = float(p_null_seq[t])
        if t == 0:
            p_bet = 0.5 * (p0 + 1.0)          # sin historia: apuesta conservadora
        else:
            p_bet = max(exitos / t, p0 + 1e-9)
            p_bet = min(p_bet, 1.0 - 1e-9)
        x = float(x_seq[t])
        num = p_bet if x else (1.0 - p_bet)
        den = p0 if x else (1.0 - p0)
        log_e += np.log(max(num, 1e-300)) - np.log(max(den, 1e-300))
        e[t] = np.exp(min(log_e, 700.0))       # evita overflow de exp
        exitos += x
    return e


def procedimiento_e_bh(e_valores: np.ndarray, alpha: float) -> np.ndarray:
    """e-BH (Wang & Ramdas 2022): rechaza el mayor prefijo, en orden
    descendente de e-valor, que cumple e_(k) >= n_tests/(alpha*k)."""
    e_valores = np.asarray(e_valores, dtype=np.float64)
    n = e_valores.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    orden = np.argsort(-e_valores, kind="stable")
    e_ord = e_valores[orden]
    umbral = n / (alpha * (np.arange(n) + 1))
    cumple = e_ord >= umbral
    rechazadas = np.zeros(n, dtype=bool)
    if cumple.any():
        kmax = int(np.max(np.where(cumple)[0]))
        rechazadas[orden[: kmax + 1]] = True
    return rechazadas


# ----------------------------------------------------------------------------
# (e) Intervalo de confianza exacto para Φ(e)
# ----------------------------------------------------------------------------

def clopper_pearson(x: int, m: int, level: float = CI_LEVEL) -> Tuple[float, float]:
    """IC exacto (Clopper-Pearson) para una proporción x/m."""
    a = 1.0 - level
    lo = float(stats.beta.ppf(a / 2.0, x, m - x + 1)) if x > 0 else 0.0
    hi = float(stats.beta.ppf(1.0 - a / 2.0, x + 1, m - x)) if x < m else 1.0
    return lo, hi


# ----------------------------------------------------------------------------
# (f) Test de permutaciones
# ----------------------------------------------------------------------------

def test_permutaciones(phi_obs: np.ndarray, n_nodes: int, n_seeds: int,
                       p_threshold: float, eta: float = 0.0,
                       B: int = N_PERMUTACIONES,
                       rng: Optional[np.random.Generator] = None
                       ) -> Dict[str, np.ndarray]:
    """Nulo empírico: matrices i.i.d. uniformes normalizadas por fila (§8.3).

    Además del p-valor marginal por arista se calcula el p-valor tipo max-T
    (Westfall-Young): la distribución nula del MÁXIMO de Φ sobre todas las
    aristas. Ese sí controla la FWER sin asumir independencia entre aristas,
    que es justo el supuesto que el informe reconoce como violado (seleccionar
    una arista reduce la probabilidad de las demás dentro de la misma fila).
    """
    rng = rng or np.random.default_rng(SEMILLA_NULO)
    ge = np.zeros((n_nodes, n_nodes), dtype=np.int64)
    max_nulo = np.empty(B, dtype=np.float64)
    k_nulo = np.empty(B, dtype=np.float64)

    for b in range(B):
        A_rand = rng.random((n_seeds, n_nodes, n_nodes))
        A_rand /= A_rand.sum(axis=2, keepdims=True)          # fila-estocástica
        mask, k_eff, _ = top_p_mask_batch(A_rand, p_threshold, eta)
        phi_b = mask.mean(axis=0)
        ge += (phi_b >= phi_obs)
        max_nulo[b] = phi_b.max()
        k_nulo[b] = k_eff.mean()

    # estimador con +1 (Davison & Hinkley): nunca devuelve p = 0
    p_marginal = (ge + 1.0) / (B + 1.0)
    p_maxT = np.array([(np.sum(max_nulo >= v) + 1.0) / (B + 1.0) for v in phi_obs.ravel()]
                      ).reshape(phi_obs.shape)
    return {"p_perm": p_marginal, "p_perm_maxT": p_maxT,
            "max_nulo": max_nulo, "k_medio_nulo": float(k_nulo.mean())}


# ----------------------------------------------------------------------------
# (g) Nulo de grafo con out-grado restringido + Z-scores
# ----------------------------------------------------------------------------

def nulo_grafo_restringido(k_por_fila: np.ndarray, n_nodes: int,
                           R: int = N_NULOS_GRAFO,
                           rng: Optional[np.random.Generator] = None) -> Dict[str, np.ndarray]:
    """Grafos aleatorios que respetan el out-grado impuesto por el operador.

    CORRECCIÓN respecto al informe (§21.3): el baseline propuesto es Erdős-Rényi
    G(N, p) con p = K/(N-1), donde cada arista existe de forma independiente.
    Pero el Top-K/Top-P impone que CADA FILA tenga exactamente k_i aristas
    salientes, así que el ER simple genera grafos con una distribución de
    out-grado que el pipeline nunca podría producir, y los Z-scores de
    modularidad o reciprocidad salen inflados por esa diferencia estructural y
    no por la señal. Aquí se sortean k_i destinos por fila sin reemplazo: mismo
    número esperado de aristas que ER, pero out-grado correcto.
    """
    rng = rng or np.random.default_rng(SEMILLA_NULO + 1)
    k_por_fila = np.asarray(k_por_fila, dtype=int)
    mods, recips, dens = [], [], []
    for _ in range(R):
        G = nx.DiGraph(); G.add_nodes_from(range(n_nodes))
        for i in range(n_nodes):
            cand = [j for j in range(n_nodes) if j != i]
            ki = int(min(max(k_por_fila[i], 0), len(cand)))
            for j in rng.choice(cand, size=ki, replace=False):
                G.add_edge(i, int(j))
        dens.append(G.number_of_edges() / max(n_nodes * (n_nodes - 1), 1))
        try:
            recips.append(float(nx.reciprocity(G)))
        except Exception:
            recips.append(0.0)
        try:
            U = G.to_undirected()
            com = list(nx.community.louvain_communities(U, seed=42))
            mods.append(float(nx.community.modularity(U, com)))
        except Exception:
            mods.append(0.0)
    return {"modularity": np.array(mods), "reciprocity": np.array(recips),
            "density": np.array(dens)}


def z_score(valor: float, nulo: np.ndarray) -> float:
    sd = float(np.std(nulo))
    return float((valor - float(np.mean(nulo))) / sd) if sd > 1e-12 else 0.0


# ----------------------------------------------------------------------------
# Motor del paradigma 1
# ----------------------------------------------------------------------------

def run_paradigma1(matrices: List[np.ndarray], labels: List[str], seeds: List[int],
                   top_p: float, eta: Optional[float] = None,
                   alpha: float = ALPHA, alpha_fdr: float = FDR_TARGET,
                   n_perm: int = N_PERMUTACIONES,
                   eps_perturbacion: Optional[float] = None,
                   eps_perturbacion_max: Optional[float] = None,
                   verbose: bool = True) -> Dict:
    """Ejecuta el paradigma probabilístico completo para un umbral P dado."""
    # eta=None resuelve al valor EFECTIVO. No se pone ETA_MARGEN como default
    # del argumento porque los defaults se evalúan al definir la función y no
    # verían el recálculo que hace calcular_eta().
    eta = ETA_MARGEN if eta is None else float(eta)
    A = np.stack([np.asarray(m, dtype=np.float64) for m in matrices])   # (M,N,N)
    M, n, _ = A.shape

    mask, k_eff, margen = top_p_mask_batch(A, top_p, eta)               # (M,N,N),(M,N)
    phi = mask.mean(axis=0)                                             # persistencia Φ(e)
    x_obs = mask.sum(axis=0).astype(int)

    # --- modelo nulo por fila ---------------------------------------------
    p_null = p_azar_por_fila(k_eff, n)                                  # (M,N)
    p_null_medio = p_null.mean(axis=0)                                  # (N,)

    pvals = np.ones((n, n), dtype=np.float64)
    for i in range(n):
        pmf = poisson_binomial_pmf(p_null[:, i])                        # una vez por fila
        for j in range(n):
            if i == j:
                continue
            pvals[i, j] = poisson_binomial_sf(pmf, int(x_obs[i, j]))
    np.fill_diagonal(pvals, np.nan)

    off = ~np.eye(n, dtype=bool)
    p_flat = pvals[off]
    n_tests = int(p_flat.size)                                          # N(N-1) = 132

    rech_bonf, alpha_bonf = bonferroni(p_flat, alpha)
    rech_bh, q_flat, umbral_bh = benjamini_hochberg_bh(p_flat, alpha_fdr)

    sig_bonf = np.zeros((n, n), dtype=bool); sig_bonf[off] = rech_bonf
    sig_bh = np.zeros((n, n), dtype=bool);   sig_bh[off] = rech_bh
    qvals = np.full((n, n), np.nan);         qvals[off] = q_flat

    # --- e-valores secuenciales + e-BH (complemento de BH, ver §4 del
    # lineamiento): válidos parar en cualquier M, a diferencia del p-valor
    # binomial de arriba que asume M fijo de antemano. -----------------------
    e_final = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            e_seq = secuencia_evalor_bernoulli(mask[:, i, j].astype(np.float64),
                                               p_null[:, i])
            e_final[i, j] = e_seq[-1]
    e_flat = e_final[off]
    rech_ebh = procedimiento_e_bh(e_flat, alpha_fdr)
    sig_ebh = np.zeros((n, n), dtype=bool); sig_ebh[off] = rech_ebh

    # --- IC exacto para Φ(e) ----------------------------------------------
    ci_lo = np.zeros((n, n)); ci_hi = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                ci_lo[i, j], ci_hi[i, j] = clopper_pearson(int(x_obs[i, j]), M)

    # --- test de permutaciones --------------------------------------------
    perm = test_permutaciones(phi, n, M, top_p, eta, B=n_perm)

    # --- márgenes ------------------------------------------------------------
    margen_medio_fila = margen.mean(axis=0)
    margen_min_fila = margen.min(axis=0)
    margen_cv_fila = margen.std(axis=0) / np.maximum(margen.mean(axis=0), 1e-12)
    # Lema H3 (§7.1): el Top-K se preserva si Δ > 2ε, con ε la máxima variación
    # POR ENTRADA de la fila renormalizada (no 2·L_inf, ver nota en la FASE 1).
    cota_h3 = (2.0 * eps_perturbacion) if eps_perturbacion is not None else None
    cota_h3_peor = (2.0 * eps_perturbacion_max) if eps_perturbacion_max is not None else None
    filas_h3_ok = (int(np.sum(margen_medio_fila > cota_h3))
                   if cota_h3 is not None else None)
    filas_h3_ok_peor = (int(np.sum(margen_min_fila > cota_h3_peor))
                        if cota_h3_peor is not None else None)

    # --- grafo de consenso -------------------------------------------------
    core_freq = {(i, j): float(phi[i, j]) for i in range(n) for j in range(n)
                 if i != j and phi[i, j] >= UMBRAL_DE_ESTABILIDAD}
    core_stat = {(i, j): float(phi[i, j]) for i in range(n) for j in range(n)
                 if i != j and sig_bh[i, j]}
    consenso = {e: core_stat[e] for e in core_stat if e in core_freq}     # ambos criterios

    G_cons = nx.DiGraph(); G_cons.add_nodes_from(range(n))
    for (i, j), w in core_stat.items():
        G_cons.add_edge(i, j, weight=w)
    n_edges_cons = G_cons.number_of_edges()
    sparsity = 1.0 - n_edges_cons / max(n * (n - 1), 1)
    activos = {v for e in G_cons.edges() for v in e}
    cobertura = len(activos) / n
    w_sum = sum(d["weight"] for _, _, d in G_cons.edges(data=True))
    if w_sum > 0:
        pw = np.array([d["weight"] / w_sum for _, _, d in G_cons.edges(data=True)])
        H_struct = float(-special.xlogy(pw, pw).sum())
    else:
        H_struct = 0.0

    # --- Z-scores frente al nulo con out-grado restringido -----------------
    k_medio_fila = np.rint(k_eff.mean(axis=0)).astype(int)
    nulo = nulo_grafo_restringido(k_medio_fila, n)
    mods_obs, recips_obs, dens_obs = [], [], []
    for m_idx in range(M):
        G = nx.DiGraph(); G.add_nodes_from(range(n))
        for i, j in zip(*np.where(mask[m_idx])):
            G.add_edge(int(i), int(j))
        dens_obs.append(G.number_of_edges() / max(n * (n - 1), 1))
        try:
            recips_obs.append(float(nx.reciprocity(G)))
        except Exception:
            recips_obs.append(0.0)
        try:
            U = G.to_undirected()
            com = list(nx.community.louvain_communities(U, seed=42))
            mods_obs.append(float(nx.community.modularity(U, com)))
        except Exception:
            mods_obs.append(0.0)
    z_mod = z_score(float(np.mean(mods_obs)), nulo["modularity"])
    z_rec = z_score(float(np.mean(recips_obs)), nulo["reciprocity"])

    res = {
        "top_p": top_p, "eta": eta, "n_seeds": M, "n_nodes": n,
        "labels": list(labels), "seeds": list(seeds),
        "phi": phi, "x_obs": x_obs, "mask": mask,
        "k_eff": k_eff, "k_medio": float(k_eff.mean()),
        "k_min": int(k_eff.min()), "k_max": int(k_eff.max()),
        "p_null": p_null, "p_null_medio": p_null_medio,
        "p_azar_global": float(p_null.mean()),
        "pvals": pvals, "qvals": qvals,
        "sig_bonferroni": sig_bonf, "sig_bh": sig_bh,
        "e_values": e_final, "sig_e_bh": sig_ebh,
        "n_sig_e_bh": int(sig_ebh.sum()),
        "alpha": alpha, "alpha_bonferroni": alpha_bonf,
        "alpha_fdr": alpha_fdr, "umbral_bh": umbral_bh, "n_tests": n_tests,
        "ci_lo": ci_lo, "ci_hi": ci_hi,
        "p_perm": perm["p_perm"], "p_perm_maxT": perm["p_perm_maxT"],
        "k_medio_nulo_perm": perm["k_medio_nulo"],
        "margen_medio_fila": margen_medio_fila, "margen_min_fila": margen_min_fila,
        "margen_cv_fila": margen_cv_fila, "margen_medio": float(margen.mean()),
        "margen_min": float(margen.min()),
        "cota_h3": cota_h3, "filas_h3_ok": filas_h3_ok,
        "cota_h3_peor": cota_h3_peor, "filas_h3_ok_peor": filas_h3_ok_peor,
        "core_freq": core_freq, "core_stat": core_stat, "consenso": consenso,
        "n_sig_bonferroni": int(sig_bonf.sum()), "n_sig_bh": int(sig_bh.sum()),
        "n_core_freq": len(core_freq), "n_consenso": len(consenso),
        "sparsity": sparsity, "cobertura": cobertura, "H_estructural": H_struct,
        "n_edges_consenso": n_edges_cons,
        "modularity_obs": float(np.mean(mods_obs)),
        "reciprocity_obs": float(np.mean(recips_obs)),
        "density_obs": float(np.mean(dens_obs)),
        "z_modularity": z_mod, "z_reciprocity": z_rec,
        "nulo_modularity_mean": float(np.mean(nulo["modularity"])),
        "nulo_reciprocity_mean": float(np.mean(nulo["reciprocity"])),
    }
    res["veredicto"] = _veredicto_paradigma1(res)
    if verbose:
        _reporte_paradigma1(res, labels)
    return res


def _veredicto_paradigma1(r: Dict) -> Dict[str, str]:
    """Umbrales de la Tabla 7 (etapas 3 y 4) aplicados al resultado."""
    v = {}
    v["Margen"] = ("APRUEBA" if r["margen_min"] > 0.01
                   else "FALLA" if r["margen_min"] < 1e-6 else "DUDOSO")
    v["Significancia"] = ("APRUEBA" if r["n_sig_bh"] > 0
                          else "FALLA")
    # El abs() trataba un Z NEGATIVO como evidencia a favor. Un Z de -1.8 (medido
    # el 31/08 en Pencahue) dice que el grafo es MENOS modular que un aleatorio
    # del mismo out-grado: es evidencia CONTRA la estructura de comunidades, y
    # con el abs() habría pasado a APRUEBA al llegar a -2.1. La dirección importa.
    zm = r["z_modularity"]
    v["Z_modularidad"] = ("APRUEBA" if zm > 2
                          else "FALLA" if zm < -2
                          else "FALLA" if abs(zm) < 1 else "DUDOSO")
    v["Consenso"] = ("APRUEBA" if r["n_consenso"] > 0 else "FALLA")
    v["Sparsity"] = ("APRUEBA" if r["sparsity"] > 0.7
                     else "FALLA" if r["sparsity"] < 0.4 else "DUDOSO")

    # GLOBAL: antes bastaba con que nada dijera FALLA, así que tres DUDOSO y dos
    # APRUEBA daban APRUEBA (exactamente lo que pasó en Pencahue). Aprobar una
    # etapa exige mayoría de dimensiones aprobadas, no ausencia de suspensos.
    dims = [x for k, x in v.items() if k != "GLOBAL" and not k.startswith("_")]
    n_falla = sum(1 for x in dims if x == "FALLA")
    n_aprueba = sum(1 for x in dims if x == "APRUEBA")
    if not EMITIR_VEREDICTO_GLOBAL:
        # Se reporta la frontera y no un aprobado: Margen, Sparsity y
        # Significancia no pueden cumplirse a la vez bajo Top-P (S ~ 1-P contra
        # J creciente en P), asi que un GLOBAL agregado premiaria a la P que
        # mejor equilibra criterios incompatibles y lo presentaria como
        # validacion.
        v["GLOBAL"] = (str(n_aprueba) + " aprueban / " + str(n_falla)
                       + " fallan de " + str(len(dims)))
    elif n_falla >= 2:
        v["GLOBAL"] = "FALLA"
    elif n_falla == 0 and n_aprueba > len(dims) / 2:
        v["GLOBAL"] = "APRUEBA"
    else:
        v["GLOBAL"] = "DUDOSO"
    v["_conteo"] = f"{n_aprueba} aprueba / {n_falla} falla de {len(dims)}"
    return v


def _reporte_paradigma1(r: Dict, labels: List[str]) -> None:
    n = r["n_nodes"]; M = r["n_seeds"]
    print(f"\n{'═'*90}")
    print(f" PARADIGMA 1 · PROBABILÍSTICO — P = {r['top_p']}  ·  η = {r['eta']}")
    print(f" M = {M} semillas · N = {n} nodos · {r['n_tests']} aristas candidatas")
    print(f"{'═'*90}")

    print(f"\n[1] OPERADOR DE CORTE Y MARGEN (etapa 3, §7.1)")
    print(f"  Tamaño del top por fila: μ={r['k_medio']:.2f}  min={r['k_min']}  max={r['k_max']}")
    print(f"  Margen Δ en el corte: μ={r['margen_medio']:.5f}  mín={r['margen_min']:.5f}")
    if r["eta"] > 0:
        print(f"  Operador adaptativo Top-P_η activo: Δ̃ ≥ η = {r['eta']} por construcción")
    peor = int(np.argmin(r["margen_medio_fila"]))
    print(f"  Fila con el corte más ambiguo: {labels[peor]} "
          f"(Δ = {r['margen_medio_fila'][peor]:.5f}, CV = {r['margen_cv_fila'][peor]:.2f})")

    print(f"\n[2] MODELO NULO (etapa 3, §7.2)")
    print(f"  P_azar por fila = k_i/(N-1), adaptativo → media global = {r['p_azar_global']:.4f}")
    print(f"  Distribución nula: Poisson-binomial exacta sobre las {M} semillas")
    print(f"  (colapsa en Binomial(M, K/(N-1)) si todas las filas usan el mismo k)")

    print(f"\n[3] SIGNIFICANCIA CON CORRECCIÓN POR COMPARACIONES MÚLTIPLES")
    print(f"  Bonferroni: α_corr = {r['alpha']}/{r['n_tests']} = {r['alpha_bonferroni']:.2e} "
          f"→ {r['n_sig_bonferroni']} aristas")
    print(f"  Benjamini-Hochberg (FDR={r['alpha_fdr']}): umbral efectivo p ≤ {r['umbral_bh']:.2e} "
          f"→ {r['n_sig_bh']} aristas")
    print(f"  Frecuencia bruta Φ ≥ {UMBRAL_DE_ESTABILIDAD}: {r['n_core_freq']} aristas")
    print(f"  Consenso (significativa BH ∧ Φ ≥ {UMBRAL_DE_ESTABILIDAD}): {r['n_consenso']} aristas")

    print(f"\n[4] TEST DE PERMUTACIONES (no paramétrico, §8.3)")
    off = ~np.eye(n, dtype=bool)
    print(f"  Tamaño medio del top bajo el nulo uniforme: {r['k_medio_nulo_perm']:.2f} "
          f"vs {r['k_medio']:.2f} observado")
    n_perm_sig = int(np.sum((r["p_perm_maxT"] < 0.05) & off))
    print(f"  Aristas con control FWER tipo max-T (p<0.05): {n_perm_sig}")

    print(f"\n[5] CONTRASTE CONTRA GRAFO ALEATORIO (out-grado restringido, §21.3)")
    print(f"  Modularidad: obs={r['modularity_obs']:.3f} vs nulo={r['nulo_modularity_mean']:.3f} "
          f"→ Z = {r['z_modularity']:+.2f}")
    print(f"  Reciprocidad: obs={r['reciprocity_obs']:.3f} vs nulo={r['nulo_reciprocity_mean']:.3f} "
          f"→ Z = {r['z_reciprocity']:+.2f}")
    print(f"  (|Z|>2 aprueba, |Z|<1 indistinguible del azar)")

    print(f"\n[6] COMPACIDAD DEL GRAFO DE CONSENSO")
    print(f"  Aristas = {r['n_edges_consenso']}  ·  Sparsity S = {r['sparsity']:.3f}  "
          f"·  Cobertura C = {r['cobertura']:.3f}  ·  H(G) = {r['H_estructural']:.3f}")

    print(f"\n[7] VEREDICTO DEL PARADIGMA 1")
    for k, v in r["veredicto"].items():
        if k != "GLOBAL":
            print(f"  {k:<16s} {v}")
    print(f"  {'─'*40}")
    print(f"  {'GLOBAL':<16s} {r['veredicto']['GLOBAL']}")


def plot_paradigma1(r: Dict, labels: List[str], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    n = r["n_nodes"]; M = r["n_seeds"]; P = r["top_p"]
    off = ~np.eye(n, dtype=bool)

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))

    ax = axes[0, 0]
    im = ax.imshow(r["phi"] * 100, cmap="YlOrRd", vmin=0, vmax=100, aspect="equal")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            txt = f"{r['phi'][i,j]*100:.0f}"
            if r["sig_bh"][i, j]:
                txt += "*"
            if r["phi"][i, j] > 0:
                ax.text(j, i, txt, ha="center", va="center", fontsize=6,
                        color="white" if r["phi"][i, j] > 0.5 else "black",
                        fontweight="bold")
    ax.set_title(f"Persistencia Φ(e) en % — * = significativa (BH)\nP={P}", fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, label="Φ (%)")

    ax = axes[0, 1]
    phi_f = r["phi"][off]; p_f = np.clip(r["pvals"][off], 1e-300, 1.0)
    sig = r["sig_bh"][off]
    ax.scatter(phi_f[~sig], -np.log10(p_f[~sig]), s=28, c="#95a5a6",
               alpha=0.7, label="no significativa")
    ax.scatter(phi_f[sig], -np.log10(p_f[sig]), s=48, c="#c0392b",
               alpha=0.9, label="significativa (BH)")
    if r["umbral_bh"] > 0:
        ax.axhline(-np.log10(r["umbral_bh"]), color="orange", ls="--",
                   label=f"umbral BH ({r['umbral_bh']:.1e})")
    ax.axhline(-np.log10(r["alpha_bonferroni"]), color="darkred", ls=":",
               label=f"Bonferroni ({r['alpha_bonferroni']:.1e})")
    ax.axvline(r["p_azar_global"], color="blue", ls="--",
               label=f"P_azar = {r['p_azar_global']:.2f}")
    ax.set_xlabel("Φ(e) — persistencia observada")
    ax.set_ylabel("-log10(p)")
    ax.set_title("Volcán: persistencia vs significancia", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    idxs = sorted([(i, j) for i in range(n) for j in range(n) if i != j],
                  key=lambda e: -r["phi"][e])[:15]
    y = np.arange(len(idxs))
    vals = np.array([r["phi"][e] for e in idxs])
    lo = vals - np.array([r["ci_lo"][e] for e in idxs])
    hi = np.array([r["ci_hi"][e] for e in idxs]) - vals
    cols = ["#c0392b" if r["sig_bh"][e] else "#7f8c8d" for e in idxs]
    ax.barh(y, vals, xerr=[lo, hi], color=cols, alpha=0.85, capsize=3)
    ax.axvline(r["p_azar_global"], color="blue", ls="--",
               label=f"E[Φ] bajo H0 = {r['p_azar_global']:.2f}")
    ax.axvline(UMBRAL_DE_ESTABILIDAD, color="green", ls="--",
               label=f"umbral de consenso ({UMBRAL_DE_ESTABILIDAD})")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{labels[i]}→{labels[j]}" for i, j in idxs], fontsize=8)
    ax.invert_yaxis(); ax.set_xlim(0, 1.05)
    ax.set_xlabel("Φ(e) con IC95% de Clopper-Pearson")
    ax.set_title("Top-15 aristas por persistencia", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="x")

    ax = axes[1, 1]
    ax.hist(r["pvals"][off], bins=25, range=(0, 1), color="#2980b9",
            alpha=0.8, edgecolor="white")
    ax.axhline(r["n_tests"] / 25, color="red", ls="--",
               label="uniforme esperada bajo H0")
    ax.set_xlabel("p-valor (Poisson-binomial exacto)")
    ax.set_ylabel("nº de aristas")
    ax.set_title("Distribución de p-valores\n(pico en 0 = señal; plana = todo azar)",
                 fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    plt.suptitle(f"PARADIGMA 1 · Probabilístico — P={P} — {M} semillas — "
                 f"{r['n_sig_bh']} aristas significativas (BH)",
                 fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"paradigma1_P{P}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


def exportar_paradigm1_full_json(res_por_p: Dict, p_optimo: float, path: str) -> None:
    """Escribe paradigm1_results.json con TODAS las N(N-1) aristas candidatas
    (no solo las filtradas por significancia), en el esquema que espera la
    triangulación (build_diagnosis_matrix_5d / classify_edge_5d):
    i, j, phi (int = conteo x_obs), p_obs (float = Φ), p_value, bh_sig.

    Antes de esto, la triangulación nunca encontraba resultados del
    Paradigma 1 porque este archivo solo lo generaba el código obsoleto de
    PARTE 4 (Top-K rígido, EJECUTAR_PARTE4=False, nunca se ejecuta); el
    Paradigma 1 activo (PARTE 3, Top-P probabilístico) solo exportaba
    significant_edges_paradigma1.json, que además viene pre-filtrado a las
    aristas ya significativas — usarlo tal cual sesgaría la triangulación
    ocultando las aristas de RUIDO ESTOCÁSTICO antes de clasificarlas.
    """
    r = res_por_p[p_optimo]["paradigma1"]
    n = r["n_nodes"]
    edges = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            edges.append({
                "i": int(i), "j": int(j),
                "phi": int(r["x_obs"][i, j]), "p_obs": float(r["phi"][i, j]),
                "p_value": float(r["pvals"][i, j]),
                "bh_sig": bool(r["sig_bh"][i, j]),
                # e-BH secuencial (LINEAMIENTO_NUEVOS_PARADIGMAS.md §4): NO
                # reemplaza bh_sig como criterio de re_pass en
                # build_diagnosis_matrix_5d todavía — se deja al lado para
                # auditar cuánto difiere antes de conmutar el criterio activo.
                "e_value": (float(r["e_values"][i, j])
                           if "e_values" in r else None),
                "e_bh_sig": (bool(r["sig_e_bh"][i, j])
                            if "sig_e_bh" in r else None),
            })
    # "p_optimo" es la clave que lee p_efectivo() para que los paradigmas 2 a 5
    # discreticen con el MISMO P que eligió el Paradigma 1. Se mantiene "k" como
    # alias por compatibilidad con lectores antiguos del JSON.
    payload = {"p_optimo": float(p_optimo), "k": p_optimo,
               "operador": "Top-P_eta", "eta": float(ETA_MARGEN),
               "M": r["n_seeds"], "N": n, "edges": edges}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  ✅ Resultados numéricos (todas las aristas): {path}")
    if "sig_e_bh" in r:
        bh = r["sig_bh"]; ebh = r["sig_e_bh"]
        coinciden = int((bh & ebh).sum())
        solo_bh = int((bh & ~ebh).sum())
        solo_ebh = int((~bh & ebh).sum())
        print(f"     BH: {int(bh.sum())} sig. · e-BH: {int(ebh.sum())} sig. · "
              f"coinciden {coinciden} · solo BH {solo_bh} · solo e-BH {solo_ebh}")

def exportar_aristas_significativas(res_por_p: Dict, p_optimo: float,
                                    labels: List[str], path: str) -> None:
    """Escribe significant_edges.json con las aristas del P* elegido."""
    r = res_por_p[p_optimo]["paradigma1"]
    n = r["n_nodes"]
    edges = []
    for i in range(n):
        for j in range(n):
            if i == j or not (r["sig_bh"][i, j] or r["phi"][i, j] >= UMBRAL_DE_ESTABILIDAD):
                continue
            edges.append({
                "source": labels[i], "target": labels[j], "i": int(i), "j": int(j),
                "phi": float(r["phi"][i, j]), "x_obs": int(r["x_obs"][i, j]),
                "n_seeds": int(r["n_seeds"]),
                "p_azar": float(r["p_null_medio"][i]),
                "p_value": float(r["pvals"][i, j]),
                "q_value": float(r["qvals"][i, j]),
                "ci95": [float(r["ci_lo"][i, j]), float(r["ci_hi"][i, j])],
                "p_perm": float(r["p_perm"][i, j]),
                "p_perm_maxT": float(r["p_perm_maxT"][i, j]),
                "sig_bh": bool(r["sig_bh"][i, j]),
                "sig_bonferroni": bool(r["sig_bonferroni"][i, j]),
                "consenso": bool((i, j) in r["consenso"]),
            })
    edges.sort(key=lambda d: (d["p_value"], -d["phi"]))
    payload = {
        "paradigma": "1_probabilistico",
        "operador": "Top-P_eta", "top_p": float(p_optimo), "eta": float(r["eta"]),
        "n_seeds": int(r["n_seeds"]), "n_nodes": int(n), "labels": list(labels),
        "modelo_nulo": "Poisson-binomial con p_i^(m) = k_i^(m)/(N-1)",
        "alpha": float(r["alpha"]), "alpha_bonferroni": float(r["alpha_bonferroni"]),
        "alpha_fdr": float(r["alpha_fdr"]), "umbral_bh": float(r["umbral_bh"]),
        "n_tests": int(r["n_tests"]),
        "n_sig_bh": int(r["n_sig_bh"]), "n_sig_bonferroni": int(r["n_sig_bonferroni"]),
        "sparsity": float(r["sparsity"]), "cobertura": float(r["cobertura"]),
        "z_modularity": float(r["z_modularity"]),
        "veredicto": r["veredicto"],
        "edges": edges,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n  Aristas significativas (paradigma 1, P={p_optimo}) → {path}  ({len(edges)} aristas)")


def analyze_sensitivity_p(all_matrices: List[np.ndarray], labels: List[str],
                          seeds: List[int], p_values: List[float],
                          eps_perturbacion: Optional[float] = None,
                          eps_perturbacion_max: Optional[float] = None,
                          ejecutar_paradigma1: bool = True) -> Dict:
    """Barrido de P: discretización (etapa 2) + paradigma 1 probabilístico.

    `eps_perturbacion` viene de la FASE 1 y se usa para contrastar el margen del
    corte con la cota de Lipschitz del lema H3 (Δ_K > 2ε).
    """
    n_models = len(all_matrices)
    results_per_p = {}

    print(f"\n{'═'*90}")
    print(f" ANÁLISIS DE SENSIBILIDAD DEL UMBRAL P (Top-P) + PARADIGMA 1")
    print(f" P valores: {p_values}")
    print(f" Modelos: {n_models}")
    print(f"{'═'*90}")

    for P in p_values:
        print(f"\n{'─'*60}")
        print(f"  Ejecutando pipeline con P = {P}")
        print(f"{'─'*60}")
        analyses = []
        for idx, (matrix, seed) in enumerate(zip(all_matrices, seeds)):
            a = run_framework_pipeline(matrix, labels, seed, top_p=P)
            analyses.append(a)

        jaccard = compute_jaccard_matrix(analyses)

        edge_freq = compute_edge_frequency(analyses, labels)

        core, periphery = compute_core_periphery_decomposition(edge_freq, n_models, labels)

        # ------------------------------------------------------------------
        # PARADIGMA 1 · ¿la persistencia observada supera al azar?
        # Sustituye al criterio heurístico "frecuencia ≥ 80%" por un contraste
        # formal contra el modelo nulo, con corrección por 132 comparaciones.
        # ------------------------------------------------------------------
        p1 = run_paradigma1(all_matrices, labels, seeds, top_p=P,
                            eta=ETA_MARGEN, alpha=ALPHA, alpha_fdr=FDR_TARGET,
                            n_perm=N_PERMUTACIONES,
                            eps_perturbacion=eps_perturbacion,
                            eps_perturbacion_max=eps_perturbacion_max,
                            verbose=True) if ejecutar_paradigma1 else None

        entropies = [a.entropy for a in analyses]
        spectral_gaps = [a.spectral_gap for a in analyses]
        sigma2s = [a.svd_sigma2 for a in analyses]
        symmetry_ratios = [a.symmetry_ratio for a in analyses]

        reciprocities = [a.reciprocity for a in analyses]
        modularities = [a.modularity for a in analyses]

        role_freq = Counter()
        for a in analyses:
            for node, role in a.node_roles.items():
                role_freq[(node, role)] += 1

        # Tamaño promedio del top por fila (reemplaza al K entero de Top-K)
        avg_top_size = _avg_top_p_size(seeds, P)

        results_per_p[P] = {
            "paradigma1": p1,
            "n_sig_bh": (p1["n_sig_bh"] if p1 else 0),
            "n_sig_bonferroni": (p1["n_sig_bonferroni"] if p1 else 0),
            "n_consenso": (p1["n_consenso"] if p1 else 0),
            "sparsity": (p1["sparsity"] if p1 else np.nan),
            "z_modularity": (p1["z_modularity"] if p1 else np.nan),
            "margen_min": (p1["margen_min"] if p1 else np.nan),
            "veredicto_p1": (p1["veredicto"]["GLOBAL"] if p1 else "n/a"),
            "analyses": analyses,
            "jaccard": jaccard,
            "jaccard_mean": float(jaccard.mean()),
            "jaccard_median": float(np.median(jaccard)),
            "jaccard_std": float(jaccard.std()),
            "jaccard_gt05": float((jaccard > 0.5).mean()*100),
            "jaccard_gt03": float((jaccard > 0.3).mean()*100),
            "edge_freq": edge_freq,
            "core": core,
            "periphery": periphery,
            "n_core": len(core),
            "n_periphery": len(periphery),
            "entropy_mean": float(np.mean(entropies)),
            "entropy_std": float(np.std(entropies)),
            "spectral_gap_mean": float(np.mean(spectral_gaps)),
            "spectral_gap_std": float(np.std(spectral_gaps)),
            "sigma2_mean": float(np.mean(sigma2s)),
            "sigma2_std": float(np.std(sigma2s)),
            "symmetry_ratio_mean": float(np.mean(symmetry_ratios)),
            "reciprocity_mean": float(np.mean(reciprocities)),
            "modularity_mean": float(np.mean(modularities)),
            "modularity_std": float(np.std(modularities)),
            "role_freq": role_freq,
            "avg_top_size": avg_top_size,
            "top_p_sizes_per_seed": {s: list(GLOBAL_TOP_P_SIZES.get((s, float(P)), []))
                                     for s in seeds},
        }

        print(f"\n  P={P} RESUMEN:")
        print(f"    Jaccard: μ={jaccard.mean():.3f} ± {jaccard.std():.3f} "
              f"(mediana={np.median(jaccard):.3f})")
        print(f"    Jaccard > 0.5: {(jaccard>0.5).mean()*100:.1f}%")
        print(f"    Jaccard > 0.3: {(jaccard>0.3).mean()*100:.1f}%")
        print(f"    Core edges (≥80% freq): {len(core)}")
        print(f"    Periphery edges (<80% freq): {len(periphery)}")
        if p1:
            print(f"    Significativas BH (FDR={FDR_TARGET}): {p1['n_sig_bh']}  |  "
                  f"Bonferroni (α={ALPHA}): {p1['n_sig_bonferroni']}  |  "
                  f"consenso: {p1['n_consenso']}")
        print(f"    Tamaño promedio del top por fila: {avg_top_size:.2f}")
        print(f"    Entropía: μ={np.mean(entropies):.3f} ± {np.std(entropies):.3f}")
        print(f"    Gap espectral: μ={np.mean(spectral_gaps):.4f} ± {np.std(spectral_gaps):.4f}")
        print(f"    Reciprocidad: μ={np.mean(reciprocities):.3f}")
        print(f"    Modularidad: μ={np.mean(modularities):.3f} ± {np.std(modularities):.3f}")

        if core:
            print(f"\n    CORE EDGES (≥80% frecuencia) — con su veredicto estadístico:")
            sorted_core = sorted(core.items(), key=lambda x: -x[1])
            for (u,v), freq in sorted_core[:10]:
                pct = freq/n_models*100
                marca = ""
                if p1:
                    marca = ("  [significativa BH]" if p1["sig_bh"][u, v]
                             else f"  [NO significativa, p={p1['pvals'][u, v]:.2e}]")
                print(f"      {labels[u]:12s} → {labels[v]:12s} : "
                      f"{freq}/{n_models} ({pct:.0f}%){marca}")
        else:
            print(f"\n     NO hay aristas core (ninguna alcanza 80% frecuencia)")

        if periphery:
            sorted_periph = sorted(periphery.items(), key=lambda x: -x[1])
            print(f"\n    TOP PERIPHERY EDGES (más frecuentes pero <80%):")
            for (u,v), freq in sorted_periph[:5]:
                pct = freq/n_models*100
                print(f"      {labels[u]:12s} → {labels[v]:12s} : {freq}/{n_models} ({pct:.0f}%)")

    print(f"\n\n{'═'*90}")
    print(f" ANÁLISIS CRUZADO: CÓMO CAMBIA LA ESTABILIDAD CON P")
    print(f"{'═'*90}")

    print(f"\n[1] EVOLUCIÓN DEL JACCARD Y DENSIDAD")
    print(f"  {'P':>6s}  {'Jaccard μ':>12s}  {'Jaccard σ':>12s}  {'Mediana':>10s}  {'>0.5':>8s}  {'>0.3':>8s}  {'Core':>6s}  {'Periph':>8s}  {'AvgTop':>8s}")
    print(f"  {'─'*90}")
    for P in p_values:
        r = results_per_p[P]
        print(f"  {P:>6.2f}  {r['jaccard_mean']:>12.3f}  {r['jaccard_std']:>12.3f}  "
              f"{r['jaccard_median']:>10.3f}  {r['jaccard_gt05']:>7.1f}%  {r['jaccard_gt03']:>7.1f}%  "
              f"{r['n_core']:>6d}  {r['n_periphery']:>8d}  {r['avg_top_size']:>8.2f}")

    print(f"\n[2] CONTROL DE DENSIDAD CONTRA EL AZAR (Jaccard_real vs Jaccard_random)")
    print(f"  {'P':>6s}  {'Jaccard μ':>12s}  {'AvgTop':>8s}  {'Jaccard_random':>16s}  {'Ratio':>8s}  {'Densidad (%)':>15s}")
    print(f"  {'─'*90}")
    for P in p_values:
        r = results_per_p[P]
        k_avg = r['avg_top_size']
        jac_random = k_avg / max(2*(NUM_INDICES - 1) - k_avg, 1e-6)
        ratio = r['jaccard_mean'] / max(jac_random, 1e-6)
        density_pct = (k_avg / (NUM_INDICES - 1)) * 100
        # Se guardan para que la elección de P* use el MISMO número que se
        # imprime aquí, en vez de recalcularlo con otra fórmula.
        r['jaccard_random'] = float(jac_random)
        r['ratio_vs_azar'] = float(ratio)
        print(f"  {P:>6.2f}  {r['jaccard_mean']:>12.3f}  {k_avg:>8.2f}  {jac_random:>16.4f}  {ratio:>8.1f}x  {density_pct:>14.1f}%")

    print(f"\n[3] ARISTAS CORE QUE EMERGEN AL AUMENTAR P")
    print(f"  ¿Aparecen nuevas aristas estables (≥80% freq) que antes quedaban fuera?")
    for i in range(1, len(p_values)):
        P_prev, P_curr = p_values[i-1], p_values[i]
        core_prev = set(results_per_p[P_prev]['core'].keys())
        core_curr = set(results_per_p[P_curr]['core'].keys())
        new_core = core_curr - core_prev
        lost_core = core_prev - core_curr
        print(f"\n  P={P_prev} → P={P_curr}:")
        print(f"    Core anterior: {len(core_prev)} | Core nuevo: {len(core_curr)}")
        if new_core:
            print(f"    NUEVAS core ({len(new_core)}):")
            for (u,v) in sorted(new_core):
                freq = results_per_p[P_curr]['edge_freq'][(u,v)]
                pct = freq/n_models*100
                freq_prev = results_per_p[P_prev]['edge_freq'].get((u,v), 0)
                pct_prev = freq_prev/n_models*100
                print(f"      {labels[u]:12s} → {labels[v]:12s} : "
                      f"freq P={P_prev}={pct_prev:.0f}% → P={P_curr}={pct:.0f}%")
        else:
            print(f"      No emergen nuevas aristas core")
        if lost_core:
            print(f"    Core PERDIDAS ({len(lost_core)}):")
            for (u,v) in sorted(lost_core):
                print(f"      {labels[u]:12s} → {labels[v]:12s}")

    print(f"\n[4] MÉTRICAS ESPECTRALES vs P")
    print(f"  {'P':>6s}  {'Gap espectral':>16s}  {'Entropía':>12s}  {'σ₂':>12s}  {'Reciprocidad':>14s}  {'Modularidad':>14s}")
    print(f"  {'─'*90}")
    for P in p_values:
        r = results_per_p[P]
        print(f"  {P:>6.2f}  {r['spectral_gap_mean']:>12.4f}±{r['spectral_gap_std']:.4f}  "
              f"{r['entropy_mean']:>8.3f}±{r['entropy_std']:.3f}  "
              f"{r['sigma2_mean']:>8.4f}±{r['sigma2_std']:.4f}  "
              f"{r['reciprocity_mean']:>14.3f}  "
              f"{r['modularity_mean']:>8.3f}±{r['modularity_std']:.3f}")

    print(f"\n[5] DETECCIÓN DE PUNTO CRÍTICO")
    gaps = [results_per_p[P]['spectral_gap_mean'] for P in p_values]
    mods = [results_per_p[P]['modularity_mean'] for P in p_values]
    jacs = [results_per_p[P]['jaccard_mean'] for P in p_values]
    cores = [results_per_p[P]['n_core'] for P in p_values]

    critical_found = False
    for i in range(1, len(p_values)):
        delta_mod = abs(mods[i] - mods[i-1])
        delta_jac = abs(jacs[i] - jacs[i-1])
        delta_core = cores[i] - cores[i-1]

        if delta_mod > 0.1 or delta_jac > 0.15:
            critical_found = True
            print(f"CAMBIO ABRUPTO detectado entre P={p_values[i-1]} y P={p_values[i]}:")
            print(f"     ΔModularidad = {delta_mod:+.3f}")
            print(f"     ΔJaccard = {delta_jac:+.3f}")
            print(f"     ΔCore edges = {delta_core:+d}")

    if not critical_found:
        print(f"  No se detectaron cambios abruptos → transición suave")
        print(f"  La periferia se degrada gradualmente (no hay punto crítico discreto)")

    print(f"\n[6] PARADIGMA 1 A LO LARGO DE P (¿cuántas aristas superan el azar?)")
    if all(results_per_p[P].get("paradigma1") for P in p_values):
        print(f"  {'P':>6s}  {'k medio':>8s}  {'P_azar':>8s}  {'Δ mín':>9s}  "
              f"{'BH':>5s}  {'Bonf':>5s}  {'Consenso':>9s}  {'Sparsity':>9s}  "
              f"{'Z(mod)':>8s}  {'Veredicto':>10s}")
        print(f"  {'─'*95}")
        for P in p_values:
            p1 = results_per_p[P]["paradigma1"]
            print(f"  {P:>6.2f}  {p1['k_medio']:>8.2f}  {p1['p_azar_global']:>8.3f}  "
                  f"{p1['margen_min']:>9.5f}  {p1['n_sig_bh']:>5d}  "
                  f"{p1['n_sig_bonferroni']:>5d}  {p1['n_consenso']:>9d}  "
                  f"{p1['sparsity']:>9.3f}  {p1['z_modularity']:>+8.2f}  "
                  f"{p1['veredicto']['GLOBAL']:>10s}")
        print(f"\n  Lectura: al subir P crece k, y con él P_azar = k/(N-1). Una arista")
        print(f"  necesita entonces una persistencia mayor para seguir siendo")
        print(f"  significativa: si el nº de aristas BH cae al aumentar P, lo que se")
        print(f"  está añadiendo es cola de la distribución, no estructura nueva.")

    print(f"\n\n{'═'*90}")
    print(f" VEREDICTO: CORE vs PERIFERIA vs PUNTO CRÍTICO")
    print(f"{'═'*90}")

    jac_p_min = results_per_p[p_values[0]]['jaccard_mean']
    jac_p_max = results_per_p[p_values[-1]]['jaccard_mean']
    decay = (jac_p_min - jac_p_max) / jac_p_min * 100 if jac_p_min > 0 else 0

    core_p_min = results_per_p[p_values[0]]['n_core']
    core_p_max = results_per_p[p_values[-1]]['n_core']

    print(f"\n  Jaccard: P={p_values[0]}→{jac_p_min:.3f} | "
          f"P={p_values[-1]}→{jac_p_max:.3f} | "
          f"cambio={(jac_p_max - jac_p_min) / jac_p_min * 100 if jac_p_min > 0 else 0:+.1f}%")
    print(f"  Core edges: P={p_values[0]}→{core_p_min} | P={p_values[-1]}→{core_p_max}")

    return results_per_p

def plot_sensitivity_summary(results_per_p: Dict, p_values: List[float],
                              labels: List[str], n_models: int, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    ax = axes[0, 0]
    means = [results_per_p[P]['jaccard_mean'] for P in p_values]
    stds = [results_per_p[P]['jaccard_std'] for P in p_values]
    medians = [results_per_p[P]['jaccard_median'] for P in p_values]
    ax.errorbar(p_values, means, yerr=stds, fmt='o-', capsize=5, linewidth=2, markersize=8, label='Media ± σ')
    ax.plot(p_values, medians, 's--', color='orange', linewidth=1.5, markersize=6, label='Mediana')
    ax.set_xlabel('P (Top-P threshold)', fontsize=12)
    ax.set_ylabel('Jaccard Similarity', fontsize=12)
    ax.set_title('Jaccard vs P\n(Caída = periferia inestable)', fontsize=13, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xticks(p_values)

    ax = axes[0, 1]
    n_core = [results_per_p[P]['n_core'] for P in p_values]
    n_periph = [results_per_p[P]['n_periphery'] for P in p_values]
    ax.bar([p-0.02 for p in p_values], n_core, width=0.04, color='#2ecc71', label='Core (≥80% freq)')
    ax.bar([p+0.02 for p in p_values], n_periph, width=0.04, color='#e74c3c', label='Periphery (<80% freq)')
    ax.set_xlabel('P', fontsize=12)
    ax.set_ylabel('Número de aristas', fontsize=12)
    ax.set_title('Core vs Periphery edges\npor P', fontsize=13, fontweight='bold')
    ax.legend(); ax.set_xticks(p_values); ax.grid(True, alpha=0.3, axis='y')

    ax = axes[0, 2]
    jac_random = []
    for P in p_values:
        k_avg = results_per_p[P]['avg_top_size']
        jac_random.append(k_avg / max(2*(NUM_INDICES-1) - k_avg, 1e-6))
    jac_real = [results_per_p[P]['jaccard_mean'] for P in p_values]
    ratios = [r/j if j > 0 else 0 for r, j in zip(jac_real, jac_random)]
    ax.plot(p_values, ratios, 'D-', color='purple', linewidth=2, markersize=8)
    ax.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='Umbral "estable" (10x)')
    ax.axhline(y=20, color='green', linestyle='--', alpha=0.5, label='Umbral "muy estable" (20x)')
    ax.set_xlabel('P', fontsize=12)
    ax.set_ylabel('Jaccard_real / Jaccard_random', fontsize=12)
    ax.set_title('Jaccard Normalizado\n(>10x = estructura real)', fontsize=13, fontweight='bold')
    ax.legend(); ax.set_xticks(p_values); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    mod_means = [results_per_p[P]['modularity_mean'] for P in p_values]
    mod_stds = [results_per_p[P]['modularity_std'] for P in p_values]
    ax.errorbar(p_values, mod_means, yerr=mod_stds, fmt='o-', capsize=5, linewidth=2, color='teal')
    ax.set_xlabel('P', fontsize=12)
    ax.set_ylabel('Modularidad (Louvain)', fontsize=12)
    ax.set_title('Modularidad vs P\n(Cambio abrupto = punto crítico)', fontsize=13, fontweight='bold')
    ax.set_xticks(p_values); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    recip = [results_per_p[P]['reciprocity_mean'] for P in p_values]
    ax.plot(p_values, recip, 'o-', color='coral', linewidth=2, markersize=8)
    ax.set_xlabel('P', fontsize=12)
    ax.set_ylabel('Reciprocidad', fontsize=12)
    ax.set_title('Reciprocidad vs P\n(Aumento = más bidireccionalidad)', fontsize=13, fontweight='bold')
    ax.set_xticks(p_values); ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    data_for_box = [results_per_p[P]['jaccard'] for P in p_values]
    bp = ax.boxplot(data_for_box, tick_labels=[f'P={p}' for p in p_values], patch_artist=True)
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    for patch, color in zip(bp['boxes'], colors[:len(p_values)]):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    ax.set_ylabel('Jaccard Similarity', fontsize=12)
    ax.set_title('Distribución Jaccard por P\n(Dispersión = inestabilidad)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Análisis de Sensibilidad del Umbral P (Top-P) — {n_models} modelos',
                  fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sensitivity_P_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, len(p_values), figsize=(6*len(p_values), 7))
    if len(p_values) == 1: axes = [axes]
    n = len(labels)

    for ax, P in zip(axes, p_values):
        freq_matrix = np.zeros((n, n))
        edge_freq = results_per_p[P]['edge_freq']
        for (u, v), freq in edge_freq.items():
            freq_matrix[u, v] = freq / n_models * 100
        im = ax.imshow(freq_matrix, cmap='YlOrRd', vmin=0, vmax=100, aspect='equal')
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(f'Frecuencia de aristas P={P}\n(%, {n_models} modelos)', fontweight='bold')
        for i in range(n):
            for j in range(n):
                if freq_matrix[i,j] > 0:
                    color = 'white' if freq_matrix[i,j] > 50 else 'black'
                    ax.text(j, i, f'{freq_matrix[i,j]:.0f}', ha='center', va='center',
                            fontsize=7, color=color, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Frecuencia (%)', fraction=0.046, pad=0.04)

    plt.suptitle('Frecuencia de Aristas por P — Core (rojo) vs Periphery (claro)',
                  fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'edge_frequency_heatmaps_P.png'), dpi=150, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(16, 10))
    all_core_edges = set()
    for P in p_values:
        all_core_edges.update(results_per_p[P]['core'].keys())

    if all_core_edges:
        sorted_edges = sorted(all_core_edges,
                             key=lambda e: -max(results_per_p[P]['edge_freq'].get(e,0)
                                               for P in p_values))
        y_labels = [f"{labels[u]}→{labels[v]}" for u,v in sorted_edges]
        y_pos = range(len(sorted_edges))

        bar_height = 0.8 / len(p_values)
        cmap = plt.get_cmap("tab10" if len(p_values) <= 10 else "viridis")
        colors = [cmap(i / max(1, len(p_values) - 1)) for i in range(len(p_values))]

        for idx, P in enumerate(p_values):
            offset = (idx - len(p_values)/2 + 0.5) * bar_height
            freqs = [results_per_p[P]['edge_freq'].get(e, 0)/n_models*100 for e in sorted_edges]
            ax.barh([y + offset for y in y_pos], freqs, height=bar_height,
                   color=colors[idx], label=f'P={P}', alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels, fontsize=9)
        ax.set_xlabel('Frecuencia (%)', fontsize=12)
        ax.set_title(f'Aristas Core (≥80% en algún P) — Evolución con P\n{n_models} modelos',
                     fontsize=13, fontweight='bold')
        ax.axvline(x=80, color='green', linestyle='--', alpha=0.5, label='Umbral core (80%)')
        ax.legend(fontsize=10)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'core_edges_evolution_P.png'), dpi=150, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    gaps = [results_per_p[P]['spectral_gap_mean'] for P in p_values]
    gaps_std = [results_per_p[P]['spectral_gap_std'] for P in p_values]
    ents = [results_per_p[P]['entropy_mean'] for P in p_values]
    ents_std = [results_per_p[P]['entropy_std'] for P in p_values]

    axes[0].errorbar(p_values, gaps, yerr=gaps_std, fmt='o-', capsize=5, linewidth=2)
    axes[0].set_xlabel('P'); axes[0].set_ylabel('Gap espectral (λ₂)')
    axes[0].set_title('Gap Espectral vs P\n(Debe ser CONSTANTE — no depende del grafo)',
                       fontsize=11, fontweight='bold')
    axes[0].set_xticks(p_values); axes[0].grid(True, alpha=0.3)

    axes[1].errorbar(p_values, ents, yerr=ents_std, fmt='o-', capsize=5, linewidth=2, color='purple')
    axes[1].set_xlabel('P'); axes[1].set_ylabel('Entropía (nats)')
    axes[1].set_title('Entropía vs P\n(Debe ser CONSTANTE — no depende del grafo)',
                       fontsize=11, fontweight='bold')
    axes[1].set_xticks(p_values); axes[1].grid(True, alpha=0.3)

    plt.suptitle('Verificación: Métricas Espectrales son independientes de P',
                  fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spectral_constancy_check_P.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nVisualizaciones guardadas en {output_dir}/")

def main_parte3(resultado_fase1: Optional[Dict] = None):
    """PARTE 3 · Etapa 2 (discretización Top-P) + PARADIGMA 1 (probabilístico).

    Flujo:
      0. FASE 1 — estabilidad continua (se ejecuta aquí si no se pasó ya hecha).
         Actúa de compuerta: si la atención no es estable, los p-valores de la
         etapa 3 describen el ruido del operador, no el fenómeno.
      1. Barrido de P ∈ {0.5, 0.6, 0.7, 0.8, 0.9}: discretización, Jaccard, y
         paradigma 1 completo (Poisson-binomial + BH/Bonferroni + permutaciones
         + Z-score) para cada P.
      2. Elección de P* con criterios de estabilidad, parsimonia y potencia
         estadística (no solo Jaccard/AvgTop como antes).
      3. Exportación de las aristas significativas del P ganador.

    CORRECCIÓN: la versión anterior exigía `indices_12.npy` (salida de la
    PARTE 1) para arrancar, aunque nunca lo usaba para nada más que un
    `X_val_sample` que tampoco se llegaba a usar. Como resultado, si PARTE 1
    no se había corrido, PARTE 3 abortaba con "No se encontró indices_12.npy"
    incluso cuando las matrices attention_seed_*.npy (lo único que de verdad
    necesita) ya estaban cacheadas. Ahora carga directamente esas matrices.
    """
    all_matrices = cargar_matrices_cacheadas(SEEDS)
    if all_matrices is None:
        print("\nNo se encontraron las matrices attention_seed_*.npy.")
        print("  Ejecuta antes el entrenamiento por semillas, o copia las matrices a:")
        print(f"    {DIR_MATRICES}")
        return

    os.makedirs(DIR_SENSITIVITY_K, exist_ok=True)
    os.makedirs(DIR_MATRICES, exist_ok=True)

    # ------------------------------------------------------------------
    # η adaptativo: se fija ANTES de discretizar nada, y el valor resultante
    # lo comparten el Paradigma 1 y los paradigmas 2 a 5 (que leen ETA_MARGEN).
    # ------------------------------------------------------------------
    print(f"\n{'#'*90}")
    print(f"# OPERADOR DE CORTE: Top-P_η (única discretización del framework)")
    print(f"{'#'*90}")
    calcular_eta(all_matrices)

    # ------------------------------------------------------------------
    # FASE 1 · estabilidad continua (compuerta de la cadena de evidencia)
    # ------------------------------------------------------------------
    print(f"\n{'#'*90}")
    print(f"# FASE 1: ESTABILIDAD CONTINUA DE LA ATENCIÓN (Etapa 1)")
    print(f"{'#'*90}")
    if resultado_fase1 is None:
        resultado_fase1 = analizar_estabilidad_continua(all_matrices, INDEX_NAMES, SEEDS)
        plot_fase1(resultado_fase1, INDEX_NAMES, DIR_FASE1)
    else:
        print(f"  Reutilizando el resultado ya calculado "
              f"(veredicto: {resultado_fase1['veredicto']['GLOBAL']})")

    # Calibración contra modelo nulo: los umbrales fijos de la Tabla 7 no tienen
    # derivación publicada, y ya nos hicieron leer mal el Jaccard y el Z de
    # Erdős-Rényi. Esto añade el contraste que faltaba.
    # Va FUERA del if: main_fase1() ya calcula la FASE 1 y la pasa hecha, así que
    # metido dentro del `is None` no se ejecutaba nunca en la corrida completa y
    # Te se quedaba con el umbral fijo, vetando las 132 aristas.
    if CALIBRAR_UMBRALES and "calibracion_nulo" not in resultado_fase1:
        try:
            cal = calibrar_metricas_estabilidad(all_matrices, resultado_fase1)
            reportar_calibracion(cal)
            resultado_fase1["calibracion_nulo"] = cal
            with open(os.path.join(DIR_FASE1, "calibracion_nulo.json"), "w",
                      encoding="utf-8") as f:
                json.dump(cal, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  Calibración contra nulo saltada: {type(e).__name__}: {e}")

    # Batería de las 9 métricas contra el nulo uniforme, y veredicto recalculado.
    # Sin esto, cinco de las nueve las aprueba una atención plana (medido: D_JS,
    # Wasserstein, CV_entrada, Entropía_col y Sinks) y L_inf incluso premia al
    # nulo por encima de los datos reales.
    if CALIBRAR_UMBRALES and "calibracion_bateria" not in resultado_fase1:
        try:
            calb = calibrar_bateria_fase1(all_matrices, resultado_fase1)
            reportar_bateria_fase1(calb)
            resultado_fase1["calibracion_bateria"] = calb
            v_antes = resultado_fase1["veredicto"]["GLOBAL"]
            resultado_fase1["veredicto"] = _veredicto_fase1(resultado_fase1, calb)
            v_ahora = resultado_fase1["veredicto"]["GLOBAL"]
            deg = resultado_fase1["veredicto"].get("_degradadas")
            print(f"  Veredicto de la etapa 1: {v_antes} con los umbrales fijos "
                  f"-> {v_ahora} calibrado contra el nulo uniforme.")
            if deg:
                print(f"    Degradadas de APRUEBA a DUDOSO por no ganarle al "
                      f"nulo: {deg}")
            mot = resultado_fase1["veredicto"].get("_motivo_global")
            if mot:
                print(f"    {mot}")
            with open(os.path.join(DIR_FASE1, "calibracion_bateria.json"), "w",
                      encoding="utf-8") as f:
                json.dump(calb, f, indent=2, ensure_ascii=False)
        except Exception as e:
            import traceback
            print(f"  Batería calibrada saltada: {type(e).__name__}: {e}")
            traceback.print_exc()
    guardar_fase1(resultado_fase1, DIR_FASE1)

    eps_medio = resultado_fase1["eps_mean"]
    eps_peor = resultado_fase1["eps_max"]
    if resultado_fase1["veredicto"]["GLOBAL"] == "FALLA":
        print(f"\n  AVISO: la etapa 1 no se supera. Se continúa para poder")
        print(f"  caracterizar el fallo, pero los resultados del paradigma 1")
        print(f"  deben reportarse como diagnóstico, no como hallazgo.")

    # ------------------------------------------------------------------
    # FASE 2 · discretización Top-P + paradigma 1, barrido P ∈ P_VALUES
    # ------------------------------------------------------------------
    print(f"\n{'#'*90}")
    print(f"# FASE 2: DISCRETIZACIÓN P ∈ {P_VALUES} + PARADIGMA 1 (PROBABILÍSTICO)")
    print(f"# Mismas {len(SEEDS)} matrices → diferente umbral P")
    print(f"{'#'*90}")

    results_per_p = analyze_sensitivity_p(all_matrices, INDEX_NAMES, SEEDS, P_VALUES,
                                          eps_perturbacion=eps_medio,
                                          eps_perturbacion_max=eps_peor,
                                          ejecutar_paradigma1=True)

    # ------------------------------------------------------------------
    # FASE 3 · visualizaciones
    # ------------------------------------------------------------------
    print(f"\n{'#'*90}")
    print(f"# FASE 3: GENERANDO VISUALIZACIONES")
    print(f"{'#'*90}")

    plot_sensitivity_summary(results_per_p, P_VALUES, INDEX_NAMES,
                             len(SEEDS), os.path.join(DIR_SENSITIVITY_K, "plots"))
    for P in P_VALUES:
        p1 = results_per_p[P].get("paradigma1")
        if p1:
            plot_paradigma1(p1, INDEX_NAMES, DIR_P1)
    print(f"  Figuras del paradigma 1 en {DIR_P1}/")

    # ------------------------------------------------------------------
    # Persistencia de resultados
    # ------------------------------------------------------------------
    save_dict = {
        "matrices": np.stack(all_matrices),
        "seeds": np.array(SEEDS),
        "p_values": np.array(P_VALUES),
        "index_names": np.array(INDEX_NAMES),
        "eps_fase1": np.array(eps_medio),
        "eps_max_fase1": np.array(eps_peor),
        "linf_medio_fase1": np.array(resultado_fase1["linf_mean"]),
    }
    # tamaños del top por fila, ahora sin pisarse entre valores de P
    for (seed, P), sizes in GLOBAL_TOP_P_SIZES.items():
        save_dict[f"top_p_sizes_seed_{seed}_P{P}"] = np.array(sizes, dtype=int)

    for P in P_VALUES:
        r = results_per_p[P]
        save_dict[f"jaccard_P{P}"] = r["jaccard"]
        save_dict[f"jaccard_mean_P{P}"] = r["jaccard_mean"]
        save_dict[f"n_core_P{P}"] = r["n_core"]
        save_dict[f"modularity_mean_P{P}"] = r["modularity_mean"]
        save_dict[f"reciprocity_mean_P{P}"] = r["reciprocity_mean"]
        save_dict[f"avg_top_size_P{P}"] = r["avg_top_size"]
        p1 = r.get("paradigma1")
        if p1:
            save_dict[f"phi_P{P}"] = p1["phi"]
            save_dict[f"pvals_P{P}"] = p1["pvals"]
            save_dict[f"qvals_P{P}"] = p1["qvals"]
            save_dict[f"sig_bh_P{P}"] = p1["sig_bh"]
            save_dict[f"sig_bonf_P{P}"] = p1["sig_bonferroni"]
            save_dict[f"p_perm_P{P}"] = p1["p_perm"]
            save_dict[f"ci_lo_P{P}"] = p1["ci_lo"]
            save_dict[f"ci_hi_P{P}"] = p1["ci_hi"]
            save_dict[f"k_eff_P{P}"] = p1["k_eff"]
            save_dict[f"margen_medio_fila_P{P}"] = p1["margen_medio_fila"]

    np.savez(os.path.join(DIR_SENSITIVITY_K, "sensitivity_P_results.npz"), **save_dict)

    # ------------------------------------------------------------------
    # Resumen ejecutivo
    # ------------------------------------------------------------------
    print(f"\n\n{'═'*90}")
    print(f" RESUMEN EJECUTIVO — DISCRETIZACIÓN + PARADIGMA 1")
    print(f"{'═'*90}")

    jacs = [results_per_p[P]['jaccard_mean'] for P in P_VALUES]
    cores = [results_per_p[P]['n_core'] for P in P_VALUES]
    mods = [results_per_p[P]['modularity_mean'] for P in P_VALUES]
    avg_sizes = [results_per_p[P]['avg_top_size'] for P in P_VALUES]
    sigs = [results_per_p[P]['n_sig_bh'] for P in P_VALUES]

    print(f"\n  P     Jaccard(μ)   Core   BH   AvgTop   Modularidad   Interpretación")
    print(f"  {'─'*88}")
    for i, P in enumerate(P_VALUES):
        delta_jac = f"({jacs[i]-jacs[i-1]:+.3f})" if i > 0 else "  ---  "
        delta_core = f"({cores[i]-cores[i-1]:+d})" if i > 0 else "  ---  "
        print(f"  {P:<6.2f} {jacs[i]:>8.3f} {delta_jac:>8s}  {cores[i]:>5d} {delta_core:>8s}  "
              f"{sigs[i]:>3d}  {avg_sizes[i]:>7.2f}  {mods[i]:>8.3f}   ", end="")
        if i == 0:
            print("Baseline")
        else:
            # El signo importa: con Top-P_η el Jaccard SUBE al subir P (más
            # aristas, más solapamiento trivial), así que llamarlo "decay" y
            # leerlo como "periferia estable" invertía la conclusión. Lo que
            # informa es el cambio relativo respecto al P más bajo Y, sobre
            # todo, el ratio contra un grafo aleatorio de la misma densidad
            # (tabla [2] de arriba): si ese ratio tiende a 1, lo que se añade
            # al subir P es relleno, no estructura.
            cambio = (jacs[i] - jacs[0]) / jacs[0] * 100 if jacs[0] > 0 else 0
            if cambio > 15:
                print(f" Jaccard {cambio:+.0f}% vs P={P_VALUES[0]} → más denso, "
                      f"revisar ratio vs azar")
            elif cambio < -15:
                print(f" Jaccard {cambio:+.0f}% vs P={P_VALUES[0]} → periferia=ruido")
            else:
                print(f" Jaccard {cambio:+.0f}% vs P={P_VALUES[0]} → estable")

    # ------------------------------------------------------------------
    # Elección de P* — reemplaza el criterio heurístico (solo Jaccard+AvgTop)
    # por uno que exige además potencia estadística real del paradigma 1.
    # ------------------------------------------------------------------
    print(f"\n  P ÓPTIMO (parsimonia + estabilidad + potencia estadística):")
    MAX_AVG_TOP = 8.5   # no más de 8.5 de 11 conexiones posibles (~77% densidad)
    MIN_JACCARD = 0.5   # consenso mínimo inter-semilla
    MIN_SPARSITY = 0.5  # el grafo de consenso debe seguir siendo interpretable

    def _veredicto_de(P):
        p1 = results_per_p[P].get("paradigma1") or {}
        return (p1.get("veredicto") or {}).get("GLOBAL", "FALLA")

    def _ratio_azar(P):
        """Jaccard observado / Jaccard de un grafo aleatorio de igual densidad.

        Es el control decisivo: un Jaccard alto con densidad alta no dice nada,
        porque dos grafos casi completos se solapan por fuerza. Solo cuenta el
        exceso sobre el azar.
        """
        d = results_per_p[P]
        jr = d.get("jaccard_random")
        if jr is None or not np.isfinite(jr) or jr <= 0:
            k = d.get("avg_top_size", 0.0)
            n = len(INDEX_NAMES)
            dens = k / max(n - 1, 1)
            jr = dens / max(2 - dens, 1e-9)      # J esperado bajo azar
        return d["jaccard_mean"] / max(jr, 1e-9)

    # El veredicto GLOBAL del paradigma 1 ahora es una RESTRICCIÓN, no un dato
    # informativo. Antes P* se elegía solo por Jaccard + AvgTop + nº de aristas
    # BH, y podía caer en un P cuyo propio veredicto era DUDOSO o FALLA (pasó:
    # eligió P=0.8 con veredicto DUDOSO habiendo P=0.7 con APRUEBA). Se añade
    # además el ratio contra el azar: sin exceso sobre el azar, el Jaccard no
    # es evidencia de estabilidad.
    MIN_RATIO_AZAR = 1.2

    admisibles = [
        P for i, P in enumerate(P_VALUES)
        if avg_sizes[i] <= MAX_AVG_TOP
        and results_per_p[P]["n_sig_bh"] > 0
        and (np.isnan(results_per_p[P]["sparsity"])
             or results_per_p[P]["sparsity"] >= MIN_SPARSITY)
    ]
    print(f"    {'P':>5}{'Jaccard':>9}{'ratio azar':>12}{'BH':>5}{'AvgTop':>8}"
          f"{'Sparsity':>10}   veredicto P1")
    for i, P in enumerate(P_VALUES):
        print(f"    {P:>5.2f}{jacs[i]:>9.3f}{_ratio_azar(P):>12.2f}"
              f"{results_per_p[P]['n_sig_bh']:>5d}{avg_sizes[i]:>8.2f}"
              f"{results_per_p[P].get('sparsity', float('nan')):>10.3f}"
              f"   {_veredicto_de(P)}")

    aprueban = [P for P in admisibles if _veredicto_de(P) == "APRUEBA"
                and _ratio_azar(P) >= MIN_RATIO_AZAR]
    dudosos = [P for P in admisibles if _veredicto_de(P) == "DUDOSO"
               and _ratio_azar(P) >= MIN_RATIO_AZAR]

    if aprueban:
        # entre los que APRUEBAN, el de mayor exceso sobre el azar; a igualdad,
        # el P más bajo (grafo más compacto y más interpretable)
        best_p = max(aprueban, key=lambda P: (_ratio_azar(P), -P))
        print(f"\n    P* = {best_p}  ·  veredicto P1 = APRUEBA  ·  "
              f"ratio vs azar = {_ratio_azar(best_p):.2f}x  ·  "
              f"{results_per_p[best_p]['n_sig_bh']} aristas BH")
    elif dudosos:
        best_p = max(dudosos, key=lambda P: (_ratio_azar(P), -P))
        print(f"\n    P* = {best_p}  ·  veredicto P1 = DUDOSO  ·  "
              f"ratio vs azar = {_ratio_azar(best_p):.2f}x")
        print(f"    Ningún P aprueba el paradigma 1: el P* elegido es el menos malo.")
    else:
        best_p = P_VALUES[0]
        print(f"\n    P* = {best_p}  (ningún P admisible; se usa el baseline)")
        print(f"    Diagnóstico: revisar el veredicto de la FASE 1, el nº de")
        print(f"    semillas M, y si el ratio contra el azar llega a "
              f"{MIN_RATIO_AZAR}x en algún P.")

    p1_best = results_per_p[best_p].get("paradigma1")
    if p1_best:
        print(f"\n  GRAFO DE CONSENSO EN P* = {best_p}")
        print(f"    Aristas significativas (BH, FDR={FDR_TARGET}): {p1_best['n_sig_bh']}")
        print(f"    Aristas significativas (Bonferroni, α={ALPHA}): {p1_best['n_sig_bonferroni']}")
        print(f"    Consenso (BH ∧ Φ≥{UMBRAL_DE_ESTABILIDAD}): {p1_best['n_consenso']}")
        print(f"    Sparsity S = {p1_best['sparsity']:.3f} · Cobertura C = {p1_best['cobertura']:.3f}")
        print(f"    Veredicto del paradigma 1: {p1_best['veredicto']['GLOBAL']}")
        for (i, j), phi in sorted(p1_best["consenso"].items(),
                                  key=lambda kv: -kv[1])[:15]:
            print(f"      {INDEX_NAMES[i]:12s} → {INDEX_NAMES[j]:12s} : "
                  f"Φ={phi:.2f}  p={p1_best['pvals'][i, j]:.2e}  "
                  f"q={p1_best['qvals'][i, j]:.2e}")

        exportar_aristas_significativas(
            results_per_p, best_p, INDEX_NAMES,
            os.path.join(DIR_P1, "significant_edges_paradigma1.json"))
        exportar_paradigm1_full_json(
            results_per_p, best_p,
            os.path.join(DIR_P1, "paradigm1_results.json"))

    print(f"\n  CADENA DE EVIDENCIA — ESTADO ACTUAL")
    print(f"    Etapa 1 · estabilidad continua : {resultado_fase1['veredicto']['GLOBAL']}")
    print(f"    Etapa 2 · discretización       : "
          f"{'APRUEBA' if jacs[P_VALUES.index(best_p)] > 0.7 else 'DUDOSO' if jacs[P_VALUES.index(best_p)] > 0.4 else 'FALLA'}"
          f"  (Jaccard = {jacs[P_VALUES.index(best_p)]:.3f})")
    print(f"    Etapa 3 · margen y modelo nulo : "
          f"{p1_best['veredicto']['Margen'] if p1_best else 'n/a'} / "
          f"{p1_best['veredicto']['Significancia'] if p1_best else 'n/a'}")
    print(f"    Etapa 4 · fidelidad predictiva : pendiente (paradigma 2, ablación ΔMSE)")

    print(f"\n{'═'*90}")

    return results_per_p


if __name__ == "__main__":
    if EJECUTAR_PARTE1:
        print("\n>>> EJECUTANDO PARTE 1: Preprocesamiento de datos <<<")
        stack = descargar_y_alinear_12_indices()
        if stack is not None:
            print(f"\nResumen:")
            print(f"  Shape: {stack.shape}  (N_dates, H, W, 12_indices)")
            print(f"  Índices: {INDEX_NAMES}")
            print(f"  Listo para entrenar ConvTransformer con attention rollout")
    if EJECUTAR_PARTE2_MULTISEED:
        print("\n>>> EJECUTANDO PARTE 2b: Entrenamiento por semilla (attention_seed_*.npy) <<<")
        main_parte2_multiseed()
    preparar_matrices_atencion()
    _fase1 = None
    if EJECUTAR_FASE1:
        print("\n>>> EJECUTANDO FASE 1: Estabilidad continua de la atención <<<")
        _fase1 = main_fase1()
    if EJECUTAR_PARTE3:
        print("\n>>> EJECUTANDO PARTE 3: Discretización Top-P + Paradigma 1 <<<")
        main_parte3(_fase1)

# ============================================================================
# Utilidades compartidas: carga de matrices de atención cacheadas
# (usadas por Paradigma 1 en PARTE 3, y por Paradigmas 2, 3, 4 y 5)
# ============================================================================

def find_cached_matrices() -> Tuple[List[np.ndarray], List[int]]:
    for d in SEARCH_DIRS:
        if not os.path.isdir(d):
            continue
        files = sorted([f for f in os.listdir(d)
                        if f.startswith("attention_seed_") and f.endswith(".npy")])
        if len(files) >= 5:
            print(f"   Encontradas {len(files)} matrices cacheadas en: {d}")
            mats, seeds = [], []
            for f in files:
                try:
                    s = int(f.replace("attention_seed_", "").replace(".npy", ""))
                except ValueError:
                    s = len(mats)
                mats.append(np.load(os.path.join(d, f)))
                seeds.append(s)
            return mats, seeds
    return [], []

def load_matrices() -> Tuple[List[np.ndarray], List[int]]:
    mats, seeds = find_cached_matrices()
    if mats:
        return mats, seeds
    else:
        return


# ============================================================================
# Paradigma 2
# ============================================================================

def load_matrices():
    # Búsqueda unificada: usa SEARCH_DIRS definido en la SECCIÓN 0
    # (primero resultados/sensitivity_K_experiment/matrices, luego fallbacks).
    search = list(SEARCH_DIRS)
    for d in search:
        if not os.path.isdir(d):
            continue
        files = sorted([f for f in os.listdir(d)
                        if f.startswith("attention_seed_") and f.endswith(".npy")])
        if len(files) >= 5:
            print(f"  ✅ Matrices cacheadas: {d} ({len(files)} archivos)")
            return [np.load(os.path.join(d, f)) for f in files]
    rng = np.random.default_rng(42)
    n = NUM_INDICES
    K_STABLE = 3
    stable_targets = []
    for i in range(n):
        candidates = [j for j in range(n) if j != i]
        order = np.random.default_rng(42 + i).permutation(len(candidates))
        stable_targets.append([candidates[order[k]] for k in range(K_STABLE)])
    mats = []
    for _ in range(20):
        A = rng.dirichlet(np.ones(n) * 0.5, size=n).astype(np.float64)
        for i in range(n):
            for j in stable_targets[i]:
                A[i, j] += rng.uniform(3.0, 5.0)
        np.fill_diagonal(A, 0.0)
        A = A / np.maximum(A.sum(axis=1, keepdims=True), 1e-12)
        mats.append(A.astype(np.float32))
    os.makedirs(CACHE_DIR, exist_ok=True)
    for i, m in enumerate(mats):
        np.save(os.path.join(CACHE_DIR, f"attention_seed_{i}.npy"), m)
    print(f"  ⚠️  Generadas {len(mats)} matrices sintéticas (fallback).")
    return mats

def softmax_row(row: np.ndarray) -> np.ndarray:
    r = row - row.max()
    e = np.exp(r)
    return e / e.sum()


def ablate_soft(A: np.ndarray, i: int, j: int, alpha: float) -> np.ndarray:
    A_new = A.copy().astype(np.float64)
    A_new[i, j] = alpha * A_new[i, j]
    A_new[i] = softmax_row(np.log(np.maximum(A_new[i], 1e-12)))
    return A_new

def row_entropy(p: np.ndarray) -> float:
    p = np.maximum(p, 1e-12)
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))

def matrix_entropy(A: np.ndarray) -> float:
    return float(np.mean([row_entropy(A[i]) for i in range(A.shape[0])]))

# topk_set() eliminado: la única discretización del framework es discretizar()
# (Top-P_η), definida junto a _select_top_p. Ver el bloque
# "DISCRETIZACIÓN A → G: SOLO TOP-P" en la SECCIÓN 0.

def jaccard(s1: set, s2: set) -> float:
    if not s1 and not s2:
        return 1.0
    u = s1 | s2
    return len(s1 & s2) / len(u) if u else 0.0

@dataclass
class AblationResult:
    i: int; j: int
    label_i: str; label_j: str
    auc_degradation: float
    alpha_critical: float
    max_degradation: float
    curve_alpha: List[float]
    curve_degradation: List[float]
    jaccard_drop: float

def ablation_curve_for_edge(A: np.ndarray, i: int, j: int,
                            p: Optional[float] = None
                            ) -> Tuple[List[float], List[float], float, float, float]:
    """Curva de ablación suave de una arista.

    AVISO SOBRE QUÉ MIDE ESTO. El informe (§8.2) define la métrica de fidelidad
    como ΔMSE(e) = MSE(Ã(e)) − MSE(A), que exige RE-EJECUTAR el modelo con la
    matriz ablacionada. Esta función mide el cambio de ENTROPÍA de la propia
    matriz, sin tocar el modelo ni los datos: es una propiedad algebraica de la
    matriz 12×12. Por eso el umbral AUC > 0.005 de la Tabla 7, que está definido
    para la curva de MSE de predicción, no es comparable con el AUC que sale de
    aquí (rango medido: 0.0018 a 0.0033, con 0/120 aristas "aprobando").
    Mientras no se implemente el ΔMSE real, este AUC vale como RANKING relativo
    entre aristas, no como criterio de aceptación contra la Tabla 7.
    """
    base_entropy = matrix_entropy(A)
    base_set = discretizar(A, p)

    degradations = []
    jaccards = []
    for alpha in ALPHA_GRID:
        A_abl = ablate_soft(A, i, j, alpha)
        ent = matrix_entropy(A_abl)
        deg = ent - base_entropy
        degradations.append(deg)
        jaccards.append(jaccard(base_set, discretizar(A_abl, p)))

    degradations = np.array(degradations)
    order = np.argsort(ALPHA_GRID)
    alpha_asc = np.array(ALPHA_GRID)[order]
    deg_asc = degradations[order]
    auc = float(np.trapezoid(np.abs(deg_asc), alpha_asc))
    deriv = np.abs(np.gradient(deg_asc, alpha_asc))
    alpha_crit = float(alpha_asc[int(np.argmax(deriv))])
    max_deg = float(np.max(np.abs(degradations)))
    jaccard_drop = 1.0 - float(jaccards[-1])

    return ALPHA_GRID.tolist(), degradations.tolist(), auc, alpha_crit, max_deg, jaccard_drop

def run_ablation_analysis(matrices: List[np.ndarray], p: Optional[float] = None
                          ) -> List[AblationResult]:
    A_mean = np.mean(matrices, axis=0)
    # Antes se elegían top-K y bottom-K por fila como candidatas. Con N=12 hay
    # solo N·(N-1)=132 aristas posibles y la curva de ablación es barata, así
    # que se analizan TODAS: desaparece el K y no hay sesgo de selección por
    # haber elegido las candidatas con un criterio distinto al del grafo.
    candidates = [(i, j) for i in range(NUM_INDICES)
                  for j in range(NUM_INDICES) if i != j]
    en_grafo = discretizar(A_mean, p)

    results: List[AblationResult] = []
    print(f"  Analizando las {len(candidates)} aristas posibles "
          f"(todas, sin self-loops). De ellas {len(en_grafo)} están en el "
          f"grafo Top-P.")
    for (i, j) in candidates:
        alphas, degs, auc, a_crit, max_deg, jac_drop = \
            ablation_curve_for_edge(A_mean, i, j, p=p)
        results.append(AblationResult(
            i=i, j=j,
            label_i=INDEX_NAMES[i], label_j=INDEX_NAMES[j],
            auc_degradation=auc, alpha_critical=a_crit,
            max_degradation=max_deg,
            curve_alpha=alphas, curve_degradation=degs,
            jaccard_drop=jac_drop,
        ))
    return results

def edge_swap_counterfactual(A: np.ndarray, e1: Tuple[int, int],
                              e2: Tuple[int, int],
                              p: Optional[float] = None) -> Dict:
    A_swap = A.copy()
    A_swap[e1], A_swap[e2] = A_swap[e2], A_swap[e1]
    base_set = discretizar(A, p)
    swap_set = discretizar(A_swap, p)
    return {
        "jaccard_preserved": jaccard(base_set, swap_set),
        "entropy_delta": matrix_entropy(A_swap) - matrix_entropy(A),
        "structure_preserved": jaccard(base_set, swap_set) > 0.7,
    }

def _mse_con_atencion(model, X_val: np.ndarray, Y_val: Optional[np.ndarray],
                      A: np.ndarray, bs: int = 16) -> float:
    """MSE de validación del modelo cuando se le impone la matriz A.

    Soporta las dos tareas. Con TAREA="enmascarado", Y_val es None y X_val son
    los frames: se construyen las muestras enmascaradas y el error se mide SOLO
    en el canal oculto, igual que en el entrenamiento. Antes esta función
    asumía la tarea de predicción, así que con la tarea enmascarada el ΔMSE
    reventaba y el Paradigma 2 lo dejaba en None.
    """
    dev = next(model.parameters()).device
    At = torch.from_numpy(np.asarray(A, dtype=np.float32)).to(dev)
    total, n = 0.0, 0

    if TAREA == "enmascarado":
        ds = IndicesEnmascaradosDataset(X_val, NUM_INDICES)
        with torch.no_grad():
            for i in range(0, len(ds), bs):
                lote = [ds[k] for k in range(i, min(i + bs, len(ds)))]
                xb = torch.stack([b[0] for b in lote]).to(dev)
                yb = torch.stack([b[1] for b in lote]).to(dev)
                mb = torch.stack([b[2] for b in lote]).to(dev)
                pred = model.forward_con_atencion(xb, At)
                err2 = ((pred - yb) ** 2) * mb.unsqueeze(-1).unsqueeze(-1)
                total += float(err2.sum().item())
                n += int(mb.sum().item()) * yb.shape[-1] * yb.shape[-2]
        return total / max(n, 1)

    Yt = np.transpose(Y_val, (0, 3, 1, 2))
    with torch.no_grad():
        for i in range(0, len(X_val), bs):
            xb = torch.from_numpy(np.ascontiguousarray(
                np.transpose(X_val[i:i + bs], (0, 4, 1, 2, 3)))).float().to(dev)
            pred = model.forward_con_atencion(xb, At).float().cpu().numpy()
            err = pred - Yt[i:i + bs]
            total += float((err ** 2).sum())
            n += err.size
    return total / max(n, 1)


def delta_mse_ablacion(model, X_val: np.ndarray, Y_val: np.ndarray,
                       A_base: np.ndarray,
                       aristas: Optional[List[Tuple[int, int]]] = None,
                       alphas: Optional[List[float]] = None,
                       verbose: bool = True) -> Dict:
    """ΔMSE(e) = MSE(Ã(e)) − MSE(A), la métrica de fidelidad del informe §8.2.

    Ã(e) se construye con ablate_soft: se atenúa la entrada (i,j) por α y se
    RENORMALIZA la fila con softmax(log(·)), que es exactamente el operador que
    define el informe. Con α=0 la arista se elimina del todo.

    Devuelve, por arista, el ΔMSE a cada α, el AUC de la curva de degradación
    del MSE (no de la entropía) y el α crítico. Estos sí son comparables con los
    umbrales de la Tabla 7: ΔMSE > 0.01 aprueba, |ΔMSE| < 0.001 falla;
    AUC > 0.005 aprueba, AUC ≈ 0 falla.
    """
    if alphas is None:
        alphas = [1.0, 0.75, 0.5, 0.25, 0.0]
    n = A_base.shape[0]
    if aristas is None:
        aristas = [(i, j) for i in range(n) for j in range(n) if i != j]

    mse_base = _mse_con_atencion(model, X_val, Y_val, A_base)
    if verbose:
        print(f"      MSE con la matriz de consenso sin ablacionar: {mse_base:.6f}")
        print(f"      Ablacionando {len(aristas)} aristas × {len(alphas)} valores de α...")

    resultados = []
    for idx, (i, j) in enumerate(aristas, 1):
        curva = []
        for a in alphas:
            A_abl = ablate_soft(A_base, i, j, a)
            curva.append(_mse_con_atencion(model, X_val, Y_val, A_abl) - mse_base)
        orden = np.argsort(alphas)                      # α ascendente para el AUC
        aa = np.array(alphas, dtype=np.float64)[orden]
        cc = np.array(curva, dtype=np.float64)[orden]
        auc = float(np.trapezoid(np.abs(cc), aa))
        resultados.append(dict(
            i=int(i), j=int(j),
            label_i=INDEX_NAMES[i], label_j=INDEX_NAMES[j],
            delta_mse_alpha0=float(curva[alphas.index(0.0)] if 0.0 in alphas else cc[0]),
            auc_mse=auc,
            alpha_critico=float(aa[int(np.argmax(np.abs(np.gradient(cc, aa))))]),
            curva_alpha=[float(x) for x in alphas],
            curva_delta_mse=[float(x) for x in curva],
        ))
        if verbose and idx % 20 == 0:
            print(f"        {idx}/{len(aristas)}")

    dm = np.array([r["delta_mse_alpha0"] for r in resultados])
    au = np.array([r["auc_mse"] for r in resultados])
    for r in resultados:
        r["delta_mse_rel"] = float(r["delta_mse_alpha0"] / max(mse_base, 1e-12))
    return dict(mse_base=mse_base, alphas=alphas, aristas=resultados,
                delta_mse_media=float(dm.mean()), delta_mse_max=float(dm.max()),
                delta_mse_min=float(dm.min()),
                delta_mse_rel_media=float(dm.mean() / max(mse_base, 1e-12)),
                delta_mse_rel_max=float(dm.max() / max(mse_base, 1e-12)),
                auc_media=float(au.mean()), auc_max=float(au.max()),
                # Se conservan los conteos con el umbral fijo de la Tabla 7 para
                # poder citar que NINGUNA arista lo alcanza, pero el veredicto
                # ya no se toma con ellos: ver calibrar_delta_mse.
                n_aprueba_delta=int((dm > 0.01).sum()),
                n_falla_delta=int((np.abs(dm) < 0.001).sum()),
                n_aprueba_auc=int((au > 0.005).sum()),
                umbral_tabla7_alcanzable=bool(dm.max() > 0.01))


def calibrar_delta_mse(res_delta: Dict,
                       aristas_grafo: List[Tuple[int, int]]) -> Dict:
    """Separa el ΔMSE de las aristas DEL grafo del de las que quedaron fuera.

    POR QUÉ. El umbral absoluto de la Tabla 7 (ΔMSE > 0.01) no lo alcanza
    ninguna arista con este mecanismo de ablación, así que no discrimina: las
    reprueba todas. La pregunta que sí se puede responder es comparativa —
    ¿degrada más quitar una arista que el operador Top-P seleccionó que una que
    descartó? Si no, el grafo no señala nada y da igual qué aristas contenga.

    Necesita que `res_delta` incluya tanto aristas del grafo como no-aristas;
    el caller las pasa juntas a delta_mse_ablacion y aquí se separan.
    """
    en_grafo = set((int(i), int(j)) for i, j in aristas_grafo)
    dentro = [r for r in res_delta["aristas"]
              if (r["i"], r["j"]) in en_grafo]
    fuera = [r for r in res_delta["aristas"]
             if (r["i"], r["j"]) not in en_grafo]
    if not dentro:
        return {}
    base = max(res_delta["mse_base"], 1e-12)
    d_in = np.array([r["delta_mse_alpha0"] for r in dentro])
    out = dict(n_dentro=len(dentro), n_fuera=len(fuera), mse_base=base,
               rel_min=DELTA_MSE_REL_MIN, pctl_nulo=DELTA_MSE_PCTL_NULO)
    if fuera:
        d_out = np.array([r["delta_mse_alpha0"] for r in fuera])
        umbral_nulo = float(np.percentile(d_out, DELTA_MSE_PCTL_NULO))
        out.update(nulo_media=float(d_out.mean()), nulo_max=float(d_out.max()),
                   umbral_nulo=umbral_nulo,
                   n_supera_nulo=int((d_in > umbral_nulo).sum()),
                   # Mann-Whitney: ¿la distribución de dentro está desplazada?
                   u_p=float(stats.mannwhitneyu(d_in, d_out,
                                                alternative="greater").pvalue))
    else:
        out.update(umbral_nulo=None, n_supera_nulo=None, u_p=None)
    out["n_supera_rel"] = int((d_in / base > DELTA_MSE_REL_MIN).sum())
    out["dentro_media_rel"] = float(d_in.mean() / base)
    out["dentro_max_rel"] = float(d_in.max() / base)
    return out


def reportar_calibracion_delta_mse(cal: Dict) -> None:
    if not cal:
        return
    print(f"\n      CALIBRACIÓN DEL ΔMSE (sustituye al umbral fijo de 0.01)")
    print(f"        mse_base {cal['mse_base']:.6f} · "
          f"{cal['n_dentro']} aristas del grafo · "
          f"{cal['n_fuera']} no-aristas de control")
    print(f"        ΔMSE relativo dentro del grafo: "
          f"media {cal['dentro_media_rel']*100:.3f}% · "
          f"máx {cal['dentro_max_rel']*100:.3f}%")
    print(f"        superan el {DELTA_MSE_REL_MIN*100:.1f}% relativo: "
          f"{cal['n_supera_rel']}/{cal['n_dentro']}")
    if cal.get("umbral_nulo") is not None:
        print(f"        control (no-aristas): media "
              f"{cal['nulo_media']:+.6f} · percentil "
              f"{cal['pctl_nulo']:.0f} = {cal['umbral_nulo']:+.6f}")
        print(f"        superan el control: "
              f"{cal['n_supera_nulo']}/{cal['n_dentro']}"
              f"   Mann-Whitney p = {cal['u_p']:.4f}")
        if cal["u_p"] < 0.05:
            print(f"        -> las aristas del grafo degradan MÁS que las "
                  f"descartadas: el operador señala algo.")
        else:
            print(f"        -> indistinguible: quitar una arista del grafo "
                  f"degrada igual que quitar una que el operador descartó.")
    else:
        print(f"        sin no-aristas de control: pasa "
              f"N_NOARISTAS_NULO > 0 para calibrar.")


def run_edge_swap_analysis(matrices: List[np.ndarray],
                           p: Optional[float] = None) -> List[Dict]:
    A_mean = np.mean(matrices, axis=0)
    # "Dentro del grafo" y "fuera del grafo" según Top-P, no según un K fijo:
    # se intercambian aristas seleccionadas por el operador con aristas que el
    # operador descartó, que es el contrafactual que describe el informe.
    en_grafo = discretizar(A_mean, p)
    top_edges = sorted(en_grafo)
    bottom_edges = [(i, j) for i in range(NUM_INDICES)
                    for j in range(NUM_INDICES)
                    if i != j and (i, j) not in en_grafo]

    # Intercambios por LOTES. Un solo swap deja el Jaccard clavado en
    # (|E|-2)/|E| por aritmética pura; con k = SWAP_FRACCION·|E| swaps
    # simultáneos el Jaccard sí varía según QUÉ aristas se intercambien, que es
    # lo que el contrafactual quiere medir.
    por_fila: Dict[int, List[Tuple[int, int]]] = {}
    for e2 in bottom_edges:
        por_fila.setdefault(e2[0], []).append(e2)
    candidatos = [e1 for e1 in top_edges if por_fila.get(e1[0])]
    if not candidatos:
        return []

    k = max(1, int(round(SWAP_FRACCION * len(candidatos))))
    rng = np.random.default_rng(SEMILLA_NULO)
    base = set(en_grafo)
    results = []
    for rep in range(SWAP_REPETICIONES):
        idx = rng.choice(len(candidatos), min(k, len(candidatos)), replace=False)
        fuera_sel, dentro_sel = [], []
        for t in idx:
            e1 = candidatos[int(t)]
            opciones = por_fila[e1[0]]
            e2 = opciones[int(rng.integers(len(opciones)))]
            fuera_sel.append(e1)
            dentro_sel.append(e2)
        nuevo = (base - set(fuera_sel)) | set(dentro_sel)
        inter = len(base & nuevo)
        union = len(base | nuevo)
        jac = inter / union if union else 1.0
        results.append({
            "rep": rep, "k_intercambiadas": len(fuera_sel),
            "e1_labels": [f"{INDEX_NAMES[a]}→{INDEX_NAMES[b]}"
                          for a, b in fuera_sel[:3]],
            "e2_labels": [f"{INDEX_NAMES[a]}→{INDEX_NAMES[b]}"
                          for a, b in dentro_sel[:3]],
            "jaccard_preserved": float(jac),
            "structure_preserved": bool(jac > 0.7),
        })
    return results

def plot_ablation_curves(results: List[AblationResult], output_path: str):
    fig, ax = plt.subplots(figsize=(12, 7))
    results_sorted = sorted(results, key=lambda r: -r.auc_degradation)
    top_n = min(8, len(results_sorted))
    cmap = plt.cm.viridis(np.linspace(0, 1, top_n))

    for idx, r in enumerate(results_sorted[:top_n]):
        ax.plot(r.curve_alpha, r.curve_degradation, "-o",
                color=cmap[idx], linewidth=2, markersize=4,
                label=f"{r.label_i}→{r.label_j} (AUC={r.auc_degradation:.3f})")
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax.axvline(x=0, color="red", linestyle=":", alpha=0.5, label="α=0 (eliminación)")
    ax.set_xlabel("α (factor de atenuación)", fontsize=11)
    ax.set_ylabel("Δ Entropía de la matriz (proxy degradación)", fontsize=11)
    ax.set_title("Curvas de Ablación Suave — Top aristas por AUC de degradación\n"
                "Curva empinada → relación 'todo o nada' | Curva suave → relación umbral",
                fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight"); plt.close()

def plot_auc_bar(results: List[AblationResult], output_path: str):
    sorted_r = sorted(results, key=lambda r: -r.auc_degradation)
    labels = [f"{r.label_i}→{r.label_j}" for r in sorted_r]
    aucs = [r.auc_degradation for r in sorted_r]
    colors = ["#e74c3c" if a > np.median(aucs) else "#3498db" for a in aucs]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(labels)), aucs, color=colors, edgecolor="white")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("AUC de degradación (∫|ΔH| dα)", fontsize=11)
    ax.set_title("Ranking de aristas por impacto causal (AUC de ablación suave)\n"
                "Rojo = impacto alto | Azul = impacto bajo (prescindible)",
                fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")
    for bar, v in zip(bars, aucs):
        ax.text(v + 0.001, bar.get_y() + bar.get_height()/2,
                f"{v:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight"); plt.close()

def plot_alpha_critical(results: List[AblationResult], output_path: str):
    alphas = [r.alpha_critical for r in results]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(alphas, bins=np.linspace(0, 1, 11), color="#9b59b6",
            edgecolor="white", alpha=0.8)
    ax.set_xlabel("α crítica (punto de máxima pendencia)", fontsize=11)
    ax.set_ylabel("# aristas", fontsize=11)
    ax.set_title("Distribución de α crítica\n"
                "α≈0 → relación 'todo o nada' | α≈0.5 → relación de umbral",
                fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight"); plt.close()


def plot_edge_swap_heatmap(swap_results: List[Dict], output_path: str):
    """Distribución del Jaccard tras intercambiar un lote de aristas.

    Antes era un mapa de calor arista1 x arista2 con un swap por celda, pero con
    un solo intercambio el Jaccard es constante por aritmética y el mapa salía
    de un color plano. Con lotes lo informativo es la DISPERSIÓN: si depende de
    qué aristas se intercambien, la identidad de las aristas importa.
    """
    jac = np.array([r["jaccard_preserved"] for r in swap_results], dtype=float)
    if jac.size == 0:
        return
    k = swap_results[0].get("k_intercambiadas", "?")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(jac, bins=min(30, max(5, jac.size // 6)),
            color="#4C72B0", edgecolor="white")
    ax.axvline(0.7, color="red", ls="--",
               label="umbral de estructura preservada (0.7)")
    ax.axvline(float(jac.mean()), color="black", ls="-",
               label=f"media = {jac.mean():.4f}")
    ax.set_xlabel("Jaccard(grafo original, grafo con el lote intercambiado)",
                  fontsize=11)
    ax.set_ylabel("nº de sorteos", fontsize=11)
    ax.set_title(f"Edge swapping por lotes — {jac.size} sorteos de {k} aristas\n"
                 f"σ = {jac.std():.4f}   "
                 f"(σ ≈ 0 significaría que el test no mide nada)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

def main(p: Optional[float] = None):
    print(f"\n{'═'*82}")
    print(f" PARADIGMA 2 — INTERVENCIONAL (CAUSAL)")
    print(f" Pregunta: ¿Es necesaria la arista?")
    print(f" Fundamento: Ablación suave, AUC de degradación, edge swapping (Pearl)")
    print(f"{'═'*82}")

    matrices = load_matrices()
    M = len(matrices)
    print(f"  Modelos disponibles: M = {M}")
    print(f"  Matriz consenso:     promedio de {M} matrices")
    A_mean = np.mean(matrices, axis=0)
    print(f"  Entropía base H(A): {matrix_entropy(A_mean):.4f}")
    calcular_eta(matrices)
    p = p_efectivo(p)
    print(f"  Discretización: Top-P con P = {p} · η = {ETA_MARGEN:.6f}")
    print(f"  Aristas por fila tras discretizar: "
          f"{tamano_medio_discretizacion(matrices, p):.2f} de {NUM_INDICES - 1}")

    print(f"\n  [1] ABLACIÓN SUAVE (α ∈ [1.0 → 0.0], {len(ALPHA_GRID)} puntos)")
    ablation_results = run_ablation_analysis(matrices, p=p)
    print(f"      ✅ {len(ablation_results)} aristas analizadas")

    auc_values = np.array([r.auc_degradation for r in ablation_results])
    print(f"\n      AUC de degradación — media: {auc_values.mean():.4f} | "
          f"mediana: {np.median(auc_values):.4f} | "
          f"máx: {auc_values.max():.4f} | mín: {auc_values.min():.4f}")

    print(f"\n      TOP 8 ARISTAS POR AUC (relaciones funcionalmente críticas):")
    print(f"        {'Arista':22s} {'AUC':>8s} {'α_crit':>8s} {'MaxDeg':>8s} "
          f"{'J_drop':>8s} {'Perfil':>15s}")
    print(f"        {'─'*70}")
    sorted_abl = sorted(ablation_results, key=lambda r: -r.auc_degradation)
    for r in sorted_abl[:8]:
        if r.alpha_critical < 0.1:
            perfil = "Todo o nada"
        elif r.alpha_critical > 0.6:
            perfil = "Umbral tardío"
        else:
            perfil = "Umbral medio"
        print(f"        {r.label_i:10s}→{r.label_j:10s} "
              f"{r.auc_degradation:>8.4f} {r.alpha_critical:>8.2f} "
              f"{r.max_degradation:>8.4f} {r.jaccard_drop:>8.4f} {perfil:>15s}")

    print(f"\n      ARISTAS CON AUC BAJO (prescindibles / redundantes):")
    for r in sorted_abl[-4:]:
        print(f"        {r.label_i:10s}→{r.label_j:10s} AUC={r.auc_degradation:.4f} "
              f"(causalmente prescindible)")

    # ------------------------------------------------------------------
    # [1b] ΔMSE REAL — la métrica de fidelidad del informe (§8.2, Tabla 7)
    # ------------------------------------------------------------------
    print(f"\n  [1b] ΔMSE DE ABLACIÓN (re-ejecutando el modelo, §8.2)")
    delta_results = None
    ck_dmse = None
    if EJECUTAR_DELTA_MSE:
        vl_path = os.path.join(DIR_MATRICES, "val_losses.json")
        if not os.path.exists(OUTPUT_NPY):
            print(f"      Saltado: falta {OUTPUT_NPY} (corre la PARTE 1).")
        elif not os.path.exists(vl_path):
            print(f"      Saltado: falta val_losses.json (corre la PARTE 2b).")
        else:
            try:
                with open(vl_path) as f:
                    vl = json.load(f)
                seed_best = int(min(vl, key=vl.get))
                ck_dmse = os.path.join(CHECKPOINT_DIR, f"model_seed_{seed_best}.pth")
                stack_d = np.load(OUTPUT_NPY)
                fechas_d = cargar_fechas_stack(len(stack_d))
                (_, _), (Xv_d, Yv_d), img_d, _ = process_indices_data(
                    stack_d, seq_length=SEQ_LENGTH, dates_millis=fechas_d)
                mdl = ConvTransformer(num_indices=NUM_INDICES, seq_length=SEQ_LENGTH,
                                      img_size=img_d, num_heads=4, num_layers=2)
                mdl.load_state_dict(torch.load(ck_dmse, map_location="cpu"))
                mdl = mdl.to(device).eval()
                print(f"      Modelo: semilla {seed_best} · "
                      f"{len(Xv_d)} muestras de validación")
                aristas_eval = sorted(discretizar(A_mean, p))
                # Además del grafo, se ablacionan no-aristas de CONTROL: las que
                # el operador Top-P descartó. Sin ellas no hay forma de saber si
                # un ΔMSE de +0.002 es señal o es lo que da cualquier arista.
                n_nodos = A_mean.shape[0]
                fuera = [(i, j) for i in range(n_nodos) for j in range(n_nodos)
                         if i != j and (i, j) not in set(aristas_eval)]
                rng_ctrl = np.random.default_rng(SEMILLA_NULO)
                if fuera and N_NOARISTAS_NULO > 0:
                    k_ctrl = min(N_NOARISTAS_NULO, len(fuera))
                    idx_c = rng_ctrl.choice(len(fuera), k_ctrl, replace=False)
                    control = [fuera[int(t)] for t in idx_c]
                else:
                    control = []
                print(f"      Aristas evaluadas: las {len(aristas_eval)} del "
                      f"grafo Top-P + {len(control)} no-aristas de control")
                delta_results = delta_mse_ablacion(
                    mdl, Xv_d, Yv_d, A_mean,
                    aristas=sorted(set(aristas_eval) | set(control)))
                cal_dmse = calibrar_delta_mse(delta_results, aristas_eval)
                delta_results["calibracion"] = cal_dmse
                delta_results["aristas_grafo"] = [list(map(int, e))
                                                  for e in aristas_eval]
                reportar_calibracion_delta_mse(cal_dmse)
                d = delta_results
                print(f"\n      ΔMSE(e) con α=0 (arista eliminada):")
                print(f"        media {d['delta_mse_media']:+.6f} · "
                      f"máx {d['delta_mse_max']:+.6f} · mín {d['delta_mse_min']:+.6f}")
                print(f"      AUC del MSE: media {d['auc_media']:.6f} · "
                      f"máx {d['auc_max']:.6f}")
                print(f"\n      Contra los umbrales de la Tabla 7 "
                      f"(se citan, no se usan para el veredicto):")
                if not d.get("umbral_tabla7_alcanzable", True):
                    print(f"        NINGUNA arista alcanza ΔMSE > 0.01: con este "
                          f"mecanismo de ablación ese umbral no discrimina,")
                    print(f"        reprueba las {len(d['aristas'])} por igual. "
                          f"El veredicto lo da la calibración de arriba.")
                print(f"        ΔMSE > 0.01 (aprueba) : "
                      f"{d['n_aprueba_delta']}/{len(d['aristas'])}")
                print(f"        |ΔMSE| < 0.001 (falla): "
                      f"{d['n_falla_delta']}/{len(d['aristas'])}")
                print(f"        AUC > 0.005 (aprueba) : "
                      f"{d['n_aprueba_auc']}/{len(d['aristas'])}")
                top = sorted(d["aristas"], key=lambda r: -r["delta_mse_alpha0"])[:10]
                print(f"\n      TOP 10 ARISTAS POR ΔMSE (las que de verdad importan):")
                print(f"        {'arista':<28}{'ΔMSE(α=0)':>12}{'AUC':>11}")
                for r in top:
                    print(f"        {r['label_i']+'→'+r['label_j']:<28}"
                          f"{r['delta_mse_alpha0']:>+12.6f}{r['auc_mse']:>11.6f}")
                peor = sorted(d["aristas"], key=lambda r: r["delta_mse_alpha0"])[:3]
                print(f"\n      ΔMSE NEGATIVO (quitarlas MEJORA el modelo → artefacto):")
                for r in peor:
                    print(f"        {r['label_i']+'→'+r['label_j']:<28}"
                          f"{r['delta_mse_alpha0']:>+12.6f}")
                del mdl
                gc.collect()
            except Exception as e:
                import traceback
                print(f"      Saltado por error: {type(e).__name__}: {e}")
                traceback.print_exc(limit=3)
                delta_results = None
    else:
        print(f"      Desactivado (EJECUTAR_DELTA_MSE = False).")

    print(f"\n  [2] EDGE SWAPPING (contrafactual estructural de Pearl)")
    swap_results = run_edge_swap_analysis(matrices, p=p)
    n_swap = len(swap_results)
    n_preserved = sum(1 for r in swap_results if r["structure_preserved"])
    k_sw = swap_results[0]["k_intercambiadas"] if swap_results else 0
    print(f"      {n_swap} sorteos, {k_sw} aristas intercambiadas por "
          f"sorteo ({SWAP_FRACCION:.0%} del grafo)")
    print(f"      Estructura preservada (Jaccard>0.7): {n_preserved}/{n_swap} "
          f"({100*n_preserved/max(n_swap,1):.1f}%)")
    jacs = [r["jaccard_preserved"] for r in swap_results]
    print(f"      Jaccard medio: {np.mean(jacs):.4f} ± {np.std(jacs):.4f}"
          f"   [{np.min(jacs):.4f}, {np.max(jacs):.4f}]")
    # Si la desviación es ~0 el test no mide nada: sale el mismo Jaccard
    # sin importar QUÉ aristas se intercambien. Con un swap suelto pasaba
    # exactamente eso (σ = 1.1e-16 en las 356 comparaciones del 31/08).
    # Jaccard teorico para k intercambios disjuntos: (|E|-k)/(|E|+k). Si el
    # observado coincide, el test es ARITMETICA y no mide la identidad de
    # las aristas, solo el tamano de la perturbacion. Lo que si mide la
    # identidad es el DeltaMSE del bloque [1b], que reejecuta el modelo.
    n_E = len(discretizar(np.mean(matrices, axis=0), p))
    if k_sw and n_E:
        j_teo = (n_E - k_sw) / (n_E + k_sw)
        print(f"      Jaccard teorico para {k_sw} swaps disjuntos de "
              f"{n_E} aristas: {j_teo:.4f}")
        if abs(np.mean(jacs) - j_teo) < 0.02:
            print(f"      El observado coincide con el teorico: este test "
                  f"acota el TAMANO de la perturbacion,")
            print(f"      no el efecto de QUE aristas sean. Para eso, el "
                  f"DeltaMSE del bloque [1b].")
    if np.std(jacs) < 1e-6:
        print(f"      AVISO: sigma ~ 0. El Jaccard no depende de qué "
              f"aristas se intercambien:")
        print(f"      este test no aporta evidencia sobre la identidad "
              f"causal de las aristas.")
    if np.mean(jacs) > 0.7:
        print(f"      → El grafo es INSENSIBLE a la identidad de las aristas")
        print(f"         (solo importan los pesos, no qué arista es cuál)")
    elif np.mean(jacs) > 0.4:
        print(f"      → Sensibilidad moderada: el grafo depende parcialmente")
        print(f"         de la identidad de las aristas (estructura causal real)")
    else:
        print(f"      → El grafo es ALTAMENTE SENSIBLE a la identidad de las aristas")
        print(f"         (cada arista tiene un rol causal distintivo)")

    plot_ablation_curves(ablation_results,
                          os.path.join(DIR_P2, "ablation_curves.png"))
    plot_auc_bar(ablation_results,
                  os.path.join(DIR_P2, "auc_ranking.png"))
    plot_alpha_critical(ablation_results,
                         os.path.join(DIR_P2, "alpha_critical_hist.png"))
    plot_edge_swap_heatmap(swap_results,
                            os.path.join(DIR_P2, "edge_swap_heatmap.png"))
    print(f"\n  ✅ Visualizaciones guardadas en {DIR_P2}/")

    results = {
        "discretizacion": "Top-P", "top_p": float(p), "eta": float(ETA_MARGEN),
        "M": M,
        "alpha_grid": ALPHA_GRID.tolist(),
        "n_candidates": len(ablation_results),
        # ΔMSE real (§8.2). "ablation" de abajo es el proxy de entropía y NO es
        # comparable con los umbrales de la Tabla 7; este sí lo es.
        "delta_mse": delta_results,
        "delta_mse_checkpoint": os.path.basename(ck_dmse) if ck_dmse else None,
        "ablation": [
            {"i": int(r.i), "j": int(r.j),
             "label_i": r.label_i, "label_j": r.label_j,
             "auc_degradation": float(r.auc_degradation),
             "alpha_critical": float(r.alpha_critical),
             "max_degradation": float(r.max_degradation),
             "jaccard_drop": float(r.jaccard_drop),
             "curve_alpha": r.curve_alpha,
             "curve_degradation": r.curve_degradation}
            for r in ablation_results
        ],
        "edge_swap": [
            {"rep": int(r["rep"]),
             "k_intercambiadas": int(r["k_intercambiadas"]),
             "e1_labels": r["e1_labels"], "e2_labels": r["e2_labels"],
             "jaccard_preserved": float(r["jaccard_preserved"]),
             "structure_preserved": bool(r["structure_preserved"])}
            for r in swap_results
        ],
        "edge_swap_resumen": {
            "n_sorteos": len(swap_results),
            "fraccion": SWAP_FRACCION,
            "jaccard_media": float(np.mean(
                [r["jaccard_preserved"] for r in swap_results]))
            if swap_results else None,
            "jaccard_std": float(np.std(
                [r["jaccard_preserved"] for r in swap_results]))
            if swap_results else None,
        },
        "summary": {
            "auc_mean": float(auc_values.mean()),
            "auc_median": float(np.median(auc_values)),
            "n_high_impact": int((auc_values > np.median(auc_values)).sum()),
            "n_structure_preserved": int(n_preserved),
            "n_swap_total": int(n_swap),
            "jaccard_mean_swap": float(np.mean(jacs)),
        }
    }
    with open(os.path.join(DIR_P2, "paradigm2_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  ✅ Resultados: {DIR_P2}/paradigm2_results.json")

    print(f"\n{'═'*82}")
    print(f" RESUMEN — PARADIGMA 2 (INTERVENCIONAL / CAUSAL)")
    print(f"{'═'*82}")
    print(f"  • Bajo ablación suave, {int((auc_values > np.median(auc_values)).sum())} "
          f"aristas superan la mediana interna (AUC > {np.median(auc_values):.4f}) "
          f"— esto es un ranking relativo, no un veredicto de aceptación.")
    n_pass_informe = int((auc_values > 0.005).sum())
    print(f"  • Umbral absoluto del informe (Tabla 7, etapa 4): AUC > 0,005 aprueba, "
          f"AUC ≈ 0 falla.")
    print(f"    → {n_pass_informe}/{len(auc_values)} aristas superan ese umbral "
          f"({'ninguna: el impacto causal individual es marginal con este mecanismo '
             'de ablación' if n_pass_informe == 0 else 'ver detalle por arista arriba'}).")
    print(f"  • Perfiles funcionales detectados:")
    print(f"      - 'Todo o nada' (α*≈0): "
          f"{sum(1 for r in ablation_results if r.alpha_critical < 0.1)} aristas")
    print(f"      - 'Umbral' (α*∈[0.3,0.7]): "
          f"{sum(1 for r in ablation_results if 0.3 <= r.alpha_critical <= 0.7)} aristas")
    print(f"  • Edge swapping: {100*n_preserved/max(n_swap,1):.1f}% de pares preservan")
    print(f"    la estructura topológica → "
          f"{'alta' if n_preserved/n_swap < 0.4 else 'moderada' if n_preserved/n_swap < 0.7 else 'baja'} "
          f"dependencia de la identidad causal (mayor preservación tras el swap = "
          f"el grafo depende menos de qué arista específica sea cuál).")
    print(f"  • Continúa con Paradigma 3 (Geométrico/Topológico) para triangular.")
    print(f"{'═'*82}\n")

    return ablation_results, swap_results

if __name__ == "__main__" and EJECUTAR_PARTE5:
    main()

# ============================================================================
# Paradigma 3
# ============================================================================

def matrix_to_digraph(A: np.ndarray, p: Optional[float] = None,
                      labels: List[str] = None) -> nx.DiGraph:
    """A → grafo dirigido usando la MISMA discretización Top-P que el resto.

    Antes construía el grafo con Top-K (k=5 por fila), distinto del Top-P que
    usaba el Paradigma 1, de modo que los invariantes topológicos se medían
    sobre un grafo que no era el que se declaraba significativo.
    """
    if labels is None:
        labels = INDEX_NAMES
    n = A.shape[0]
    G = nx.DiGraph()
    for i in range(n):
        G.add_node(i, label=labels[i])
    for (i, j) in discretizar(A, p):
        G.add_edge(i, j, weight=float(A[i, j]))
    return G

def laplacian_spectrum(G: nx.DiGraph, n: int = NUM_INDICES) -> np.ndarray:
    A = np.zeros((n, n))
    for u, v, d in G.edges(data=True):
        w = d.get("weight", 1.0)
        A[u, v] = w
    A_sym = (A + A.T) / 2.0
    D = np.diag(A_sym.sum(axis=1))
    L = D - A_sym
    eigvals = np.linalg.eigvalsh(L)
    return np.sort(eigvals)

def classify_node_roles(G: nx.DiGraph, n: int = NUM_INDICES) -> Dict[int, str]:
    in_deg = dict(G.in_degree(weight="weight"))
    out_deg = dict(G.out_degree(weight="weight"))
    try:
        betw = nx.betweenness_centrality(G, weight="weight")
    except Exception:
        betw = {i: 0.0 for i in range(n)}
    in_t = float(np.mean(list(in_deg.values()))) if in_deg else 0.0
    out_t = float(np.mean(list(out_deg.values()))) if out_deg else 0.0
    betw_t = float(np.mean(list(betw.values()))) if betw else 0.0
    roles = {}
    for i in range(n):
        ih = in_deg.get(i, 0) > in_t
        oh = out_deg.get(i, 0) > out_t
        ib = betw.get(i, 0) > betw_t
        if ih and oh:
            roles[i] = "hub_source"
        elif ih:
            roles[i] = "hub"
        elif oh:
            roles[i] = "source"
        elif ib:
            roles[i] = "bridge"
        else:
            roles[i] = "isolated"
    return roles

def graph_edit_distance_approx(G1: nx.DiGraph, G2: nx.DiGraph,
                                n: int = NUM_INDICES) -> int:
    E1 = set(G1.edges())
    E2 = set(G2.edges())
    return len(E1.symmetric_difference(E2))

def graph_edit_distance_normalized(G1: nx.DiGraph, G2: nx.DiGraph) -> float:
    E1 = set(G1.edges())
    E2 = set(G2.edges())
    sym_diff = len(E1.symmetric_difference(E2))
    union = len(E1 | E2)
    return sym_diff / union if union > 0 else 0.0

@dataclass
class PairwiseAnalysis:
    seed_a: int; seed_b: int
    ged: int
    ged_normalized: float
    spectral_distance: float
    jaccard_edges: float
    role_overlap: float

def analyze_pair(A_a: np.ndarray, A_b: np.ndarray,
                  seed_a: int, seed_b: int,
                  p: Optional[float] = None) -> PairwiseAnalysis:
    G1 = matrix_to_digraph(A_a, p=p)
    G2 = matrix_to_digraph(A_b, p=p)
    ged = graph_edit_distance_approx(G1, G2)
    ged_norm = graph_edit_distance_normalized(G1, G2)
    spec1 = laplacian_spectrum(G1)
    spec2 = laplacian_spectrum(G2)
    spec_dist = float(np.linalg.norm(spec1 - spec2))
    E1, E2 = set(G1.edges()), set(G2.edges())
    jaccard_e = (len(E1 & E2) / len(E1 | E2)) if (E1 | E2) else 1.0
    r1 = classify_node_roles(G1)
    r2 = classify_node_roles(G2)
    same = sum(1 for i in range(NUM_INDICES) if r1.get(i) == r2.get(i))
    role_overlap = same / NUM_INDICES

    return PairwiseAnalysis(
        seed_a=seed_a, seed_b=seed_b,
        ged=ged, ged_normalized=ged_norm,
        spectral_distance=spec_dist,
        jaccard_edges=jaccard_e,
        role_overlap=role_overlap,
    )

def topological_invariants(G: nx.DiGraph, n: int = NUM_INDICES) -> Dict:
    scc = nx.number_strongly_connected_components(G)
    wcc = nx.number_weakly_connected_components(G)
    density = nx.density(G)
    try:
        recip = float(nx.reciprocity(G))
    except Exception:
        recip = 0.0
    try:
        und = G.to_undirected()
        communities = list(nx.community.louvain_communities(und, weight="weight", seed=42))
        mod = float(nx.community.modularity(und, communities, weight="weight"))
    except Exception:
        communities, mod = [], 0.0
    degrees = [d for _, d in G.degree()]
    deg_mean = float(np.mean(degrees)) if degrees else 0.0
    deg_std = float(np.std(degrees)) if degrees else 0.0
    n_hubs = int(sum(1 for d in degrees if d > deg_mean + deg_std))

    return {
        "scc": int(scc), "wcc": int(wcc),
        "modularity": mod, "density": float(density),
        "reciprocity": recip,
        "n_hubs": n_hubs,
        "n_communities": len(communities),
        "mean_degree": deg_mean, "std_degree": deg_std,
    }

def plot_ged_matrix(pairwise: List[PairwiseAnalysis], seeds: List[int],
                    output_path: str):
    M = len(seeds)
    mat_ged = np.zeros((M, M))
    mat_spec = np.zeros((M, M))
    idx = {(s, s): i for i, s in enumerate(seeds)}
    for s in seeds:
        idx[(s, s)] = seeds.index(s)
    for p in pairwise:
        i = seeds.index(p.seed_a)
        j = seeds.index(p.seed_b)
        mat_ged[i, j] = p.ged_normalized
        mat_ged[j, i] = p.ged_normalized
        mat_spec[i, j] = p.spectral_distance
        mat_spec[j, i] = p.spectral_distance

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    im1 = axes[0].imshow(mat_ged, cmap="YlOrRd", aspect="equal", vmin=0, vmax=1)
    axes[0].set_title("Graph Edit Distance Normalizada\n(0 = idénticos, 1 = totalmente distintos)",
                       fontsize=12, fontweight="bold")
    axes[0].set_xticks(range(M)); axes[0].set_yticks(range(M))
    axes[0].set_xticklabels(seeds, fontsize=7, rotation=45)
    axes[0].set_yticklabels(seeds, fontsize=7)
    plt.colorbar(im1, ax=axes[0], label="GED_norm", fraction=0.046)
    vmax = max(mat_spec.max(), 1e-6)
    im2 = axes[1].imshow(mat_spec, cmap="viridis", aspect="equal", vmin=0, vmax=vmax)
    axes[1].set_title("Distancia Espectral del Laplaciano\n‖λ(L⁽ᵃ⁾) − λ(L⁽ᵇ⁾)‖₂",
                       fontsize=12, fontweight="bold")
    axes[1].set_xticks(range(M)); axes[1].set_yticks(range(M))
    axes[1].set_xticklabels(seeds, fontsize=7, rotation=45)
    axes[1].set_yticklabels(seeds, fontsize=7)
    plt.colorbar(im2, ax=axes[1], label="‖Δλ‖₂", fraction=0.046)
    plt.suptitle(f"Distancias Geométricas/Topológicas entre pares — M={M} modelos",
                  fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight"); plt.close()

def plot_role_stability(role_history: List[Dict[int, str]],
                         output_path: str, labels: List[str]):
    M = len(role_history)
    role_names = ["hub_source", "hub", "source", "bridge", "isolated"]
    counts = np.zeros((NUM_INDICES, len(role_names)))
    for rh in role_history:
        for node, role in rh.items():
            if role in role_names:
                counts[node, role_names.index(role)] += 1
    freq = counts / M

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(freq, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(role_names))); ax.set_yticks(range(NUM_INDICES))
    ax.set_xticklabels(role_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Rol estructural", fontsize=11)
    ax.set_ylabel("Nodo (índice espectral)", fontsize=11)
    ax.set_title(f"Estabilidad de roles — probabilidad por nodo (M={M} modelos)\n"
                "Verde fuerte = rol estable (>0.7) | Azul claro = rol inestable",
                fontsize=12, fontweight="bold")
    for i in range(NUM_INDICES):
        for j in range(len(role_names)):
            v = freq[i, j]
            color = "white" if v > 0.5 else "black"
            if v > 0:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold")
    plt.colorbar(im, ax=ax, label="P(rol | nodo)", fraction=0.046)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight"); plt.close()

def plot_topological_invariants(invariants: List[Dict], seeds: List[int],
                                  output_path: str):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics = [
        ("scc", "# componentes SCC", "#3498db"),
        ("modularity", "Modularidad (Louvain)", "#e74c3c"),
        ("density", "Densidad", "#2ecc71"),
        ("reciprocity", "Reciprocidad", "#9b59b6"),
        ("n_hubs", "# Hubs", "#f39c12"),
        ("n_communities", "# Comunidades", "#1abc9c"),
    ]
    for ax, (key, title, color) in zip(axes.flatten(), metrics):
        vals = [inv[key] for inv in invariants]
        ax.hist(vals, bins=min(10, len(set(vals))), color=color,
                edgecolor="white", alpha=0.8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Valor"); ax.set_ylabel("# modelos")
        ax.grid(True, alpha=0.3)
        if vals:
            m = float(np.mean(vals))
            ax.axvline(x=m, color="black", linestyle="--", alpha=0.6,
                       label=f"μ={m:.3f}")
            ax.legend(fontsize=8)
    plt.suptitle(f"Invariantes Topológicos a través de {len(seeds)} modelos\n"
                "Distribución estrecha → topología estable",
                fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight"); plt.close()

def plot_ged_vs_jaccard(pairwise: List[PairwiseAnalysis], output_path: str):
    fig, ax = plt.subplots(figsize=(10, 7))
    geds = [p.ged_normalized for p in pairwise]
    jacs = [p.jaccard_edges for p in pairwise]
    specs = [p.spectral_distance for p in pairwise]
    sc = ax.scatter(geds, jacs, c=specs, cmap="viridis", s=60, alpha=0.75,
                    edgecolors="white", linewidths=0.5)
    ax.set_xlabel("GED normalizada (topología)", fontsize=11)
    ax.set_ylabel("Jaccard de aristas (coincidencia exacta)", fontsize=11)
    ax.set_title("GED vs Jaccard: ¿la topología cuenta la misma historia?\n"
                "Puntos sobre la diagonal = estructuras equivalentes\n"
                "Color = distancia espectral del Laplaciano",
                fontsize=12, fontweight="bold")
    cbar = plt.colorbar(sc, ax=ax, label="‖Δλ_L‖₂", fraction=0.046)
    ax.grid(True, alpha=0.3)
    xs = np.linspace(0, 1, 50)
    ax.plot(xs, 1 - xs, "r--", alpha=0.5, label="GED = 1 − Jaccard (teórica)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight"); plt.close()

def calibrar_estabilidad_rol_nulo(role_history: List[Dict[int, str]],
                                  n_indices: int = NUM_INDICES,
                                  B: int = 2000,
                                  seed: int = SEMILLA_NULO) -> Dict:
    """Calibra P(rol) por nodo contra un nulo, en vez de compararlo con un
    umbral fijo.

    El umbral anterior, `ROLE_STABLE_THRESH = max(0.50, 2/n_roles)`, se eligió
    DESPUÉS de ver que 0.7 fijo dejaba a los 12 nodos fuera (ver comentario en
    build_diagnosis_matrix_5d) — umbral ajustado al dato, no al revés. Esta
    función reemplaza esa comparación por el mismo patrón que ya usa
    `te_global_pass` (calibracion_nulo.json): un p-valor empírico contra un
    nulo bien definido, decidido antes de mirar si el nodo pasa o no.

    El nulo permuta, DENTRO de cada modelo, qué nodo recibe cada rol (rng
    distinto por sorteo, misma semilla base que el resto del framework). Esto
    conserva la distribución de roles de esa semilla (cuántos hubs, sources,
    etc. salieron) y rompe únicamente la identidad del nodo, que es
    exactamente la hipótesis nula: "el nodo i tiene su rol dominante por
    azar, no porque ese índice cumpla consistentemente esa función".

    Devuelve, por nodo: probabilidad observada, p-valor empírico (fracción del
    nulo que iguala o supera lo observado) y el percentil 95 del nulo como
    referencia. Un nodo se considera estable si p_valor < 0.05.
    """
    rng = np.random.default_rng(seed)
    M = len(role_history)
    if M == 0:
        return {"obs_prob": {}, "p_valor": {}, "umbral95_nulo": {}, "B": B}

    obs_prob = {}
    for i in range(n_indices):
        c = Counter([rh.get(i, "isolated") for rh in role_history])
        obs_prob[i] = c.most_common(1)[0][1] / M

    null_max_prob = np.zeros((B, n_indices))
    for b in range(B):
        conteos = [Counter() for _ in range(n_indices)]
        for rh in role_history:
            roles_modelo = [rh.get(i, "isolated") for i in range(n_indices)]
            permutado = rng.permutation(roles_modelo)
            for i, r in enumerate(permutado):
                conteos[i][r] += 1
        for i in range(n_indices):
            null_max_prob[b, i] = conteos[i].most_common(1)[0][1] / M

    p_valor, umbral95 = {}, {}
    for i in range(n_indices):
        p_valor[i] = float((null_max_prob[:, i] >= obs_prob[i]).mean())
        umbral95[i] = float(np.percentile(null_max_prob[:, i], 95))

    return {"obs_prob": obs_prob, "p_valor": p_valor,
            "umbral95_nulo": umbral95, "B": B}


# ----------------------------------------------------------------------------
# Instrumento topológico POR ARISTA — reemplazo del hueco que dejó P3 en la
# clasificación por arista (LINEAMIENTO_NUEVOS_PARADIGMAS.md §1).
#
# AVISO DE FIDELIDAD, para no mal-citar el paper: esto NO es el algoritmo de
# KHAN (Liang, Liu, Zhou, Zhang, Lu — 2024, "The Wreaths of KHAN"), que usa un
# Gram-Schmidt discreto sobre p-valores combinados de generadores de
# homología y no se reprodujo acá por no tener acceso al paper completo
# (solo al resumen de la búsqueda de literatura). Es una construcción propia,
# más simple, pero igual de auditable: un test de permutación sobre
# PARTICIPACIÓN EN CICLOS de la filtración por peso de arista (misma familia
# de idea — persistencia topológica con FDR por arista — pero un algoritmo
# distinto). Si se cita KHAN en la tesis, aclarar que el código implementa
# una variante propia inspirada en el paper, no el paper mismo.
# ----------------------------------------------------------------------------

def _filtracion_union_find(edges_ordenadas: List[Tuple[int, int]],
                           pesos_ordenados: np.ndarray,
                           n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Filtracion descendente por peso con union-find.

    Devuelve, POR POSICION k de la filtracion:
      crea_ciclo[k]  True si esa arista une dos nodos ya conectados.
      w_conexion[k]  peso al que i y j quedaron conectados por primera vez.
                     Para una arista de arbol es su propio peso (ella es la
                     conexion). Para una que cierra ciclo, es el peso de la
                     arista que cerro la union de sus componentes, es decir el
                     cuello de botella del mejor camino indirecto: todas las
                     aristas procesadas antes tienen peso >= ese valor.
    """
    padre = list(range(n))
    # w_merge[r] no sirve: hace falta el peso al que se unieron ESTOS dos nodos,
    # asi que se guarda, por componente, el peso de la ultima fusion que la
    # formo, y se propaga por el arbol de union-find.
    w_union = np.full((n, n), -np.inf)      # w_union[a,b] = peso de conexion
    miembros = {k: [k] for k in range(n)}

    def find(a):
        while padre[a] != a:
            padre[a] = padre[padre[a]]
            a = padre[a]
        return a

    m = len(edges_ordenadas)
    crea_ciclo = np.zeros(m, dtype=bool)
    w_conexion = np.zeros(m, dtype=np.float64)
    for k, (i, j) in enumerate(edges_ordenadas):
        w = float(pesos_ordenados[k])
        ri, rj = find(i), find(j)
        if ri == rj:
            crea_ciclo[k] = True
            w_conexion[k] = w_union[i, j]
        else:
            w_conexion[k] = w             # la arista ES la conexion
            # al fusionar, todo par cruzado queda conectado a este peso
            for a in miembros[ri]:
                for b in miembros[rj]:
                    w_union[a, b] = w
                    w_union[b, a] = w
            if len(miembros[ri]) < len(miembros[rj]):
                ri, rj = rj, ri
            padre[rj] = ri
            miembros[ri].extend(miembros[rj])
            del miembros[rj]
    return crea_ciclo, w_conexion


def _brecha_por_par(pesos: np.ndarray, pares: List[Tuple[int, int]],
                    n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Brecha de cuello de botella por PAR (no por posicion).

    brecha[p] = w_conexion(p) - w(p) >= 0.  Cero si el par es arista de arbol
    en la filtracion; positiva si existe un camino indirecto cuyo cuello de
    botella es mas fuerte que la arista directa.
    """
    orden = np.argsort(-pesos, kind="stable")
    edges = [pares[k] for k in orden]
    crea, w_con = _filtracion_union_find(edges, pesos[orden], n)
    brecha_pos = w_con - pesos[orden]
    brecha = np.empty(len(pares)); brecha[orden] = brecha_pos
    ciclo = np.zeros(len(pares), dtype=bool); ciclo[orden] = crea
    return brecha, ciclo


def topologia_persistente_edge_pvalues(A_mean: np.ndarray, labels: List[str],
                                       n_permutaciones: int = 2000,
                                       alpha_fdr: float = FDR_TARGET,
                                       seed: int = SEMILLA_NULO,
                                       matrices: Optional[List[np.ndarray]] = None,
                                       nulo: str = "columnas") -> Dict:
    """Instrumento topologico por arista: brecha de cuello de botella.

    Para cada par no dirigido {i,j} sobre A_mean simetrizada, se recorre la
    filtracion en orden descendente de peso y se mide:

        brecha(i,j) = w_conexion(i,j) - w(i,j)

    donde w_conexion es el peso al que i y j quedaron conectados por primera
    vez usando SOLO aristas de peso mayor o igual. Brecha 0 significa que la
    arista directa es la unica via a ese nivel (arista del arbol de maximo peso,
    estructuralmente esencial). Brecha positiva significa que existe un camino
    indirecto con cuello de botella mas fuerte: la relacion directa es
    transitivamente redundante. Es el mismo cuello de botella que define la
    persistencia de enlace simple, y a diferencia del indicador binario
    "cierra ciclo" tiene resolucion continua.

    QUE SE ARREGLO, Y POR QUE IMPORTA. La version anterior de esta funcion:
      (1) acumulaba el conteo nulo por POSICION de la filtracion en vez de por
          PAR, asi que el p-valor era P(la posicion k cierra un ciclo) y lo
          observado nunca entraba en la comparacion. Medido sobre A_mean:
          rho(p_valor, peso) = -0.923, te_sig = exactamente los 3 pares de
          mayor peso, y uno de esos 3 tenia crea_ciclo=False.
      (2) usaba un estadistico binario que sobre un grafo COMPLETO no tiene
          informacion: en cualquier orden de las 66 aristas de K12 hay
          exactamente 11 de arbol y 55 que cierran ciclo.

    DOS NULOS, y por que el segundo es el que sirve.

    nulo="permutacion" — permuta que par recibe cada peso. Es INTERCAMBIABLE
        entre pares: la distribucion nula marginal es identica para los 66, asi
        que el p-valor sale como una transformacion monotona del propio
        estadistico y ningun par puede pasar por separado. Medido: te_sig = 0 de
        66, porque un par con brecha 0 tiene p ~ 11/66 = 0.167 por
        construccion (en cualquier orden de las 66 aristas de K12 hay
        exactamente 11 de arbol, y el nulo las reparte al azar). Solo sirve
        como referencia de "donde corta el azar", no como test.

    nulo="columnas" (por defecto) — usa generar_nulo_atencion(matrices,
        "solo_columnas"), el mismo nulo que ya calibra Te global y la FASE 1:
        conserva el perfil de columna (los sumideros) y borra la estructura
        especifica de fila. NO es intercambiable entre pares, porque un par que
        toca una columna sumidero tiene una distribucion nula distinta de uno
        que no. Con esto el p-valor si responde la pregunta correcta: la brecha
        de este par es mas chica de lo que da un ensemble que reproduce los
        sumideros pero no la relacion indice-a-indice? Requiere `matrices`.

    Que este instrumento aporte o no se decide igual en matriz_independencia(),
    comparando la BRECHA contra Phi y contra el dMSE.

    Devuelve {(i,j): {...}} con entrada espejada en (j,i): la filtracion no
    distingue direccion, P1 y P2 ya cubren la asimetria.
    """
    n = A_mean.shape[0]
    W = (np.abs(A_mean) + np.abs(A_mean.T)) / 2.0
    pares = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pesos = np.array([W[i, j] for i, j in pares], dtype=np.float64)

    brecha_obs, ciclo_obs = _brecha_por_par(pesos, pares, n)

    # Nulo, acumulado POR PAR (el bug anterior lo acumulaba por posicion).
    rng = np.random.default_rng(seed)
    usar_columnas = (nulo == "columnas" and matrices is not None
                     and len(matrices) >= 2)
    if nulo == "columnas" and not usar_columnas:
        print(f"      AVISO: nulo='columnas' necesita `matrices`; se cae al de "
              f"permutacion, que es intercambiable y no discrimina por arista.")
    B = int(n_permutaciones if not usar_columnas else min(n_permutaciones, 500))
    n_le = np.zeros(len(pares), dtype=np.int64)     # nulo <= observado
    brechas_nulo = np.empty((B, len(pares)))
    for b in range(B):
        if usar_columnas:
            # Ensemble nulo que conserva el perfil de columna y borra la
            # estructura de fila; se promedia igual que A_mean y se le mide la
            # misma brecha, asi que observado y nulo pasan por el mismo codigo.
            nul = generar_nulo_atencion(matrices, "solo_columnas", rng)
            An = np.mean([np.asarray(x, dtype=np.float64) for x in nul], axis=0)
            Wn = (np.abs(An) + np.abs(An.T)) / 2.0
            pesos_b = np.array([Wn[i, j] for i, j in pares], dtype=np.float64)
        else:
            pesos_b = rng.permutation(pesos)
        brecha_perm, _ = _brecha_por_par(pesos_b, pares, n)
        brechas_nulo[b] = brecha_perm
        n_le += (brecha_perm <= brecha_obs)
    n_permutaciones = B

    # Cola INFERIOR: brecha pequena = arista esencial. +1/+1 de Laplace, igual
    # que en calibrar_metricas_estabilidad, para que el p nunca sea 0.
    p_por_par = (n_le + 1) / (n_permutaciones + 1)
    rechazadas, qvals, _ = benjamini_hochberg_bh(p_por_par, alpha_fdr)

    nulo_media = brechas_nulo.mean(axis=0)
    nulo_p05 = np.percentile(brechas_nulo, 5, axis=0)

    resultados = {}
    for idx, (i, j) in enumerate(pares):
        entry = {"crea_ciclo": bool(ciclo_obs[idx]),
                 "nulo": ("solo_columnas" if usar_columnas else "permutacion"),
                 "brecha": float(brecha_obs[idx]),
                 "peso": float(pesos[idx]),
                 "brecha_nulo_media": float(nulo_media[idx]),
                 "brecha_nulo_p05": float(nulo_p05[idx]),
                 "p_valor": float(p_por_par[idx]),
                 "q_valor": float(qvals[idx]),
                 "te_sig": bool(rechazadas[idx])}
        resultados[(i, j)] = entry
        resultados[(j, i)] = dict(entry)   # copia: mutar una no muta la otra
    return resultados


def main(p_top: Optional[float] = None):
    print(f"\n{'═'*82}")
    print(f" PARADIGMA 3 — GEOMÉTRICO / TOPOLÓGICO")
    print(f" Pregunta: ¿La forma (del grafo) se mantiene?")
    print(f" Fundamento: GED, espectro del Laplaciano, estabilidad de roles")
    print(f"{'═'*82}")

    matrices = load_matrices()
    M = len(matrices)
    seeds = list(range(M))
    print(f"  Modelos disponibles: M = {M}")
    calcular_eta(matrices)
    p_top = p_efectivo(p_top)
    aristas_fila = tamano_medio_discretizacion(matrices, p_top)

    print(f"\n  [1] CONSTRUYENDO GRAFOS Top-P (P={p_top}, η={ETA_MARGEN:.6f}) "
          f"para cada semilla...")
    print(f"      (media de {aristas_fila:.2f} aristas por fila de "
          f"{NUM_INDICES-1} posibles → ~{aristas_fila*NUM_INDICES:.0f} aristas)")
    graphs = [matrix_to_digraph(A, p=p_top) for A in matrices]
    invariants = [topological_invariants(G) for G in graphs]

    print(f"\n  INVARIANTES TOPOLÓGICOS (resumen sobre M={M} modelos):")
    print(f"    {'Métrica':18s} {'μ':>10s} {'σ':>10s} {'min':>8s} {'max':>8s}")
    print(f"    {'─'*54}")
    for key in ["scc", "wcc", "modularity", "density", "reciprocity",
                "n_hubs", "n_communities", "mean_degree"]:
        vals = np.array([inv[key] for inv in invariants])
        print(f"    {key:18s} {vals.mean():>10.4f} {vals.std():>10.4f} "
              f"{vals.min():>8.4f} {vals.max():>8.4f}")

    print(f"\n  [2] ANÁLISIS POR PARES ({M*(M-1)//2} pares)...")
    pairwise: List[PairwiseAnalysis] = []
    for a in range(M):
        for b in range(a + 1, M):
            pa = analyze_pair(matrices[a], matrices[b], a, b, p=p_top)
            pairwise.append(pa)

    geds = np.array([p.ged_normalized for p in pairwise])
    specs = np.array([p.spectral_distance for p in pairwise])
    jacs = np.array([p.jaccard_edges for p in pairwise])
    roles = np.array([p.role_overlap for p in pairwise])

    # ρrole (informe §19.3, Tabla 7 etapa 3): correlación de Spearman entre
    # vectores de centralidad (PageRank) de cada par de semillas. NO es lo
    # mismo que "role_overlap" arriba (que solo cuenta coincidencia de
    # etiqueta discreta hub/source/bridge/isolated, una proxy más burda).
    # ρrole es el estadístico que el informe realmente exige en la etapa 3.
    pagerank_vecs = []
    for G in graphs:
        try:
            pr = nx.pagerank(G, weight="weight")
        except Exception:
            pr = {}
        pagerank_vecs.append(np.array([pr.get(i, 0.0) for i in range(NUM_INDICES)]))
    rho_roles = []
    for a in range(M):
        for b in range(a + 1, M):
            rho = stats.spearmanr(pagerank_vecs[a], pagerank_vecs[b]).statistic
            rho_roles.append(0.0 if np.isnan(rho) else float(rho))
    rho_roles = np.array(rho_roles)
    rho_role_mean = float(rho_roles.mean()) if len(rho_roles) else 0.0

    print(f"\n  RESUMEN ESTADÍSTICO (sobre {len(pairwise)} pares):")
    print(f"    GED normalizada:        μ={geds.mean():.3f} ± {geds.std():.3f}  "
          f"[min={geds.min():.3f}, max={geds.max():.3f}]")
    print(f"    Distancia espectral:    μ={specs.mean():.4f} ± {specs.std():.4f}  "
          f"[min={specs.min():.4f}, max={specs.max():.4f}]")
    print(f"    Jaccard de aristas:     μ={jacs.mean():.3f} ± {jacs.std():.3f}  "
          f"[min={jacs.min():.3f}, max={jacs.max():.3f}]")
    print(f"    Overlap de roles (categórico):  μ={roles.mean():.3f} ± {roles.std():.3f}  "
          f"[min={roles.min():.3f}, max={roles.max():.3f}]")
    print(f"    ρrole (Spearman, PageRank):     μ={rho_role_mean:.3f} ± {rho_roles.std():.3f}  "
          f"[min={rho_roles.min():.3f}, max={rho_roles.max():.3f}]")
    veredicto_rho = ("APRUEBA" if rho_role_mean > 0.9
                      else "FALLA" if rho_role_mean < 0.7 else "DUDOSO")
    print(f"    → Etapa 3 (Tabla 7, informe): ρrole {veredicto_rho} "
          f"(umbral: >0.9 aprueba, <0.7 falla)")

    print(f"\n  INTERPRETACIÓN:")
    if geds.mean() < 0.3:
        print(f"    ✅ GED baja ({geds.mean():.3f}): la topología macroscópica "
              f"se preserva entre semillas.")
    elif geds.mean() < 0.6:
        print(f"    🟡 GED moderada ({geds.mean():.3f}): la topología es parcialmente "
              f"estable; algunas aristas periféricas fluctúan.")
    else:
        print(f"    ❌ GED alta ({geds.mean():.3f}): las estructuras topológicas "
              f"difieren significativamente entre semillas.")

    if roles.mean() > 0.7:
        print(f"    ✅ Roles estables (overlap={roles.mean():.3f}): los nodos "
              f"conservan su función (hub/source/bridge) entre semillas.")
    else:
        print(f"    ⚠️  Roles inestables (overlap={roles.mean():.3f}): los nodos "
              f"cambian de función entre semillas.")

    print(f"\n  [3] ESTABILIDAD DE ROLES POR NODO...")
    role_history = [classify_node_roles(G) for G in graphs]

    # Calibración contra nulo en vez de umbral fijo (ver
    # calibrar_estabilidad_rol_nulo): decide "estable" con p-valor empírico,
    # no con un P(rol) > constante elegida al ver los datos.
    cal_rol = calibrar_estabilidad_rol_nulo(role_history, NUM_INDICES)

    print(f"\n  ROL MÁS FRECUENTE POR NODO (estable = p-valor < 0.05 vs nulo "
          f"de permutación):")
    role_names = ["hub_source", "hub", "source", "bridge", "isolated"]
    print(f"    {'Nodo':12s} {'Rol dominante':15s} {'P(rol)':>8s} "
          f"{'p-valor':>9s} {'p95 nulo':>10s} {'Estable':>10s}")
    print(f"    {'─'*68}")
    node_role_summary = {}
    for i in range(NUM_INDICES):
        role_count = Counter([rh.get(i, "isolated") for rh in role_history])
        most_role, most_count = role_count.most_common(1)[0]
        p = most_count / M
        p_val = cal_rol["p_valor"].get(i, 1.0)
        p95_nulo = cal_rol["umbral95_nulo"].get(i, 1.0)
        estable = p_val < 0.05
        stable_flag = "✓" if estable else "—"
        print(f"    {INDEX_NAMES[i]:12s} {most_role:15s} {p:>8.2f} "
              f"{p_val:>9.3f} {p95_nulo:>10.2f} {stable_flag:>10s}")
        node_role_summary[INDEX_NAMES[i]] = {
            "role": most_role, "prob": float(p),
            "p_valor_nulo": float(p_val), "umbral95_nulo": float(p95_nulo),
            "estable_calibrado": bool(estable),
        }

    print(f"\n  [4] INSTRUMENTO TOPOLÓGICO POR ARISTA (inspirado en KHAN, "
          f"Liang et al. 2024 — ver aviso de fidelidad en el código, no es "
          f"su algoritmo exacto)...")
    A_mean = np.mean([np.asarray(A, dtype=np.float64) for A in matrices], axis=0)
    topo_edges = topologia_persistente_edge_pvalues(
        A_mean, INDEX_NAMES, matrices=matrices, nulo="columnas")
    n_topo_sig = sum(1 for (i, j), v in topo_edges.items() if i < j and v["te_sig"])
    print(f"      {n_topo_sig}/{len(topo_edges)//2} pares con p_valor FDR-significativo "
          f"(crea/evita ciclo en la filtración más de lo esperado por azar)")

    plot_ged_matrix(pairwise, seeds,
                     os.path.join(DIR_P3, "ged_spectral_matrices.png"))
    plot_role_stability(role_history,
                         os.path.join(DIR_P3, "role_stability.png"),
                         INDEX_NAMES)
    plot_topological_invariants(invariants, seeds,
                                 os.path.join(DIR_P3, "topological_invariants.png"))
    plot_ged_vs_jaccard(pairwise,
                         os.path.join(DIR_P3, "ged_vs_jaccard.png"))
    print(f"\n  ✅ Visualizaciones guardadas en {DIR_P3}/")

    results = {
        "discretizacion": "Top-P", "top_p": float(p_top),
        "eta": float(ETA_MARGEN), "aristas_por_fila": float(aristas_fila),
        "M": M,
        "topologia_persistente": {
            "aviso_fidelidad": ("Inspirado en KHAN (Liang et al. 2024), NO es "
                                "su algoritmo Gram-Schmidt exacto — ver "
                                "docstring de topologia_persistente_edge_pvalues."),
            "estadistico": ("brecha de cuello de botella: peso al que i y j "
                            "quedaron conectados por primera vez en la "
                            "filtracion descendente, menos el peso directo; "
                            "0 = arista esencial del arbol de maximo peso"),
            "nulo": "solo_columnas (conserva sumideros, borra la fila)",
            "alpha_fdr": float(FDR_TARGET),
            "n_pares_significativos": int(n_topo_sig),
            "edges": [
                {"i": int(i), "j": int(j),
                 "crea_ciclo": v["crea_ciclo"], "p_valor": v["p_valor"],
                 "q_valor": v["q_valor"], "te_sig": v["te_sig"],
                 # la brecha es el ESTADISTICO; es lo que entra a
                 # matriz_independencia(), no el p-valor
                 "brecha": v["brecha"], "peso": v["peso"],
                 "brecha_nulo_media": v["brecha_nulo_media"],
                 "nulo_usado": v["nulo"]}
                for (i, j), v in topo_edges.items() if i < j
            ],
        },
        "pairwise": [
            {"seed_a": int(p.seed_a), "seed_b": int(p.seed_b),
             "ged": int(p.ged), "ged_normalized": float(p.ged_normalized),
             "spectral_distance": float(p.spectral_distance),
             "jaccard_edges": float(p.jaccard_edges),
             "role_overlap": float(p.role_overlap)}
            for p in pairwise
        ],
        "invariants": invariants,
        "node_role_summary": node_role_summary,
        "summary": {
            "ged_mean": float(geds.mean()), "ged_std": float(geds.std()),
            "spectral_mean": float(specs.mean()), "spectral_std": float(specs.std()),
            "jaccard_mean": float(jacs.mean()), "jaccard_std": float(jacs.std()),
            "role_overlap_mean": float(roles.mean()),
            "role_overlap_std": float(roles.std()),
            "role_stability_spearman_mean": rho_role_mean,
            "role_stability_spearman_std": float(rho_roles.std()) if len(rho_roles) else 0.0,
            "role_stability_spearman_veredicto": veredicto_rho,
            # GED normalizada y Jaccard de aristas son LA MISMA medida:
            # con la misma densidad, GED = 1 - Jaccard exacto (comprobado
            # el 31/08: suma 1.000000 y desviaciones identicas). No son dos
            # evidencias, es una. Y el mismo numero sale otra vez en el
            # J(attn,attn) del paradigma 5.
            "ged_es_1_menos_jaccard": bool(
                abs(float(geds.mean()) + float(jacs.mean()) - 1.0) < 1e-6),
            "nota_redundancia": ("GED = 1 - Jaccard; reportar las dos como "
                                 "evidencias independientes es contarlas "
                                 "dos veces. Lo unico no redundante de "
                                 "este paradigma es spectral_mean."),
            "n_nodos_estables_calibrado": int(sum(
                1 for v in node_role_summary.values()
                if v.get("estable_calibrado"))),
            "calibracion_rol_B": cal_rol["B"],
            "calibracion_rol_metodo": ("permutacion intra-modelo de roles "
                                       "entre nodos, p<0.05 vs nulo"),
        }
    }
    if abs(float(geds.mean()) + float(jacs.mean()) - 1.0) < 1e-6:
        print("")
        print(f"  AVISO: GED ({geds.mean():.4f}) + Jaccard "
              f"({jacs.mean():.4f}) = 1.000000 exacto.")
        print(f"  Son la misma medida. En la tesis va UNA, no dos, y el "
              f"J(attn,attn) del P5 tambien es ese mismo numero.")
    with open(os.path.join(DIR_P3, "paradigm3_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  ✅ Resultados: {DIR_P3}/paradigm3_results.json")

    print(f"\n{'═'*82}")
    print(f" RESUMEN — PARADIGMA 3 (GEOMÉTRICO / TOPOLÓGICO)")
    print(f"{'═'*82}")
    print(f"  • GED normalizada media: {geds.mean():.3f} ± {geds.std():.3f}")
    print(f"    → {'topología preservada' if geds.mean() < 0.3 else 'topología parcialmente preservada' if geds.mean() < 0.6 else 'topología inestable'}")
    print(f"  • Distancia espectral media: {specs.mean():.4f}")
    print(f"    → {'espectro del Laplaciano estable' if specs.mean() < 0.1 else 'espectro fluctúa'}")
    print(f"  • Overlap de roles (categórico): {roles.mean():.3f}")
    print(f"    → {'roles estables' if roles.mean() > 0.7 else 'roles inestables'}")
    print(f"  • ρrole (Spearman, Tabla 7 etapa 3): {rho_role_mean:.3f} → {veredicto_rho}")
    print(f"  • La triangulación con los Paradigmas 1 y 2 permite clasificar")
    print(f"    cada arista en la matriz de diagnóstico multidimensional.")
    print(f"{'═'*82}\n")

    return pairwise, invariants, node_role_summary

if __name__ == "__main__" and EJECUTAR_PARTE6:
    main()

# ============================================================================
# Paradigma 4
# ============================================================================

def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p + 1e-12
    q = q + 1e-12
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    dkl_pm = np.sum(p * np.log(p / m))
    dkl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (dkl_pm + dkl_qm))


def wasserstein_distance_1d(p: np.ndarray, q: np.ndarray) -> float:
    p = p + 1e-12
    q = q + 1e-12
    p = p / p.sum()
    q = q / q.sum()
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    return float(np.sum(np.abs(cdf_p - cdf_q)))

def estimate_mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import mutual_info_score
    n = len(x)
    if n < 10:
        return 0.0
    x_disc = np.digitize(x, bins=np.linspace(x.min(), x.max() + 1e-9, 10))
    y_disc = np.digitize(y, bins=np.linspace(y.min(), y.max() + 1e-9, 10))
    return float(mutual_info_score(x_disc, y_disc))

def compute_information_theory_metrics(
    all_matrices: List[np.ndarray],
    consensus_matrix: np.ndarray,
    val_losses: List[float],
    labels: List[str],
    p_top: Optional[float] = None,
) -> Dict:
    n_models = len(all_matrices)
    n = len(labels)
    print(f"\n{'━'*82}")
    print(f"  PARADIGMA 4: TEORÍA DE LA INFORMACIÓN")
    print(f"{'━'*82}")
    calcular_eta(all_matrices)
    p_top = p_efectivo(p_top)
    print(f"  Discretización: Top-P con P = {p_top} · η = {ETA_MARGEN:.6f}")

    djs_values = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            djs_rows = [jensen_shannon_divergence(all_matrices[i][row],
                                                  all_matrices[j][row])
                        for row in range(n)]
            djs_values.append(np.mean(djs_rows))
    djs_values = np.array(djs_values)
    print(f"\n  [4.0] Divergencia Jensen-Shannon (D_JS) entre pares:")
    print(f"    Media:   {djs_values.mean():.6f}")
    print(f"    Std:     {djs_values.std():.6f}")
    print(f"    Mediana: {np.median(djs_values):.6f}")
    print(f"    Min/Max: {djs_values.min():.6f} / {djs_values.max():.6f}")
    print(f"    (D_JS<0.05=casi idénticas, >0.3=claramente distintas)")

    wasserstein_values = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            w_rows = [wasserstein_distance_1d(all_matrices[i][row],
                                                all_matrices[j][row])
                       for row in range(n)]
            wasserstein_values.append(np.mean(w_rows))
    wasserstein_values = np.array(wasserstein_values)
    print(f"\n  [4.2] Distancia de Wasserstein-1 (EMD) entre pares:")
    print(f"    Media:   {wasserstein_values.mean():.6f}")
    print(f"    Std:     {wasserstein_values.std():.6f}")
    print(f"    Mediana: {np.median(wasserstein_values):.6f}")
    print(f"    Min/Max: {wasserstein_values.min():.6f} / {wasserstein_values.max():.6f}")
    print(f"    (W bajo + D_JS alto = movimiento suave, estructura preservada)")

    all_flat = np.stack([m.flatten() for m in all_matrices])
    per_entry_std = all_flat.std(axis=0)
    valid_mask = per_entry_std > 1e-8
    h_diff = 0.5 * np.log(2 * np.pi * np.e * per_entry_std[valid_mask]**2 + 1e-12)
    h_diff_mean = float(h_diff.mean())
    print(f"\n  [4.3] Entropía diferencial de la atención:")
    print(f"    h(A) promedio: {h_diff_mean:.6f}")
    print(f"    (baja=concentrada, alta=dispersa)")

    edge_presence = {}
    for m_idx, matrix in enumerate(all_matrices):
        # Indicador binario de presencia de arista, con la MISMA discretización
        # Top-P que el resto del framework (antes: Top-K con K_FOR_MI=5).
        top_set = discretizar(matrix, p_top)
        for e in top_set:
            if e not in edge_presence:
                edge_presence[e] = np.zeros(n_models)
            edge_presence[e][m_idx] = 1

    val_losses_arr = np.array(val_losses)
    mi_values = {}
    if val_losses_arr.std() > 1e-8 and len(edge_presence) > 0:
        for edge, presence in edge_presence.items():
            if 0 < presence.sum() < n_models:
                mi = estimate_mutual_information(presence, val_losses_arr)
                mi_values[edge] = mi
        print(f"\n  [4.1] Información mutua I(arista; val_loss):")
        if mi_values:
            mi_arr = np.array(list(mi_values.values()))
            print(f"    MI promedio: {mi_arr.mean():.6f}")
            print(f"    MI máximo:   {mi_arr.max():.6f}")
            print(f"    Top-5 aristas con mayor MI:")
            for (i, j), mi in sorted(mi_values.items(), key=lambda x: -x[1])[:5]:
                print(f"      {labels[i]:12s} → {labels[j]:12s} : MI = {mi:.6f}")
        else:
            print(f"    No se pudo estimar MI (poca variabilidad en val_loss)")
    else:
        print(f"\n  [4.1] Información mutua: omitida (val_loss constante)")
        print(f"         val_loss σ = {val_losses_arr.std():.6f}")
        print(f"         RECOMENDACIÓN: usar pérdida por batch para mayor variabilidad.")

    if len(edge_presence) >= 2:
        edges_list = list(edge_presence.keys())[:30]
        presence_matrix = np.array([edge_presence[e] for e in edges_list])
        nmi_pairs = []
        for i in range(len(edges_list)):
            for j in range(i + 1, len(edges_list)):
                p1, p2 = presence_matrix[i], presence_matrix[j]
                if p1.sum() > 0 and p2.sum() > 0:
                    nmi = estimate_mutual_information(p1, p2)
                    nmi_pairs.append((edges_list[i], edges_list[j], nmi))
        print(f"\n  [4.1b] NMI entre pares de aristas (co-ocurrencia):")
        if nmi_pairs:
            nmi_values = [x[2] for x in nmi_pairs]
            print(f"    NMI promedio: {np.mean(nmi_values):.6f}")
            print(f"    NMI máximo:   {np.max(nmi_values):.6f}")
            print(f"    Top-5 pares con mayor NMI (sustitutivas/complementarias):")
            for (i1, j1), (i2, j2), nmi in sorted(nmi_pairs, key=lambda x: -x[2])[:5]:
                print(f"      {labels[i1]:8s}→{labels[j1]:8s}  ∥  "
                      f"{labels[i2]:8s}→{labels[j2]:8s}  : NMI = {nmi:.6f}")
    else:
        print(f"\n  [4.1b] NMI entre aristas: omitida (pocas aristas)")

    return {
        "djs_mean": float(djs_values.mean()),
        "djs_std": float(djs_values.std()),
        "djs_median": float(np.median(djs_values)),
        "wasserstein_mean": float(wasserstein_values.mean()),
        "wasserstein_std": float(wasserstein_values.std()),
        "wasserstein_median": float(np.median(wasserstein_values)),
        "h_differential": h_diff_mean,
        "mi_values": {f"{i}_{j}": v for (i, j), v in mi_values.items()},
        "nmi_max": float(np.max([x[2] for x in nmi_pairs])) if nmi_pairs else 0.0,
        "nmi_mean": float(np.mean([x[2] for x in nmi_pairs])) if nmi_pairs else 0.0,
    }

def compute_graph_level_mi(
    all_matrices: List[np.ndarray],
    val_losses: List[float],
    labels: List[str],
    p_top: Optional[float] = None,
) -> Dict:
    from sklearn.metrics import mutual_info_score
    from itertools import combinations

    n_models = len(all_matrices)
    n = len(labels)
    print(f"\n{'━'*82}")
    print(f"  HIPÓTESIS DE INFORMACIÓN DISTRIBUIDA: I(e;L) vs I(G;L)")
    print(f"{'━'*82}")

    edge_presence = {}
    graph_vectors = []
    all_possible_edges = [(i, j) for i in range(n) for j in range(n) if i != j]
    p_top = p_efectivo(p_top)
    for m_idx, matrix in enumerate(all_matrices):
        top_set = discretizar(matrix, p_top)     # Top-P, igual que el resto
        vec = np.array([1.0 if e in top_set else 0.0 for e in all_possible_edges])
        graph_vectors.append(vec)
        for e in top_set:
            if e not in edge_presence:
                edge_presence[e] = np.zeros(n_models)
            edge_presence[e][m_idx] = 1.0
    graph_vectors = np.array(graph_vectors)
    val_losses_arr = np.array(val_losses)

    print(f"\n  [1] I(e;L) — Información mutua por arista individual:")
    mi_per_edge = {}
    if val_losses_arr.std() > 1e-8:
        for edge, presence in edge_presence.items():
            if 0 < presence.sum() < n_models:
                vl_disc = np.digitize(val_losses_arr,
                    bins=np.linspace(val_losses_arr.min(), val_losses_arr.max() + 1e-9, 5))
                mi = float(mutual_info_score(presence.astype(int), vl_disc))
                mi_per_edge[edge] = mi
        if mi_per_edge:
            mi_values = np.array(list(mi_per_edge.values()))
            print(f"    I(e;L) media:   {mi_values.mean():.6f}")
            print(f"    I(e;L) máximo:  {mi_values.max():.6f}")
            print(f"    I(e;L) mediana: {np.median(mi_values):.6f}")
            print(f"    Top-5 aristas con mayor I(e;L):")
            for (i, j), mi in sorted(mi_per_edge.items(), key=lambda x: -x[1])[:5]:
                print(f"      {labels[i]:12s} → {labels[j]:12s} : I(e;L) = {mi:.6f}")
        else:
            print(f"    No se pudo estimar (poca variabilidad en val_loss)")
    else:
        print(f"    val_loss prácticamente constante (σ={val_losses_arr.std():.6f})")
        print(f"    No se puede estimar MI con val_loss agregado.")
        print(f"    RECOMENDACIÓN: usar pérdida por batch para mayor variabilidad.")

    print(f"\n  [2] I(G;L) — Información mutua del grafo completo:")
    igl = 0.0
    n_clusters = 0
    if val_losses_arr.std() > 1e-8 and n_models >= 10:
        jaccard_matrix = np.zeros((n_models, n_models))
        for i, j in combinations(range(n_models), 2):
            intersection = np.sum(graph_vectors[i] * graph_vectors[j])
            union = np.sum(np.maximum(graph_vectors[i], graph_vectors[j]))
            jac = intersection / max(union, 1)
            jaccard_matrix[i, j] = jac
            jaccard_matrix[j, i] = jac

        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        dist_matrix = 1 - jaccard_matrix
        np.fill_diagonal(dist_matrix, 0)
        condensed = squareform(dist_matrix, checks=False)
        Z = linkage(condensed, method='average')
        clusters = fcluster(Z, t=0.5, criterion='distance')
        n_clusters = len(set(clusters))

        vl_disc = np.digitize(val_losses_arr,
            bins=np.linspace(val_losses_arr.min(), val_losses_arr.max() + 1e-9, 5))
        vl_counts = np.bincount(vl_disc)
        vl_probs = vl_counts / max(vl_counts.sum(), 1)
        H_L = -np.sum(vl_probs[vl_probs > 0] * np.log(vl_probs[vl_probs > 0]))

        H_L_given_G = 0.0
        for c in set(clusters):
            mask = clusters == c
            n_c = mask.sum()
            if n_c < 2:
                continue
            vl_c = vl_disc[mask]
            vl_c_counts = np.bincount(vl_c)
            vl_c_probs = vl_c_counts / max(vl_c_counts.sum(), 1)
            H_L_c = -np.sum(vl_c_probs[vl_c_probs > 0] * np.log(vl_c_probs[vl_c_probs > 0]))
            H_L_given_G += (n_c / n_models) * H_L_c
        igl = H_L - H_L_given_G

        print(f"    H(L) = {H_L:.6f}")
        print(f"    H(L|G) = {H_L_given_G:.6f}")
        print(f"    I(G;L) = {igl:.6f}")
        print(f"    Ratio I(G;L)/H(L) = {igl/max(H_L, 1e-12):.4f}")
        if mi_per_edge:
            mi_edge_mean = np.mean(list(mi_per_edge.values()))
            print(f"\n    Comparación:")
            print(f"      I(e;L) media  = {mi_edge_mean:.6f}")
            print(f"      I(G;L)        = {igl:.6f}")
            ratio = igl / max(mi_edge_mean, 1e-12)
            print(f"      Ratio I(G;L)/I(e;L) = {ratio:.2f}x")
            if ratio > 2:
                print(f"      ✅ Evidencia de información distribuida: I(G;L) >> I(e;L)")
            elif ratio > 1:
                print(f"      🟡 Evidencia moderada: I(G;L) > I(e;L)")
            else:
                print(f"      ❌ No hay evidencia de información distribuida")
    else:
        print(f"    No se pudo estimar (val_loss constante o pocos modelos)")

    return {
        "igl_graph": float(igl),
        "igl_edge_mean": float(np.mean(list(mi_per_edge.values()))) if mi_per_edge else 0.0,
        "igl_edge_max": float(np.max(list(mi_per_edge.values()))) if mi_per_edge else 0.0,
        "mi_per_edge": {f"{i}_{j}": v for (i, j), v in mi_per_edge.items()},
        "n_clusters": int(n_clusters),
    }

def plot_information_metrics(info_results: Dict, output_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.bar(["D_JS media", "D_JS mediana", "D_JS max"],
           [info_results["djs_mean"], info_results["djs_median"], info_results.get("djs_max", info_results["djs_mean"])],
           color=["#3498db", "#2980b9", "#1abc9c"], edgecolor="white")
    ax.axhline(y=0.05, color="orange", linestyle="--", label="umbral 'casi idénticas' (0.05)")
    ax.axhline(y=0.3, color="red", linestyle="--", label="umbral 'claramente distintas' (0.3)")
    ax.set_title("Divergencia Jensen-Shannon (D_JS)", fontweight="bold")
    ax.set_ylabel("D_JS"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    ax.bar(["Wasserstein media", "W mediana", "h(A) entropía"],
           [info_results["wasserstein_mean"], info_results["wasserstein_median"],
            info_results["h_differential"]],
           color=["#9b59b6", "#8e44ad", "#16a085"], edgecolor="white")
    ax.set_title("Distancia Wasserstein-1 + Entropía diferencial", fontweight="bold")
    ax.set_ylabel("Valor"); ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Paradigma 4: Teoría de la Información", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight"); plt.close()

def main():
    print(f"\n{'═'*82}")
    print(f" PARADIGMA 4 — TEORÍA DE LA INFORMACIÓN")
    print(f" Pregunta: ¿Aporta conocimiento el grafo de atención?")
    print(f" Fundamento: D_JS, Wasserstein, MI, NMI")
    print(f"{'═'*82}")

    # find_cached_matrices() devuelve (matrices, seeds) — un tercer valor
    # aquí (antes "batch_losses_all") nunca existió en esa función y hacía
    # que esta línea lanzara ValueError al desempaquetar. El val_loss real
    # se lee aparte, del val_losses.json que ahora guarda main_parte2_multiseed.
    all_matrices, seeds = find_cached_matrices()
    if not all_matrices:
        print("  ❌ No se encontraron matrices cacheadas.")
        return None
    M = len(all_matrices)
    print(f"  Modelos disponibles: M = {M}")

    val_losses_path = os.path.join(DIR_MATRICES, "val_losses.json")
    if os.path.exists(val_losses_path):
        with open(val_losses_path) as f:
            val_losses_dict = {int(k): v for k, v in json.load(f).items()}
        val_losses = [val_losses_dict.get(s, 0.0) for s in seeds]
        n_missing = sum(1 for s in seeds if s not in val_losses_dict)
        if n_missing:
            print(f"  ⚠️  {n_missing}/{M} semillas sin val_loss guardado "
                  f"(entrenadas antes de este fix) — la MI puede subestimarse.")
    else:
        val_losses = [0.0] * M
        print(f"  ⚠️  No existe {val_losses_path} — estas 50 semillas se "
              f"entrenaron antes de que se empezara a guardar val_loss. "
              f"Reentrena (forzar_reentrenamiento=True) para habilitar I(G;Y) real.")

    consensus_matrix = np.mean(all_matrices, axis=0)
    info_results = compute_information_theory_metrics(
        all_matrices=all_matrices,
        consensus_matrix=consensus_matrix,
        val_losses=val_losses,
        labels=INDEX_NAMES,
    )

    igl_results = compute_graph_level_mi(
        all_matrices=all_matrices,
        val_losses=val_losses,
        labels=INDEX_NAMES,
    )

    plot_information_metrics(info_results,
                              os.path.join(DIR_P4, "information_metrics.png"))
    print(f"\n  ✅ Visualizaciones guardadas en {DIR_P4}/")

    results = {
        "discretizacion": "Top-P", "top_p": p_efectivo(None),
        "eta": float(ETA_MARGEN),
        "M": M,
        **info_results,
        "igl_graph": igl_results["igl_graph"],
        "igl_edge_mean": igl_results["igl_edge_mean"],
        "igl_edge_max": igl_results["igl_edge_max"],
        "n_clusters": igl_results["n_clusters"],
        # La triangulación (Ie) lee esta clave; antes se calculaba pero
        # nunca se guardaba, así que Ie siempre salía "no evaluable" aunque
        # el val_loss estuviera disponible.
        "mi_per_edge": igl_results["mi_per_edge"],
    }
    with open(os.path.join(DIR_P4, "paradigm4_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  ✅ Resultados: {DIR_P4}/paradigm4_results.json")

    print(f"\n{'═'*82}")
    print(f" RESUMEN — PARADIGMA 4 (TEORÍA DE LA INFORMACIÓN)")
    print(f"{'═'*82}")
    print(f"  • D_JS media = {info_results['djs_mean']:.4f} → "
          f"{'casi idénticas' if info_results['djs_mean'] < 0.05 else 'diferencia notable' if info_results['djs_mean'] < 0.3 else 'claramente distintas'}")
    print(f"  • Wasserstein media = {info_results['wasserstein_mean']:.4f}")
    print(f"  • Entropía diferencial h(A) = {info_results['h_differential']:.4f}")
    print(f"  • NMI máxima entre aristas = {info_results['nmi_max']:.4f}")
    print(f"  • I(G;L) = {igl_results['igl_graph']:.4f} (información distribuida)")
    if info_results['nmi_max'] > 0.4:
        print(f"  • ⭐ Detectadas aristas funcionalmente redundantes (NMI > 0.4)")
    print(f"{'═'*82}\n")

    return info_results

if __name__ == "__main__" and EJECUTAR_PARTE9:
    main()


# ============================================================================
# Paradigma 5
# ============================================================================

def build_pearson_baseline(stack: np.ndarray, labels: List[str],
                            threshold: float = 0.8) -> Set[Tuple[int, int]]:
    n = stack.shape[-1]
    flat = stack.reshape(-1, n)
    corr = np.corrcoef(flat.T)
    edges = set()
    for i in range(n):
        for j in range(n):
            if i != j and abs(corr[i, j]) > threshold:
                edges.add((i, j))
    return edges

def matriz_correlacion_parcial(stack: np.ndarray) -> np.ndarray:
    """|correlacion parcial| entre los n indices, controlando por los otros n-2.

    POR QUE HACIA FALTA ESTO Y NO ESTABA. El unico baseline lineal del
    pipeline era Pearson, que es correlacion MARGINAL. Sobre estos 12 indices
    la marginal es densa y casi inutil (|r| medio 0.836, mediana 0.893, 47% de
    los pares por encima de 0.9): todo correlaciona con todo porque comparten
    el modo dominante de la escena. La correlacion PARCIAL, en cambio, sale
    escasa y especifica (|pc| medio 0.283, mediana 0.242), y ordena los pares
    de forma casi independiente de la marginal (Spearman entre ambos ordenes
    sobre los 66 pares = 0.194).

    Es ademas el baseline CORRECTO para lo que el framework afirma medir: una
    arista i->j deberia significar "j aporta algo sobre i que no aportan los
    demas", que es exactamente dependencia condicional, no marginal.

    Validacion sobre indices_12.npy: el ranking por |pc| coloca en los puestos
    3 y 4 de 66 las dos identidades algebraicas exactas del dominio
    (MARI = ARI*B7 con |pc|=0.774, KNDVI = tanh(NDVI^2) con |pc|=0.743), sin
    que nadie se las declare. O sea, la parcial recupera el ground truth
    conocido y sirve como control positivo del framework.
    """
    n = stack.shape[-1]
    X = stack.reshape(-1, n).astype(np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    C = np.corrcoef(X.T)
    # Ridge minimo para que la inversion no explote con canales colineales.
    P = np.linalg.inv(C + 1e-8 * np.eye(n))
    d = np.sqrt(np.diag(P))
    PC = -P / np.outer(d, d)
    np.fill_diagonal(PC, 1.0)
    return np.abs(PC)


def build_partial_correlation_baseline(stack: np.ndarray, labels: List[str],
                                       threshold: float = 0.3
                                       ) -> Set[Tuple[int, int]]:
    """Grafo de dependencia CONDICIONAL (matriz de precision).

    Simetrico por construccion, asi que se emiten las dos direcciones de cada
    par: la parcial no distingue sentido, y hay que declararlo al compararla
    con un grafo dirigido.
    """
    PC = matriz_correlacion_parcial(stack)
    n = PC.shape[0]
    edges = set()
    for i in range(n):
        for j in range(n):
            if i != j and PC[i, j] > threshold:
                edges.add((i, j))
    return edges


def build_granger_baseline(stack: np.ndarray, labels: List[str],
                            max_lag: int = 3, p_threshold: float = 0.05) -> Set[Tuple[int, int]]:
    """Grafo de causalidad de Granger sobre las series espaciales medias.

    LIMITACIÓN QUE HAY QUE DECLARAR: Granger asume muestreo EQUIESPACIADO, y
    estas fechas no lo son. Medido sobre las 286 imágenes: hueco mediano 5 días,
    p90 20 días, máximo 110. Con max_lag=3, "tres pasos atrás" significa 15 días
    en una parte de la serie y 300 en otra, así que el test no mide lo que dice
    medir. Se deja porque el informe lo pide como baseline (§21), pero su
    resultado no es interpretable como causalidad temporal.
    """
    n = stack.shape[-1]
    series = stack.mean(axis=(1, 2))
    fechas = cargar_fechas_stack(len(stack))
    if fechas is not None and len(fechas) > 1:
        g = np.diff(np.asarray(fechas, dtype=np.int64)) / 86_400_000.0
        print(f"    [!] Granger asume muestreo equiespaciado. Huecos reales: "
              f"mediana {np.median(g):.0f} d · p90 {np.percentile(g, 90):.0f} d "
              f"· máx {g.max():.0f} d.")
        print(f"        Con max_lag={max_lag}, el desfase real varía entre "
              f"{max_lag*np.median(g):.0f} y {max_lag*g.max():.0f} días según el "
              f"tramo: el resultado NO es interpretable como causalidad temporal.")
    edges = set()
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                data = np.column_stack([series[:, j], series[:, i]])
                try:
                    results = grangercausalitytests(data, maxlag=max_lag, verbose=False)
                    p_value = results[1][0]['ssr_ftest'][1]
                    if p_value < p_threshold:
                        edges.add((i, j))
                except Exception:
                    pass
    except ImportError:
        print("    ⚠️ statsmodels no disponible, saltando Granger")
    return edges


# ----------------------------------------------------------------------------
# Convergent Cross Mapping (CCM) — reemplazo propuesto de Granger para P5
# (LINEAMIENTO_NUEVOS_PARADIGMAS.md §2). Se agrega AL LADO de Granger, sin
# tocar sus claves de salida (granger_graph_size, jaccard_granger, etc.):
# primero se audita cuánto difieren antes de que uno reemplace al otro como
# baseline temporal-causal de la compuerta Se.
#
# Algoritmo estándar (Sugihara et al. 2012, "Detecting Causality in Complex
# Ecosystems"): para inferir si el índice i influye sobre el índice j, se
# reconstruye el atractor sombra de j (delay embedding) y se intenta
# recuperar i desde los vecinos más cercanos en ESE atractor. Si se recupera
# bien, la dinámica de j "contiene" información de i → arista (i,j).
# Significancia: contra un nulo de superrogados de fase aleatoria (Theiler et
# al. 1992), que conserva el espectro de potencia y rompe el orden temporal.
#
# LIMITACIÓN SIN VALIDAR (declarar si se cita el resultado): el embedding usa
# índice temporal (orden de muestra), no tiempo real en días. Con huecos de
# hasta 110 días (medido en build_granger_baseline de arriba), un "vecino
# más cercano en índice" puede estar lejos en tiempo real. Ningún paper
# 2024-2026 revisado confirma tolerancia de CCM a huecos de esta magnitud —
# correr un control sintético con el mismo patrón de huecos antes de tratar
# el resultado como causalidad válida (ver Ndikum Nji, Mostafa & Wang 2025,
# PoIDS, para el benchmarking de CCM contra Granger/PCMCI/VarLiNGAM en datos
# climáticos reales que motiva este reemplazo).
# ----------------------------------------------------------------------------

def _shadow_manifold(y: np.ndarray, E: int, tau: int) -> Tuple[np.ndarray, np.ndarray]:
    """Delay embedding de y en E dimensiones con paso tau.

    Devuelve (M, idx): M es (T', E), fila t = [y[t], y[t-tau], ..., y[t-(E-1)tau]];
    idx son los índices temporales originales de cada fila de M.
    """
    T = len(y)
    idx = np.arange((E - 1) * tau, T)
    if len(idx) == 0:
        return np.zeros((0, E)), idx
    M = np.stack([y[idx - k * tau] for k in range(E)], axis=1)
    return M, idx


def _ccm_skill(x: np.ndarray, y: np.ndarray, E: int = 3, tau: int = 1) -> float:
    """Habilidad de cross-map: predice x desde los E+1 vecinos más cercanos
    en el atractor sombra de y (ponderados por distancia, kernel exponencial
    de Sugihara et al. 2012). rho = correlación(x_predicho, x_observado)."""
    My, idx = _shadow_manifold(y, E, tau)
    T = len(My)
    if T < E + 3:
        return float("nan")
    x_target = x[idx]
    x_hat = np.empty(T)
    k = E + 1
    for t in range(T):
        d = np.linalg.norm(My - My[t], axis=1)
        d[t] = np.inf
        vecinos = np.argpartition(d, k)[:k]
        dmin = d[vecinos].min()
        w = np.exp(-d[vecinos] / max(dmin, 1e-9))
        w_sum = w.sum()
        if w_sum <= 0 or not np.isfinite(w_sum):
            x_hat[t] = np.nan
            continue
        x_hat[t] = np.dot(w / w_sum, x_target[vecinos])
    valid = np.isfinite(x_hat)
    if valid.sum() < 3 or np.std(x_hat[valid]) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x_hat[valid], x_target[valid])[0, 1])


def _surrogate_fase(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Superrogado de fase aleatoria (Theiler et al. 1992): mismo espectro de
    potencia que x, fase aleatorizada — nulo estándar para causalidad no
    lineal, rompe el orden temporal sin tocar la autocorrelación de 2° orden."""
    X = np.fft.rfft(x)
    fases = rng.uniform(0.0, 2 * np.pi, size=X.shape[0])
    fases[0] = 0.0
    if len(x) % 2 == 0 and len(fases) > 1:
        fases[-1] = 0.0
    X_sur = np.abs(X) * np.exp(1j * fases)
    return np.fft.irfft(X_sur, n=len(x))


def build_ccm_baseline(stack: np.ndarray, labels: List[str],
                       embedding_dim: int = 3, tau: int = 1,
                       p_threshold: float = 0.05,
                       n_surrogates: int = 100,
                       ) -> Tuple[Set[Tuple[int, int]], np.ndarray]:
    """Grafo de Convergent Cross Mapping sobre las series espaciales medias.

    Ver limitación sin validar en el comentario de la sección de arriba.
    Devuelve (edges, ccm_matrix): edges es el grafo dirigido significativo
    (p empírico < p_threshold contra el nulo de superrogados), ccm_matrix es
    la matriz continua (n,n) de habilidad de cross-map (para usar en
    discretizar_igualando, igual que granger_matrix).
    """
    n = stack.shape[-1]
    series = stack.mean(axis=(1, 2)).astype(np.float64)      # (T, n)
    mu, sd = series.mean(axis=0), series.std(axis=0)
    series = (series - mu) / np.where(sd > 1e-12, sd, 1.0)

    rng = np.random.default_rng(SEMILLA_NULO)
    ccm_matrix = np.zeros((n, n))
    edges: Set[Tuple[int, int]] = set()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            rho_obs = _ccm_skill(series[:, i], series[:, j], embedding_dim, tau)
            if not np.isfinite(rho_obs):
                continue
            ccm_matrix[i, j] = rho_obs
            rho_nulo = np.array([
                _ccm_skill(series[:, i], _surrogate_fase(series[:, j], rng),
                          embedding_dim, tau)
                for _ in range(n_surrogates)
            ])
            rho_nulo = rho_nulo[np.isfinite(rho_nulo)]
            if len(rho_nulo) < 10:
                continue
            p_emp = float((rho_nulo >= rho_obs).mean())
            if p_emp < p_threshold:
                edges.add((i, j))
    return edges, ccm_matrix


def build_random_baseline(n: int, k: int, n_realizations: int = 1000) -> List[Set]:
    p = k / (n * (n - 1))
    graphs = []
    for _ in range(n_realizations):
        edges = set()
        for i in range(n):
            for j in range(n):
                if i != j and np.random.random() < p:
                    edges.add((i, j))
        graphs.append(edges)
    return graphs

def build_consensus_graph(all_matrices: List[np.ndarray],
                           p_top: Optional[float] = None,
                           threshold: float = 0.7) -> Set[Tuple[int, int]]:
    """Grafo de consenso: aristas con frecuencia ≥ threshold sobre las M
    realizaciones, discretizando cada matriz con Top-P (§8.1 del informe)."""
    n_models = len(all_matrices)
    edge_freq = Counter()
    p_top = p_efectivo(p_top)
    for matrix in all_matrices:
        top_set = discretizar(matrix, p_top)
        for e in top_set:
            edge_freq[e] += 1
    consensus = set()
    for edge, freq in edge_freq.items():
        if freq / n_models >= threshold:
            consensus.add(edge)
    return consensus
def compute_baselines_comparison(
    all_matrices: List[np.ndarray],
    consensus_graph: Set,
    stack: np.ndarray,
    labels: List[str],
    p_top: Optional[float] = None,
) -> Dict:
    n = len(labels)
    print(f"\n{'━'*82}")
    print(f"  PARADIGMA 5: BASELINES DETERMINISTAS — CONTROL DE CALIDAD")
    print(f"{'━'*82}")

    print(f"\n  [5.1] Grafo de correlación de Pearson (τ=0.8):")
    if stack is not None:
        pearson_graph = build_pearson_baseline(stack, labels, threshold=0.8)
        print(f"    Aristas en G_Pearson: {len(pearson_graph)}")
        if pearson_graph:
            for i, j in sorted(pearson_graph)[:10]:
                print(f"      {labels[i]:12s} → {labels[j]:12s}")
        j_pearson = len(consensus_graph & pearson_graph) / max(len(consensus_graph | pearson_graph), 1)
        print(f"    Jaccard(G_consensus, G_Pearson) = {j_pearson:.4f}")
        print(f"    (J>0.7=solo lineal, J<0.3=no lineal complejo)")
    else:
        print(f"    ⚠️ Stack no disponible, saltando Pearson")
        pearson_graph = set()
        j_pearson = 0.0

    print(f"\n  [5.1b] Grafo de correlacion PARCIAL (dependencia condicional, tau=0.3):")
    if stack is not None:
        parcial_graph = build_partial_correlation_baseline(stack, labels,
                                                           threshold=0.3)
        print(f"    Aristas en G_parcial: {len(parcial_graph)}")
        PCm = matriz_correlacion_parcial(stack)
        pares = sorted(((PCm[i, j], labels[i], labels[j])
                        for i in range(len(labels))
                        for j in range(i + 1, len(labels))), reverse=True)
        print(f"    Top-6 por |pc| (ground truth de dependencia condicional):")
        for pc_v, na, nb in pares[:6]:
            print(f"      {na:12s} - {nb:12s}  |pc|={pc_v:.3f}")
        j_parcial = (len(consensus_graph & parcial_graph) /
                     max(len(consensus_graph | parcial_graph), 1))
        print(f"    Jaccard(G_consensus, G_parcial) = {j_parcial:.4f}  "
              f"(vs Pearson {j_pearson:.4f})")
        # AVISO: comparar las dos Jaccard directamente es INCORRECTO. Jaccard
        # depende del tamano del grafo, y G_Pearson (tau=0.8) y G_parcial
        # (tau=0.3) no tienen la misma densidad. Medido en la corrida (4):
        # Pearson |G|=100, recall 0.871, esperado por azar 0.758 -> 1.15x
        # parcial |G|= 54, recall 0.484, esperado por azar 0.409 -> 1.18x
        # o sea enriquecimiento casi identico; la diferencia de Jaccard era
        # puro efecto de tamano. El test valido es el de rango, sin umbral:
        _tot = len(labels) * (len(labels) - 1) / 2
        for _nm, _g in (("Pearson", pearson_graph), ("parcial", parcial_graph)):
            _inter = len(consensus_graph & _g)
            _rec = _inter / max(len(consensus_graph), 1)
            _esp = len(_g) / max(2 * _tot, 1)
            print(f"      {_nm:<8s} recall {_rec:.3f}  azar {_esp:.3f}  "
                  f"enriquecimiento {_rec / max(_esp, 1e-9):.2f}x")

        # TEST SIN UMBRAL: correlacion de rangos entre la frecuencia de cada
        # arista y cada baseline, sobre los n(n-1)/2 pares. No depende de
        # ningun corte y no lo confunde el tamano del grafo. Es el que decide
        # si la atencion sigue dependencia condicional o correlacion marginal.
        try:
            from scipy.stats import spearmanr as _sp
            _Cm = np.corrcoef(stack.reshape(-1, len(labels)).T)
            _pares = [(i, j) for i in range(len(labels))
                      for j in range(i + 1, len(labels))]
            # Peso medio de atencion por par sobre las M semillas,
            # simetrizado. Sin umbral en ninguno de los dos lados: no depende
            # de Top-P ni del corte de los baselines.
            _Am = np.mean(np.stack([np.asarray(m, dtype=np.float64)
                                    for m in all_matrices]), axis=0)
            _f = np.array([max(_Am[i, j], _Am[j, i]) for i, j in _pares])
            _pc = np.array([PCm[i, j] for i, j in _pares])
            _rr = np.array([abs(_Cm[i, j]) for i, j in _pares])
            _sq, _sr = _sp(_f, _pc), _sp(_f, _rr)
            print(f"    TEST SIN UMBRAL (frecuencia de arista vs baseline, "
                  f"{len(_pares)} pares):")
            print(f"      Spearman(frec, |pc| parcial ) = {_sq.statistic:+.3f}"
                  f"   p={_sq.pvalue:.4f}")
            print(f"      Spearman(frec, |r|  marginal) = {_sr.statistic:+.3f}"
                  f"   p={_sr.pvalue:.4f}")
            if _sr.statistic > _sq.statistic:
                print(f"      -> la atencion sigue correlacion MARGINAL: "
                      f"atajo colineal, no dependencia condicional.")
            else:
                print(f"      -> la atencion sigue dependencia CONDICIONAL, "
                      f"que es lo que el framework afirma medir.")
        except Exception as _e:
            print(f"    (test de rango no disponible: {_e})")
    else:
        print(f"    [!] Stack no disponible, saltando parcial")
        parcial_graph = set()
        j_parcial = 0.0

    print(f"\n  [5.2] Grafo de causalidad de Granger (lag=3, p<0.05):")
    if stack is not None:
        granger_graph = build_granger_baseline(stack, labels, max_lag=3)
        print(f"    Aristas en G_Granger: {len(granger_graph)}")
        if granger_graph:
            for i, j in sorted(granger_graph)[:10]:
                print(f"      {labels[i]:12s} → {labels[j]:12s}")
        j_granger = len(consensus_graph & granger_graph) / max(len(consensus_graph | granger_graph), 1)
        print(f"    Jaccard(G_consensus, G_Granger) = {j_granger:.4f}")
        print(f"    (J alto=plausibilidad temporal, J bajo=relaciones no temporales)")
    else:
        print(f"    ⚠️ Stack no disponible, saltando Granger")
        granger_graph = set()
        j_granger = 0.0

    print(f"\n  [5.2b] Convergent Cross Mapping (reemplazo propuesto de "
          f"Granger, ver build_ccm_baseline):")
    if stack is not None:
        ccm_graph, _ccm_matrix_consensus = build_ccm_baseline(stack, labels)
        print(f"    Aristas en G_CCM: {len(ccm_graph)}")
        if ccm_graph:
            for i, j in sorted(ccm_graph)[:10]:
                print(f"      {labels[i]:12s} → {labels[j]:12s}")
        j_ccm = len(consensus_graph & ccm_graph) / max(len(consensus_graph | ccm_graph), 1)
        print(f"    Jaccard(G_consensus, G_CCM) = {j_ccm:.4f}  "
              f"(auditar contra J(G_consensus,G_Granger)={j_granger:.4f} "
              f"antes de reemplazar)")
    else:
        print(f"    ⚠️ Stack no disponible, saltando CCM")
        ccm_graph = set()
        j_ccm = 0.0

    print(f"\n  [5.3] Baseline aleatorio Erdős-Rényi:")
    # El nulo se genera con el MISMO número esperado de aristas que el grafo de
    # consenso. Antes se le pasaba K=5, así que el nulo tenía ~4.9 aristas
    # frente a las 15-27 del consenso y el Z-score de tamaño salía alto (4.7)
    # solo por comparar densidades distintas: no medía estructura.
    consensus_size = len(consensus_graph)
    random_graphs = build_random_baseline(n, consensus_size, n_realizations=1000)
    random_sizes = [len(g) for g in random_graphs]
    random_mean_size = np.mean(random_sizes)
    random_std_size = np.std(random_sizes)
    z_size = (consensus_size - random_mean_size) / max(random_std_size, 1e-12)
    print(f"    Grafo de consenso: {consensus_size} aristas")
    print(f"    Erdős-Rényi (densidad igualada): "
          f"{random_mean_size:.1f} ± {random_std_size:.1f} aristas")
    print(f"    Z-score (tamaño): {z_size:.4f}")
    print(f"    Con la densidad igualada este Z DEBE salir ~0: es una comprobación")
    print(f"    de que el nulo está bien construido, no evidencia de estructura.")
    print(f"    El contraste estructural real (modularidad y reciprocidad contra")
    print(f"    un nulo de out-grado restringido) lo hace el Paradigma 1, §21.3.")

    print(f"\n  [5.4] Resumen comparativo:")
    print(f"    {'Baseline':<25s} {'Aristas':>8s} {'Jaccard':>10s} {'Interpretación'}")
    print(f"    {'─'*70}")
    interp_p = 'Solo lineal' if j_pearson > 0.7 else 'No lineal' if j_pearson < 0.3 else 'Mixto'
    interp_g = 'Temporal' if j_granger > 0.5 else 'No temporal' if j_granger < 0.3 else 'Mixto'
    print(f"    {'Pearson (τ=0.8)':<25s} {len(pearson_graph):>8d} {j_pearson:>10.4f}   {interp_p}")
    print(f"    {'Granger (lag=3)':<25s} {len(granger_graph):>8d} {j_granger:>10.4f}   {interp_g}")
    print(f"    {'Erdős-Rényi':<25s} {random_mean_size:>8.1f} {'—':>10s}   Z={z_size:.2f}")

    return {
        "pearson_graph_size": int(len(pearson_graph)),
        "parcial_graph_size": int(len(parcial_graph)),
        "jaccard_parcial": float(j_parcial),
        "granger_graph_size": int(len(granger_graph)),
        "ccm_graph_size": int(len(ccm_graph)),
        "jaccard_pearson": float(j_pearson),
        "jaccard_granger": float(j_granger),
        "jaccard_ccm": float(j_ccm),
        "z_score_size": float(z_size),
        "consensus_size": int(consensus_size),
        "random_mean_size": float(random_mean_size),
        "random_std_size": float(random_std_size),
    }
def discretizar_igualando(B: np.ndarray,
                          referencia: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    """Discretiza B tomando, en cada fila, tantas aristas como tiene esa misma
    fila en `referencia`.

    Es el sustituto de los baselines "K-matched" ahora que la discretización es
    Top-P: el objetivo de ese control es comparar grafos con la MISMA densidad,
    y con Top-P la densidad ya no es un K fijo sino el reparto por fila que
    produjo el operador. Igualar fila a fila conserva la intención del control.
    """
    B = np.asarray(B, dtype=np.float64)
    n = B.shape[0]
    por_fila = Counter(i for (i, _) in referencia)
    aristas: Set[Tuple[int, int]] = set()
    for i in range(n):
        k_i = int(por_fila.get(i, 0))
        if k_i <= 0:
            continue
        fila = B[i].copy()
        fila[i] = -np.inf
        k_i = min(k_i, n - 1)
        top = np.argpartition(fila, -k_i)[-k_i:]
        for j in top:
            aristas.add((int(i), int(j)))
    return aristas


def compute_kmatched_baselines(
    all_matrices: List[np.ndarray],
    stack: np.ndarray,
    labels: List[str],
    p_values: Optional[List[float]] = None,
) -> Dict:
    n = len(labels)
    n_models = len(all_matrices)
    print(f"\n{'━'*82}")
    print(f"  BASELINES K-MATCHED: comparación justa con misma densidad")
    print(f"{'━'*82}")

    if stack is None:
        print(f"  ⚠️ Stack no disponible, no se pueden construir baselines K-matched.")
        return {}

    flat = stack.reshape(-1, n)
    corr_pearson = np.corrcoef(flat.T)

    print(f"  Construyendo matriz de Granger...")
    series = stack.mean(axis=(1, 2))
    granger_matrix = np.zeros((n, n))
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                data = np.column_stack([series[:, j], series[:, i]])
                try:
                    results = grangercausalitytests(data, maxlag=3, verbose=False)
                    p_value = results[1][0]['ssr_ftest'][1]
                    granger_matrix[i, j] = -p_value
                except Exception:
                    granger_matrix[i, j] = 0
    except ImportError:
        print(f"  ⚠️ statsmodels no disponible")

    print(f"  Construyendo matriz de CCM (reemplazo propuesto de Granger, "
          f"LINEAMIENTO_NUEVOS_PARADIGMAS.md §2)...")
    _, ccm_matrix = build_ccm_baseline(stack, labels)

    if p_values is None:
        p_values = [p_efectivo(None)]
    calcular_eta(all_matrices, verbose=False)

    results = {}
    for K in p_values:
        # Los tres grafos con la MISMA densidad: el de atención por Top-P, y
        # Pearson y Granger igualando fila a fila el reparto que produjo Top-P.
        mean_matrix = np.mean(all_matrices, axis=0)
        attn_edges = discretizar(mean_matrix, K)
        pearson_edges = discretizar_igualando(np.abs(corr_pearson), attn_edges)
        granger_edges = discretizar_igualando(granger_matrix, attn_edges)
        ccm_edges = discretizar_igualando(ccm_matrix, attn_edges)
        print(f"\n  P = {K}  ·  {len(attn_edges)} aristas "
              f"({len(attn_edges)/len(labels):.2f} por fila):")

        j_attn_pearson = len(attn_edges & pearson_edges) / max(len(attn_edges | pearson_edges), 1)
        j_attn_granger = len(attn_edges & granger_edges) / max(len(attn_edges | granger_edges), 1)
        j_attn_ccm = len(attn_edges & ccm_edges) / max(len(attn_edges | ccm_edges), 1)
        j_pearson_granger = len(pearson_edges & granger_edges) / max(len(pearson_edges | granger_edges), 1)

        attn_jaccard_pairs = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                edges_i = discretizar(all_matrices[i], K)
                edges_j = discretizar(all_matrices[j], K)
                j_inter = len(edges_i & edges_j)
                j_union = len(edges_i | edges_j)
                attn_jaccard_pairs.append(j_inter / max(j_union, 1))
        j_attn_attn = float(np.mean(attn_jaccard_pairs))

        print(f"    J(atención, atención) = {j_attn_attn:.4f}  ← estabilidad interna")
        print(f"    J(atención, Pearson)  = {j_attn_pearson:.4f}  ← densidad igualada")
        print(f"    J(atención, Granger)  = {j_attn_granger:.4f}  ← densidad igualada")
        print(f"    J(atención, CCM)      = {j_attn_ccm:.4f}  ← densidad igualada "
              f"(reemplazo propuesto de Granger, auditar antes de conmutar)")
        print(f"    J(Pearson, Granger)   = {j_pearson_granger:.4f}  ← referencia")
        ratio_p = j_attn_attn / max(j_attn_pearson, 1e-6)
        ratio_g = j_attn_attn / max(j_attn_granger, 1e-6)
        ratio_c = j_attn_attn / max(j_attn_ccm, 1e-6)
        print(f"    Ratio J(attn,attn)/J(attn,Pearson) = {ratio_p:.2f}x")
        print(f"    Ratio J(attn,attn)/J(attn,Granger) = {ratio_g:.2f}x")
        print(f"    Ratio J(attn,attn)/J(attn,CCM)      = {ratio_c:.2f}x")
        if ratio_p > 2 and ratio_g > 2:
            print(f"    ✅ El grafo de atención se parece más a otras realizaciones del")
            print(f"       mismo framework que a Pearson o Granger con la misma densidad.")
        else:
            print(f"    🟡 El grafo de atención tiene similitud moderada con baselines")

        results[K] = {
            "j_attn_attn": j_attn_attn,
            "j_attn_pearson": float(j_attn_pearson),
            "j_attn_granger": float(j_attn_granger),
            "j_attn_ccm": float(j_attn_ccm),
            "j_pearson_granger": float(j_pearson_granger),
            "ratio_attn_vs_pearson": float(ratio_p),
            "ratio_attn_vs_granger": float(ratio_g),
            "ratio_attn_vs_ccm": float(ratio_c),
        }

    print(f"\n  Tabla resumen con densidad igualada:")
    print(f"    {'P':>4s}  {'J(attn,attn)':>12s}  {'J(attn,Pear)':>12s}  "
          f"{'J(attn,Gran)':>12s}  {'J(Pear,Gran)':>12s}  {'Ratio A/P':>10s}  {'Ratio A/G':>10s}")
    print(f"    {'─'*80}")
    for K in p_values:
        r = results[K]
        ratio_p = r["j_attn_attn"] / max(r["j_attn_pearson"], 1e-6)
        ratio_g = r["j_attn_attn"] / max(r["j_attn_granger"], 1e-6)
        print(f"    {K:>4.2f}  {r['j_attn_attn']:>12.4f}  {r['j_attn_pearson']:>12.4f}  "
              f"{r['j_attn_granger']:>12.4f}  {r['j_pearson_granger']:>12.4f}  "
              f"{ratio_p:>10.2f}  {ratio_g:>10.2f}")

    return results
def compute_conditional_edge_dependencies(
    all_matrices: List[np.ndarray],
    labels: List[str],
    p_top: Optional[float] = None,
    n_top_pairs: int = 15,
) -> Dict:
    n = len(labels)
    n_models = len(all_matrices)
    print(f"\n{'━'*82}")
    print(f"  DEPENDENCIAS CONDICIONALES ENTRE ARISTAS")
    print(f"  P(e2=1|e1=1) vs P(e2=1|e1=0) — ¿coordinación estructural?")
    print(f"{'━'*82}")

    edge_presence: Dict[Tuple[int, int], np.ndarray] = {}
    all_possible = [(i, j) for i in range(n) for j in range(n) if i != j]
    p_top = p_efectivo(p_top)
    for m_idx, matrix in enumerate(all_matrices):
        top_set = discretizar(matrix, p_top)     # Top-P, igual que el resto
        for e in all_possible:
            if e not in edge_presence:
                edge_presence[e] = np.zeros(n_models)
            edge_presence[e][m_idx] = 1.0 if e in top_set else 0.0

    frequent_edges = [(e, edge_presence[e]) for e in all_possible if edge_presence[e].sum() >= 3]
    print(f"\n  Aristas con frecuencia ≥ 3: {len(frequent_edges)}")

    if len(frequent_edges) < 2:
        print(f"  Pocos datos para análisis condicional")
        return {}

    conditional_results = []
    todos_p: List[float] = []      # p de Wald de TODAS las comparaciones, para BH
    for idx1 in range(len(frequent_edges)):
        e1, p1 = frequent_edges[idx1]
        for idx2 in range(idx1 + 1, len(frequent_edges)):
            e2, p2 = frequent_edges[idx2]
            mask_e1_present = p1 == 1
            n_e1_present = mask_e1_present.sum()
            p_e2_given_e1 = float(p2[mask_e1_present].mean()) if n_e1_present > 0 else 0.0
            mask_e1_absent = p1 == 0
            n_e1_absent = mask_e1_absent.sum()
            p_e2_given_not_e1 = float(p2[mask_e1_absent].mean()) if n_e1_absent > 0 else 0.0

            # Sin suficientes modelos en las dos ramas no hay nada que estimar.
            if n_e1_present < COND_N_MIN or n_e1_absent < COND_N_MIN:
                continue

            # Tabla 2x2 con corrección de Haldane-Anscombe (+0.5). Evita el
            # infinito y da un odds ratio finito con cualquier celda vacía.
            a = float(p2[mask_e1_present].sum()) + 0.5          # e1=1, e2=1
            b = float(n_e1_present - p2[mask_e1_present].sum()) + 0.5
            c = float(p2[mask_e1_absent].sum()) + 0.5           # e1=0, e2=1
            dd = float(n_e1_absent - p2[mask_e1_absent].sum()) + 0.5
            odds_ratio = (a / b) / (c / dd)
            # Diferencia de riesgos: acotada en [-1, 1], es la que se puede
            # ordenar y citar sin que un denominador cero la haga explotar.
            dif_riesgo = p_e2_given_e1 - p_e2_given_not_e1
            # IC de Wald al 95% del log-odds, con las celdas corregidas.
            se_log = float(np.sqrt(1 / a + 1 / b + 1 / c + 1 / dd))
            lo = float(np.exp(np.log(odds_ratio) - 1.96 * se_log))
            hi = float(np.exp(np.log(odds_ratio) + 1.96 * se_log))
            # El ratio crudo se conserva sólo para trazabilidad, marcado.
            ratio_crudo = (p_e2_given_e1 / p_e2_given_not_e1
                           if p_e2_given_not_e1 > 0 else float("inf"))
            # p de Wald sobre el log-odds, para poder corregir por las miles de
            # comparaciones que se hacen (con 106 aristas frecuentes son 5565).
            z_wald = float(np.log(odds_ratio) / max(se_log, 1e-12))
            p_wald = float(2.0 * stats.norm.sf(abs(z_wald)))
            todos_p.append(p_wald)
            if abs(dif_riesgo) >= 0.25 and not (lo < 1.0 < hi):
                conditional_results.append({
                    "p_wald": p_wald, "z_wald": z_wald,
                    "e1": e1, "e2": e2,
                    "p_e2_given_e1": p_e2_given_e1,
                    "p_e2_given_not_e1": p_e2_given_not_e1,
                    "dif_riesgo": dif_riesgo,
                    "odds_ratio": odds_ratio,
                    "or_ic95": [lo, hi],
                    "ratio_crudo_no_citar": ratio_crudo,
                    "ratio": odds_ratio,   # compatibilidad con el resto
                    "n_e1_present": int(n_e1_present),
                    "n_e1_absent": int(n_e1_absent),
                })

    # Se ordena por la diferencia de riesgos, no por el odds ratio: la
    # diferencia esta acotada en [-1,1] y no la inflan las celdas vacias.
    conditional_results.sort(key=lambda x: -abs(x["dif_riesgo"]))

    print(f"\n  Top-{n_top_pairs} pares con mayor dependencia condicional:")
    print(f"  {'e1':<24s} {'e2':<24s} {'P(e2|e1=1)':>11s} "
          f"{'P(e2|e1=0)':>11s} {'dif':>7s} {'OR':>8s} {'IC95 OR':>17s} "
          f"{'n1/n0':>8s}")
    print(f"  {'─'*95}")
    for r in conditional_results[:n_top_pairs]:
        e1_str = f"{labels[r['e1'][0]]}→{labels[r['e1'][1]]}"
        e2_str = f"{labels[r['e2'][0]]}→{labels[r['e2'][1]]}"
        ic = r["or_ic95"]
        print(f"  {e1_str:<24s} {e2_str:<24s} {r['p_e2_given_e1']:>11.3f} "
              f"{r['p_e2_given_not_e1']:>11.3f} {r['dif_riesgo']:>+7.2f} "
              f"{r['odds_ratio']:>8.1f} "
              f"[{ic[0]:>6.1f},{ic[1]:>8.1f}] "
              f"{r['n_e1_present']:>3d}/{r['n_e1_absent']:<4d}")

    print(f"\n  Interpretación:")
    print(f"    dif > 0: cuando e1 aparece, e2 aparece más (coordinación)")
    print(f"    dif < 0: cuando e1 aparece, e2 aparece menos (sustitución)")
    print(f"    Sólo se listan los pares con |dif| >= 0.25 y cuyo IC95 del odds")
    print(f"    ratio excluye el 1. El ratio crudo P(e2|e1=1)/P(e2|e1=0) NO se")
    print(f"    reporta: con el denominador en 0 exacto se iba a 1e6 y empataba")
    print(f"    todos los pares del top, dejando el orden al azar.")

    if conditional_results:
        difs = np.array([r["dif_riesgo"] for r in conditional_results])
        n_coordinated = int((difs > 0).sum())
        n_substitutive = int((difs < 0).sum())
        ors = np.array([r["odds_ratio"] for r in conditional_results])
        print(f"\n  Resumen ({len(conditional_results)} pares significativos "
              f"de {len(frequent_edges) * (len(frequent_edges) - 1) // 2} "
              f"comparaciones):")
        print(f"    Coordinación (dif > 0): {n_coordinated}")
        print(f"    Sustitución  (dif < 0): {n_substitutive}")
        print(f"    |dif| mediana {np.median(np.abs(difs)):.3f} · "
              f"máx {np.abs(difs).max():.3f}")
        print(f"    Odds ratio (Haldane) mediana {np.median(ors):.1f} · "
              f"máx {ors.max():.1f}")
        # Benjamini-Hochberg sobre TODAS las comparaciones, no sólo las que
        # pasaron el filtro. Sin esto se reportaban 223 pares "significativos"
        # cuando con 5565 comparaciones y α=0.05 el azar ya produce 278: el
        # análisis no separaba señal de multiplicidad.
        n_comp = len(todos_p)
        esperados = 0.05 * n_comp
        try:
            _, q_all, umbral_bh = benjamini_hochberg_bh(
                np.array(todos_p), 0.05)
            n_bh = int((np.array(todos_p) <= umbral_bh).sum()) if umbral_bh > 0 else 0
        except Exception:
            umbral_bh, n_bh = 0.0, 0
        sobreviven = [r for r in conditional_results
                      if umbral_bh > 0 and r.get("p_wald", 1.0) <= umbral_bh]
        print(f"    Corrección por multiplicidad (BH, FDR=0.05) sobre las "
              f"{n_comp} comparaciones con n >= {COND_N_MIN} en las dos "
              f"ramas:")
        print(f"      umbral efectivo p <= {umbral_bh:.2e} · "
              f"sobreviven {n_bh}")
        print(f"      de los {len(conditional_results)} listados arriba, "
              f"{len(sobreviven)} pasan BH")
        print(f"      (a α=0.05 sin corregir el azar ya daría "
              f"{esperados:.0f} falsos positivos, así que un recuento sin BH "
              f"no es interpretable)")
        if n_bh == 0:
            # No es un resultado nulo, es falta de potencia: con M modelos
            # una tabla 2x2 con la corrección de Haldane no puede bajar de
            # cierto p, y BH sobre miles de tests exige mucho menos.
            p_min_posible = float(min(todos_p)) if todos_p else 1.0
            print(f"      POTENCIA: con M={n_models} modelos el p más "
                  f"pequeño alcanzable fue {p_min_posible:.2e},")
            print(f"      y BH exige p <= {0.05 / max(n_comp, 1):.2e} para "
                  f"la primera. No es un resultado nulo:")
            print(f"      es que 20 semillas no dan para estimar "
                  f"asociaciones 2x2 con esta multiplicidad.")
    else:
        n_coordinated = n_substitutive = 0
        print(f"\n  Ningún par supera |dif| >= 0.25 con IC95 que excluya el 1.")

    def _lim(v):
        """inf no es JSON valido en sentido estricto; se acota."""
        if isinstance(v, (list, tuple)):
            return [float(min(max(x, -1e9), 1e9)) for x in v]
        if isinstance(v, bool):
            return bool(v)
        if isinstance(v, int):
            return int(v)
        v = float(v)
        return float(min(max(v, -1e9), 1e9))

    return {
        "conditional_results": [
            {"e1": [int(r["e1"][0]), int(r["e1"][1])],
             "e2": [int(r["e2"][0]), int(r["e2"][1])],
             "e1_label": f"{labels[r['e1'][0]]}→{labels[r['e1'][1]]}",
             "e2_label": f"{labels[r['e2'][0]]}→{labels[r['e2'][1]]}",
             **{k: _lim(v) for k, v in r.items() if k not in ("e1", "e2")}}
            for r in conditional_results[:30]
        ],
        "n_frequent_edges": int(len(frequent_edges)),
        "n_comparaciones": int(len(frequent_edges) * (len(frequent_edges) - 1) // 2),
        "n_significativos": int(len(conditional_results)),
        "n_coordinated": n_coordinated,
        "n_substitutive": n_substitutive,
        "n_significativos_bh": int(len(sobreviven)) if conditional_results else 0,
        "umbral_bh": float(umbral_bh) if conditional_results else None,
        "falsos_positivos_esperados": float(esperados) if conditional_results else None,
        "criterio": ("|P(e2|e1=1) - P(e2|e1=0)| >= 0.25 y el IC95 del odds "
                     "ratio de Haldane excluye 1; n >= COND_N_MIN en las dos "
                     "ramas. Sin corrección por comparaciones múltiples."),
    }
def main():
    print(f"\n{'═'*82}")
    print(f" PARADIGMA 5 — BASELINES DETERMINISTAS")
    print(f" Pregunta: ¿Supera el framework a la estadística básica?")
    print(f" Fundamento: Pearson, Granger, Erdős-Rényi, densidad igualada, "
          f"deps condicionales")
    print(f"{'═'*82}")

    # find_cached_matrices() devuelve (matrices, seeds), no (matrices, stack)
    # — "stack" aquí era en realidad la lista de semillas (enteros), lo que
    # rompía build_pearson_baseline/build_granger_baseline (esperan un
    # ndarray con .shape). El stack real de índices se carga aparte.
    all_matrices, seeds = find_cached_matrices()
    if not all_matrices:
        print("  ❌ No se encontraron matrices cacheadas.")
        return None
    M = len(all_matrices)
    print(f"  Modelos disponibles: M = {M}")

    stack = np.load(OUTPUT_NPY) if os.path.exists(OUTPUT_NPY) else None
    if stack is None:
        print(f"  ⚠️  No se encontró {OUTPUT_NPY}: los baselines de Pearson/Granger "
              f"(requieren las series originales) se omiten; solo corre Erdős-Rényi.")

    calcular_eta(all_matrices)
    p_top = p_efectivo(None)
    consensus_graph = build_consensus_graph(all_matrices, p_top=p_top,
                                              threshold=TAU_CONSENSUS)
    print(f"\n  Grafo de consenso (Top-P P={p_top}, η={ETA_MARGEN:.6f}, "
          f"τ={TAU_CONSENSUS}): {len(consensus_graph)} aristas")
    for i, j in sorted(consensus_graph):
        print(f"    {INDEX_NAMES[i]:12s} → {INDEX_NAMES[j]:12s}")

    baseline_results = compute_baselines_comparison(
        all_matrices=all_matrices,
        consensus_graph=consensus_graph,
        stack=stack,
        labels=INDEX_NAMES,
        p_top=p_top,
    )

    if stack is not None:
        kmatched_results = compute_kmatched_baselines(
            all_matrices=all_matrices,
            stack=stack,
            labels=INDEX_NAMES,
            p_values=[p_top],
        )
    else:
        print(f"\n  ⚠️ Stack no disponible: saltando baselines de densidad igualada.")
        kmatched_results = {}

    conditional_results = compute_conditional_edge_dependencies(
        all_matrices=all_matrices,
        labels=INDEX_NAMES,
        p_top=p_top,
        n_top_pairs=15,
    )

    results = {
        "M": M,
        "discretizacion": "Top-P",
        "top_p": float(p_top),
        "eta": float(ETA_MARGEN),
        "TAU_CONSENSUS": TAU_CONSENSUS,
        "consensus_size": int(len(consensus_graph)),
        # El extractor (PARTE 11) lee esta clave; antes solo se guardaba el
        # tamaño y nunca la lista de aristas, así que "Grafo consenso (P5)"
        # siempre salía en 0 aunque el consenso tuviera aristas reales.
        "consensus_edges": [[int(i), int(j)] for i, j in sorted(consensus_graph)],
        "baseline_comparison": baseline_results,
        "kmatched": {str(k): v for k, v in kmatched_results.items()},
        "conditional_dependencies": conditional_results,
    }
    with open(os.path.join(DIR_P5, "paradigm5_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  ✅ Resultados: {DIR_P5}/paradigm5_results.json")

    print(f"\n{'═'*82}")
    print(f" RESUMEN — PARADIGMA 5 (BASELINES DETERMINISTAS)")
    print(f"{'═'*82}")
    print(f"  • Jaccard(G_consensus, G_Pearson) = {baseline_results['jaccard_pearson']:.4f} "
          f"→ {'No lineal' if baseline_results['jaccard_pearson'] < 0.3 else 'Mixto' if baseline_results['jaccard_pearson'] < 0.7 else 'Solo lineal'}")
    print(f"  • Jaccard(G_consensus, G_Granger) = {baseline_results['jaccard_granger']:.4f} "
          f"→ {'No temporal' if baseline_results['jaccard_granger'] < 0.3 else 'Mixto' if baseline_results['jaccard_granger'] < 0.5 else 'Temporal'}")
    print(f"  • Z-score de TAMAÑO vs Erdős-Rényi = {baseline_results['z_score_size']:.2f}")
    print(f"    Con densidad igualada debe salir ~0: comprueba que el nulo está")
    print(f"    bien construido, NO mide estructura. El contraste estructural lo")
    print(f"    da el Z(modularidad) del Paradigma 1 (§21.3).")
    # Antes esto leía kmatched_results[5] (la clave del viejo Top-K con K=5);
    # tras migrar a Top-P la clave es el valor de P, así que el bloque imprimía
    # la cabecera y ningún número. Ahora recorre las claves reales.
    if kmatched_results:
        print(f"  • Densidad igualada (Top-P):")
        for clave in sorted(kmatched_results):
            r = kmatched_results[clave] or {}
            if not r:
                continue
            print(f"      P = {float(clave):.2f}")
            print(f"        J(attn,attn)    = {r.get('j_attn_attn', 0):.4f}"
                  f"   ← estabilidad interna")
            print(f"        J(attn,Pearson) = {r.get('j_attn_pearson', 0):.4f}"
                  f"   Ratio A/P = {r.get('ratio_attn_vs_pearson', 0):.2f}x")
            print(f"        J(attn,Granger) = {r.get('j_attn_granger', 0):.4f}"
                  f"   Ratio A/G = {r.get('ratio_attn_vs_granger', 0):.2f}x")
            peor = min(r.get('ratio_attn_vs_pearson', 0.0),
                       r.get('ratio_attn_vs_granger', 0.0))
            if peor < 1.0:
                print(f"        [!] Ratio < 1: el grafo de atención se parece MÁS a un")
                print(f"            baseline determinista que a otra realización de sí")
                print(f"            mismo. A esa densidad no aporta sobre Pearson/Granger.")
    print(f"{'═'*82}\n")

    return baseline_results

if __name__ == "__main__" and EJECUTAR_PARTE10:
    main()

# ============================================================================
# TRIANGULACIÓN — 5 PARADIGMAS (Matriz de Diagnóstico Multidimensional, §22)
#
# Implementa el Algoritmo 2 del informe con las 5 coordenadas Se, Fe, Te, Ie, Be.
# Limitaciones honestas frente al informe (documentadas para no sobre-reportar):
#   - RobK(e) (robustez del Top-K entre valores de K) no está implementado:
#     Se se reduce a Φ(e) (o significancia BH como vía alterna).
#   - Specdir(e) (especificidad direccional vía edge-swapping con ΔMSE real)
#     no está implementado: Fe se reduce a AUCdegrad(e) de la ablación suave.
#   - Ie usa el proxy I(arista∈Top-K; val_loss) de Paradigma 4, NO la fórmula
#     exacta I(G;Y|X)/H(Y) del informe (esa requiere Y de un set de test
#     independiente y condicionar en X, que no están disponibles aquí). Si
#     las 50 semillas no tienen val_loss guardado (corridas previas a este
#     fix), Ie queda marcada explícitamente como "no evaluable".
#   - Be = 1 - Jaccard(G_consensus, G_Pearson) SÍ es literal al informe,
#     tomado de Paradigma 5.
# ============================================================================

@dataclass
class EdgeProfile5D:
    i: int; j: int
    label_i: str; label_j: str
    phi: int; p_obs: float; p_value: float; bh_sig: bool
    auc_degradation: float
    ie_proxy: Optional[float]
    se_pass: bool; fe_pass: bool; te_pass: bool
    ie_pass: Optional[bool]; be_diff: bool
    # ΔMSE sólo se mide sobre las aristas del grafo Top-P de A_mean (74 de
    # 132 el 31/08). Las otras 58 entraban a "Ruido Estocástico" con
    # fe_pass=False, y eso mezcla "medido y no degrada" con "nunca medido".
    fe_evaluado: bool = True
    delta_mse: Optional[float] = None
    # Los tres instrumentos que sobrevivieron a la prueba de independencia.
    # Re y Ne son por arista; Se es global y actúa como compuerta.
    re_pass: Optional[bool] = None
    ne_pass: Optional[bool] = None
    perfil_3d: str = ""
    profile: str = "UNCLASSIFIED"
    # Auditoría e-BH (LINEAMIENTO_NUEVOS_PARADIGMAS.md §4): guardado al lado
    # de re_pass, no lo reemplaza todavía en perfil_3d.
    e_value: Optional[float] = None
    re_pass_ebh: Optional[bool] = None
    # Te por arista, instrumento inspirado en KHAN (ver aviso de fidelidad en
    # topologia_persistente_edge_pvalues). Tampoco decide perfil_3d todavía —
    # falta correr matriz_independencia() con esto agregado antes de confiar
    # en que es una dimensión genuinamente distinta de Re/Ne/Se.
    te_pass_khan: Optional[bool] = None
    te_p_valor_khan: Optional[float] = None
    te_brecha: Optional[float] = None

def load_paradigm_results_5d() -> Dict:
    paths = {
        "p1": [os.path.join(DIR_P1, "paradigm1_results.json")],
        "p2": [os.path.join(DIR_P2, "paradigm2_results.json")],
        "p3": [os.path.join(DIR_P3, "paradigm3_results.json")],
        "p4": [os.path.join(DIR_P4, "paradigm4_results.json")],
        "p5": [os.path.join(DIR_P5, "paradigm5_results.json")],
    }
    data = {}
    for k, candidatas in paths.items():
        data[k] = None
        for p in candidatas:
            if not os.path.exists(p):
                continue
            with open(p, "r") as f:
                data[k] = json.load(f)
            print(f"  ✅ Cargado: {p}")
            break
        if data[k] is None:
            print(f"  ⚠️  No encontrado: {candidatas[0]}")
    return data


# ============================================================================
# INSTRUMENTOS INDEPENDIENTES  (refinamiento de los 5 paradigmas)
# ============================================================================
#
# El informe pone como criterio central: "si todas las métricas miden la misma
# dimensión latente, su concordancia no aporta evidencia adicional". Ese
# criterio nunca se había COMPROBADO. Medido el 31/08 sobre las 132 aristas,
# correlacionando el puntaje por arista de cada paradigma (Spearman):
#
#                    P1 Phi   P1 -log p   P2 AUC   P4 MI   P5/grafo
#   P1 Phi            1.000       0.995    0.598   0.194      0.800
#   P1 -log p         0.995       1.000    0.631   0.193      0.800
#   P2 AUC ent        0.598       0.631    1.000  -0.025      0.585
#   P4 MI(e;L)        0.194       0.193   -0.025   1.000     -0.172
#   P5/grafo          0.800       0.800    0.585  -0.172      1.000
#
# Lecturas, y qué se hace con cada una:
#
#   Phi(e) y su p-valor: rho = 0.995. Son la MISMA cantidad. En el veredicto del
#   paradigma 1 aparecían como dos dimensiones separadas ("Significancia" y
#   "Consenso"), o sea contadas dos veces.
#
#   P1 y el grafo de consenso del P5: rho = 0.800. El consenso se CONSTRUYE con
#   Phi >= 0.8, así que la concordancia es una tautología, no evidencia.
#
#   P3 no aparece: no produce ningún puntaje por arista, sólo escalares
#   globales, y su GED cumple GED = 1 - Jaccard exacto, que es a su vez el
#   J(attn,attn) del P5. Tres números que son uno.
#
#   P4 es el único no correlacionado con nada (|rho| <= 0.19). Pero no lo es por
#   medir una dimensión nueva: la MI se estima con M=20 muestras, y en esta
#   corrida I(G;L) salió 0.000000 exacto con H(L|G) = H(L). Ruido independiente
#   sigue siendo ruido. No entra en la clasificación.
#
# Quedan TRES dimensiones que sí son independientes entre sí, y que se
# corresponden con las tres transformaciones del propio marco (Phi, Psi, F):
#
#   Re · REPRODUCIBILIDAD  (Psi)  la arista reaparece entre semillas por encima
#                                 del nulo de perfil de columna, no del nulo
#                                 uniforme, que ignora los sumideros.
#   Ne · NECESIDAD          (F)   quitarla degrada el modelo MÁS que quitar una
#                                 arista que el operador ya había descartado.
#   Se · SUFICIENCIA              el grafo aporta sobre Pearson a densidad
#                                 igualada: J(attn,attn) > J(attn,Pearson).
#
# Se es global (una propiedad del grafo, no de cada arista), así que actúa como
# compuerta: si el grafo no supera a Pearson, ninguna arista puede reclamar
# relación estructural por mucho que sea reproducible y necesaria.

def matriz_independencia(series: Dict[str, Dict[Tuple[int, int], float]],
                         umbral_redundante: float = 0.7,
                         umbral_independiente: float = 0.3) -> Dict:
    """Spearman entre los puntajes por arista de cada instrumento.

    Es la comprobación que el informe pide y que faltaba: dos instrumentos con
    |rho| alto miden la misma dimensión latente y su concordancia no es
    evidencia adicional, es la misma evidencia contada dos veces.
    """
    nombres = [n for n, d in series.items() if len(d) >= 30]
    n = len(nombres)
    M = np.full((n, n), np.nan)
    for a in range(n):
        for b in range(n):
            comunes = sorted(set(series[nombres[a]]) & set(series[nombres[b]]))
            if len(comunes) < 30:
                continue
            x = np.array([series[nombres[a]][k] for k in comunes], float)
            y = np.array([series[nombres[b]][k] for k in comunes], float)
            if x.std() < 1e-12 or y.std() < 1e-12:
                continue
            M[a, b] = float(stats.spearmanr(x, y).statistic)

    redundantes, independientes = [], []
    for a in range(n):
        for b in range(a + 1, n):
            r = M[a, b]
            if not np.isfinite(r):
                continue
            if abs(r) > umbral_redundante:
                redundantes.append((nombres[a], nombres[b], float(r)))
            elif abs(r) < umbral_independiente:
                independientes.append((nombres[a], nombres[b], float(r)))
    return dict(nombres=nombres, matriz=M.tolist(),
                redundantes=redundantes, independientes=independientes,
                umbral_redundante=umbral_redundante,
                umbral_independiente=umbral_independiente)


def reportar_independencia(res: Dict) -> None:
    if not res or not res.get("nombres"):
        return
    nombres, M = res["nombres"], np.array(res["matriz"], float)
    print("\n" + "=" * 88)
    print(" INDEPENDENCIA ENTRE INSTRUMENTOS (Spearman de los puntajes por arista)")
    print(" Criterio del informe: si dos métricas miden la misma dimensión")
    print(" latente, su concordancia NO aporta evidencia adicional.")
    print("=" * 88)
    print(" " * 16 + "".join(f"{n[:12]:>14}" for n in nombres))
    for a, na in enumerate(nombres):
        fila = f"  {na[:14]:<14}"
        for b in range(len(nombres)):
            fila += (f"{M[a, b]:>14.3f}" if np.isfinite(M[a, b])
                     else f"{'-':>14}")
        print(fila)
    if res["redundantes"]:
        print(f"\n  REDUNDANTES (|rho| > {res['umbral_redundante']}): "
              f"una de las dos sobra")
        for a, b, r in res["redundantes"]:
            print(f"     {a:<16} y {b:<16} rho = {r:+.3f}")
    if res["independientes"]:
        print(f"\n  INDEPENDIENTES (|rho| < {res['umbral_independiente']}): "
              f"aportan evidencia distinta")
        for a, b, r in res["independientes"]:
            print(f"     {a:<16} y {b:<16} rho = {r:+.3f}")
    print(f"\n  CUIDADO: independiente no es lo mismo que informativo. Una")
    print(f"  métrica de puro ruido tampoco correlaciona con nada. Antes de")
    print(f"  contar un instrumento como dimensión nueva hay que comprobar que")
    print(f"  supera a su propio nulo (eso lo hacen Re con el nulo de columna y")
    print(f"  Ne con las no-aristas de control).")
    print("=" * 88)


def clasificar_edge_3d(re_pass: bool, ne_pass: bool,
                       se_global: bool) -> str:
    """Clasificación con las TRES dimensiones que resultaron independientes.

    Sustituye a classify_edge_5d, que usaba cinco de las cuales Ie era ruido
    (M=20) y Te venía de un paradigma cuyos escalares son 1 - Jaccard. Aquí no
    hay dimensión que no haya pasado la prueba de independencia.

      Re  reproducible por encima del nulo de perfil de columna
      Ne  necesaria: su ablación degrada más que la de una no-arista
      Se  global: el grafo supera a Pearson a densidad igualada
    """
    if not se_global:
        # Sin superar a Pearson, ninguna arista puede reclamar estructura: lo
        # que haya lo explica la correlación lineal.
        return ("Explicable por Pearson" if (re_pass or ne_pass)
                else "Ruido Estocástico")
    if re_pass and ne_pass:
        return "Relación Estructural"
    if re_pass and not ne_pass:
        return "Estable pero Prescindible"
    if ne_pass and not re_pass:
        return "Necesaria pero Inestable"
    return "Ruido Estocástico"


def classify_edge_5d(se: bool, fe: bool, te: bool,
                      ie: Optional[bool], be_diff: bool) -> str:
    """Algoritmo 2 del informe (§22.3), literal salvo Ie cuando no es evaluable
    (se trata como 'desconocida': no se usa para forzar Correlación Lineal
    Trivial ni Estructura Latente Oculta, ambas rutas que la necesitan)."""
    if not se:
        return "Relación Contextual" if fe else "Ruido Estocástico"
    # Se == True (arista estable)
    if not fe:
        if te and ie is False:
            return "Artefacto / Attention Sink"
        if te and ie and not be_diff:
            return "Correlación Lineal Trivial"
        return "Estable pero Irrelevante"
    # Fe == True (arista fiel y necesaria)
    if te and ie and be_diff:
        return "Relación Estructural Fuerte"
    if te and ie and not be_diff:
        return "Estructura Latente Oculta"
    return "Atajo Predictivo / Shortcut"

def build_diagnosis_matrix_5d(data: Dict) -> List[EdgeProfile5D]:
    p1_data = data.get("p1") or {}
    p2_data = data.get("p2") or {}
    p3_data = data.get("p3") or {}
    p4_data = data.get("p4") or {}
    p5_data = data.get("p5") or {}

    p1_edges = {(e["i"], e["j"]): e for e in p1_data.get("edges", [])}
    p2_edges = {(e["i"], e["j"]): e for e in p2_data.get("ablation", [])}
    # Te por arista (LINEAMIENTO_NUEVOS_PARADIGMAS.md §1): no dirigido en
    # origen (i<j en el JSON), se espeja a ambas direcciones acá.
    _topo_raw = (p3_data.get("topologia_persistente") or {}).get("edges", [])
    p3b_edges = {}
    for e in _topo_raw:
        p3b_edges[(e["i"], e["j"])] = e
        p3b_edges[(e["j"], e["i"])] = e

    # --- Te (Topología): ρrole global + GEDnorm global + estabilidad de rol del nodo ---
    p3_summary = p3_data.get("summary", {})
    rho_role = p3_summary.get("role_stability_spearman_mean", 0.0)
    ged_mean = p3_summary.get("ged_mean", 1.0)
    node_role_summary = p3_data.get("node_role_summary", {})

    # Antes: umbral fijo P(rol) > max(0.50, 2/n_roles), con n_roles=4
    # (hub/source/sink/isolated) codificado a mano. Dos problemas: (a)
    # classify_node_roles() devuelve 5 categorías, no 4 (falta "bridge", y
    # "sink" no existe) — el propio n_roles estaba mal; (b) el umbral se
    # fijó DESPUÉS de ver que 0.7 dejaba a los 12 nodos fuera, que es ajustar
    # el umbral al dato. Ahora node_role_summary trae, por nodo, un p-valor
    # calibrado contra un nulo de permutación (ver
    # calibrar_estabilidad_rol_nulo en el paradigma 3): estable si p<0.05,
    # decidido antes de mirar si el nodo pasa. Fallback al umbral fijo solo
    # para leer JSON generados antes de este cambio.
    ROLE_STABLE_THRESH_FALLBACK = 0.50

    def node_stable(node_name: str) -> bool:
        info = node_role_summary.get(node_name, {})
        if "p_valor_nulo" in info:
            return bool(info.get("estable_calibrado", False))
        return info.get("prob", 0.0) > ROLE_STABLE_THRESH_FALLBACK

    # Te global: se prefiere la calibración contra modelo nulo de la FASE 1
    # sobre el umbral fijo ρrole>0.9 de la Tabla 7.
    #
    # POR QUÉ. Con el umbral fijo, te_global_pass era False (ρrole=0.287) y como
    # multiplicaba a te_pass de CADA arista, vetaba las 132 de golpe: ninguna
    # podía alcanzar "Relación Estructural Fuerte" por mucho que pasara los
    # otros cuatro paradigmas. Un solo escalar global decidía por todas.
    # La calibración pregunta lo correcto: ¿la estabilidad observada supera a la
    # de un nulo que conserva los sumideros y borra la estructura por fila?
    # Medido: Spearman 0.404 contra 0.078 del nulo, 5.2x.
    cal_z = cal_p = cal_raz = None
    ruta_cal = os.path.join(DIR_FASE1, "calibracion_nulo.json")
    if os.path.exists(ruta_cal):
        try:
            with open(ruta_cal, encoding="utf-8") as f:
                _cal = json.load(f)
            _r = _cal.get("solo_columnas", {}).get("rho_mean", {})
            cal_z = _r.get("z")
            cal_p = _r.get("p_empirico")
            cal_raz = _r.get("razon")
        except Exception:
            cal_z = None

    # La compuerta va por el p EMPÍRICO cuando está disponible. El Z de la
    # calibración es de dos cifras (+17 el 31/08) sólo porque la métrica es
    # un promedio de 120 pares × 12 filas y eso encoge la σ del nulo: usarlo
    # como umbral es tan arbitrario como el ρ>0.9 de la Tabla 7 que sustituye.
    # El p empírico cuenta excedencias sobre 40 nulos, así que su suelo real
    # es 1/41 = 0.024 y no se puede inflar promediando.
    if cal_p is not None:
        te_global_pass = bool(cal_p < 0.05)
        te_origen = (f"calibración vs nulo (p={cal_p:.3f}"
                     + (f", {cal_raz:.1f}x" if cal_raz else "") + ")")
    elif cal_z is not None:
        te_global_pass = bool(cal_z > 2.0)
        te_origen = f"calibración vs nulo (Z={cal_z:+.1f}, sin p empírico)"
    else:
        te_global_pass = (rho_role > 0.9) and (ged_mean < 0.2)
        te_origen = "umbral fijo Tabla 7 (sin calibración disponible)"

    # --- Ie (Información): proxy MI(arista; val_loss) de Paradigma 4 ---
    mi_per_edge_raw = p4_data.get("mi_per_edge", {}) or {}
    mi_per_edge = {}
    for key, v in mi_per_edge_raw.items():
        try:
            i_str, j_str = key.split("_")
            mi_per_edge[(int(i_str), int(j_str))] = float(v)
        except (ValueError, AttributeError):
            continue
    ie_disponible = bool(mi_per_edge) and p4_data.get("igl_edge_mean", 0.0) > 0
    ie_median = float(np.median(list(mi_per_edge.values()))) if mi_per_edge else 0.0

    # --- Be (Baseline): 1 - Jaccard(G_consensus, G_Pearson), literal al informe ---
    j_pearson = p5_data.get("baseline_comparison", {}).get("jaccard_pearson")
    be_disponible = j_pearson is not None
    be_diff_global = (j_pearson is not None) and (j_pearson < 0.5)

    # Fidelidad (Fe): se usa el ΔMSE REAL del §8.2 si está disponible. El
    # "auc_degradation" de la ablación es el cambio de ENTROPÍA de la matriz,
    # una propiedad algebraica que nunca ejecuta el modelo, y compararlo contra
    # el umbral 0.005 de la Tabla 7 —definido para la curva de MSE de
    # predicción— es un error de categoría. Por eso Fe daba 0 siempre.
    dmse_raw = (p2_data.get("delta_mse") or {}).get("aristas") or []
    dmse_edges = {(int(r["i"]), int(r["j"])): r for r in dmse_raw}
    if dmse_edges:
        vals = np.array([abs(r["delta_mse_alpha0"]) for r in dmse_edges.values()])
        # Umbral relativo: una arista es fiel si su |ΔMSE| está en el tercio
        # superior de lo observado Y por encima del umbral de fallo del informe.
        # Umbral de fidelidad. Se prefiere el calibrado contra no-aristas
        # (percentil 95 de las que el operador descartó) sobre el percentil 66
        # de las propias aristas, que es circular: define "importante" como
        # "por encima de la mediana de sí mismas", así que siempre aprueba a un
        # tercio por construcción, hasta con ΔMSE de ruido.
        umbral_cal = ((p2_data or {}).get("delta_mse", {})
                      .get("calibracion", {}).get("umbral_nulo"))
        if umbral_cal is not None:
            umbral_fe = float(umbral_cal)
            fe_origen = (f"ΔMSE real (§8.2) calibrado: percentil "
                         f"{DELTA_MSE_PCTL_NULO:.0f} de las no-aristas "
                         f"= {umbral_fe:+.5f}")
        else:
            umbral_fe = max(0.001, float(np.percentile(vals, 66)))
            fe_origen = (f"ΔMSE real (§8.2), umbral {umbral_fe:.5f} "
                         f"= max(0.001, percentil 66) — CIRCULAR, "
                         f"corre el paradigma 2 con no-aristas de control")
    else:
        umbral_fe = None
        fe_origen = "AUC de entropía (proxy; ΔMSE no disponible)"

    # --- Se (Suficiencia): compuerta GLOBAL del grafo -------------------
    # El grafo tiene que parecerse más a otra realización de sí mismo que a
    # Pearson, a la MISMA densidad. Si no, lo que contiene lo explica la
    # correlación lineal y ninguna arista puede reclamar estructura propia.
    # Medido: A/P = 0.68x en Aconcagua, 0.93x en Pencahue con k=1, 1.10x con
    # k=3. Sólo el último supera el 1.
    _km = (p5_data.get("kmatched") or {})
    _km0 = next(iter(_km.values()), {}) if _km else {}
    ratio_ap = _km0.get("ratio_attn_vs_pearson")
    se_global_3d = bool(ratio_ap is not None and ratio_ap > 1.0)

    profiles: List[EdgeProfile5D] = []
    for (i, j), p1e in p1_edges.items():
        p2e = p2_edges.get((i, j), {})
        auc = p2e.get("auc_degradation", 0.0)
        label_i, label_j = INDEX_NAMES[i], INDEX_NAMES[j]

        se_pass = p1e.get("bh_sig", False) or (p1e.get("p_obs", 0.0) >= 0.8)
        r_d = dmse_edges.get((i, j))
        if umbral_fe is not None:
            fe_evaluado = r_d is not None
            fe_pass = bool(fe_evaluado
                           and abs(r_d["delta_mse_alpha0"]) >= umbral_fe)
        else:
            fe_evaluado = True
            fe_pass = auc > 0.005
        dmse_val = (float(r_d["delta_mse_alpha0"]) if r_d is not None
                    else None)
        te_pass = te_global_pass and (node_stable(label_i) or node_stable(label_j))

        ie_proxy = mi_per_edge.get((i, j))
        if not ie_disponible or ie_proxy is None:
            ie_pass = None
        else:
            ie_pass = ie_proxy > ie_median

        be_diff = be_diff_global if be_disponible else False

        profile = classify_edge_5d(se_pass, fe_pass, te_pass, ie_pass, be_diff)

        # --- Los tres instrumentos independientes -----------------------
        # Re: reproducible por encima del nulo. Se usa la significancia BH del
        # paradigma 1, que es la única forma de reproducibilidad calibrada que
        # hay; Phi >= 0.8 a secas no vale porque con densidad 50% el azar ya da
        # coincidencias altas.
        re_pass = bool(p1e.get("bh_sig", False))
        # Re (e-BH): mismo rol que re_pass pero con e-valor secuencial en vez
        # de p-valor binomial de M fijo (LINEAMIENTO_NUEVOS_PARADIGMAS.md §4).
        # Se calcula y se guarda, pero NO reemplaza a re_pass como criterio de
        # perfil_3d todavía: primero hay que auditar cuánto difieren en una
        # corrida real (ver el conteo que imprime exportar_paradigm1_full_json)
        # antes de conmutar cuál decide.
        re_pass_ebh = (bool(p1e["e_bh_sig"]) if p1e.get("e_bh_sig") is not None
                      else None)
        p3be = p3b_edges.get((i, j), {})
        te_pass_khan = p3be.get("te_sig")
        # Ne: su ablación degrada MÁS que el percentil 95 de las no-aristas de
        # control. Sin ΔMSE medido no se puede afirmar, así que queda en None y
        # la arista no puede reclamar necesidad.
        ne_pass = (None if not fe_evaluado else bool(fe_pass))
        perfil_3d = clasificar_edge_3d(re_pass, bool(ne_pass), se_global_3d)

        profiles.append(EdgeProfile5D(
            i=i, j=j, label_i=label_i, label_j=label_j,
            phi=p1e.get("phi", 0), p_obs=p1e.get("p_obs", 0.0),
            p_value=p1e.get("p_value", 1.0), bh_sig=p1e.get("bh_sig", False),
            auc_degradation=auc, ie_proxy=ie_proxy,
            se_pass=se_pass, fe_pass=fe_pass, te_pass=te_pass,
            ie_pass=ie_pass, be_diff=be_diff, profile=profile,
            fe_evaluado=fe_evaluado, delta_mse=dmse_val,
            re_pass=re_pass, ne_pass=ne_pass, perfil_3d=perfil_3d,
            e_value=p1e.get("e_value"), re_pass_ebh=re_pass_ebh,
            te_pass_khan=te_pass_khan, te_p_valor_khan=p3be.get("p_valor"),
            te_brecha=p3be.get("brecha"),
        ))
    return profiles, {
        "rho_role": rho_role, "ged_mean": ged_mean, "te_global_pass": te_global_pass,
        "te_origen": te_origen, "te_z_calibrado": cal_z,
        "te_p_empirico": cal_p, "te_razon_vs_nulo": cal_raz,
        "role_stable_metodo": ("p<0.05 vs nulo de permutacion (calibrado) "
                               "o fallback fijo 0.50 para JSON antiguos"),
        "n_nodos_estables": int(sum(
            1 for n in INDEX_NAMES if node_stable(n))),
        "fe_origen": fe_origen, "fe_umbral": umbral_fe,
        "se_global_3d": se_global_3d, "ratio_attn_vs_pearson": ratio_ap,
        "ie_disponible": ie_disponible, "jaccard_pearson": j_pearson,
        "be_disponible": be_disponible,
    }

PROFILE_COLORS_5D = {
    "Relación Estructural Fuerte": "#2ecc71",
    "Estructura Latente Oculta": "#27ae60",
    "Atajo Predictivo / Shortcut": "#f39c12",
    "Correlación Lineal Trivial": "#3498db",
    "Artefacto / Attention Sink": "#9b59b6",
    "Estable pero Irrelevante": "#7f8c8d",
    "Relación Contextual": "#e67e22",
    "Ruido Estocástico": "#95a5a6",
}

def plot_diagnosis_matrix_5d(profiles: List[EdgeProfile5D], output_path: str):
    M_color = np.zeros((NUM_INDICES, NUM_INDICES, 3))
    for p in profiles:
        color = PROFILE_COLORS_5D.get(p.profile, "#bdc3c7").lstrip("#")
        rgb = tuple(int(color[k:k+2], 16) / 255.0 for k in (0, 2, 4))
        M_color[p.i, p.j] = rgb
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(M_color, aspect="equal")
    ax.set_xticks(range(NUM_INDICES)); ax.set_yticks(range(NUM_INDICES))
    ax.set_xticklabels(INDEX_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(INDEX_NAMES, fontsize=9)
    ax.set_title("Matriz de Diagnóstico Multidimensional (5 paradigmas)\n"
                "Color = perfil técnico (Tabla 6 del informe)",
                fontsize=12, fontweight="bold")
    for p in profiles:
        short = p.profile[:8]
        ax.text(p.j, p.i, short, ha="center", va="center",
                fontsize=6, color="white", fontweight="bold")
    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor=c, edgecolor="white", label=k)
                     for k, c in PROFILE_COLORS_5D.items()
                     if any(p.profile == k for p in profiles)]
    ax.legend(handles=legend_elems, loc="upper right",
              bbox_to_anchor=(1.5, 1.0), fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight"); plt.close()

def plot_profile_distribution_5d(profiles: List[EdgeProfile5D], output_path: str):
    counts = {}
    for p in profiles:
        counts[p.profile] = counts.get(p.profile, 0) + 1
    labels = list(counts.keys())
    sizes = list(counts.values())
    colors = [PROFILE_COLORS_5D.get(l, "#bdc3c7") for l in labels]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
           startangle=90, textprops={"fontsize": 8})
    ax.set_title("Distribución de perfiles técnicos\n(triangulación de 5 paradigmas)",
                fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight"); plt.close()

def main():
    print(f"\n{'═'*82}")
    print(f" TRIANGULACIÓN DE EVIDENCIA — 5 PARADIGMAS")
    print(f" Construcción de la Matriz de Diagnóstico Multidimensional (§22)")
    print(f"{'═'*82}")

    data = load_paradigm_results_5d()
    if data.get("p1") is None:
        print("  ❌ No se encontraron resultados del Paradigma 1. Corre PARTE 3 primero.")
        return

    profiles, contexto = build_diagnosis_matrix_5d(data)
    print(f"\n  ✅ {len(profiles)} aristas clasificadas en perfiles técnicos")

    print(f"\n  CONTEXTO GLOBAL (aplicado igual a todas las aristas):")
    print(f"    ρrole (Spearman, P3):        {contexto['rho_role']:.3f}  "
          f"({'aprueba' if contexto['rho_role'] > 0.9 else 'falla'} >0.9)")
    print(f"    GEDnorm medio (P3):          {contexto['ged_mean']:.3f}  "
          f"({'aprueba' if contexto['ged_mean'] < 0.2 else 'falla'} <0.2)")
    print(f"    Te global (Topología):       "
          f"{'APRUEBA' if contexto['te_global_pass'] else 'FALLA'}"
          f"   [{contexto.get('te_origen', 'n/d')}]")
    print(f"    Umbral de rol estable:       {contexto.get('role_stable_thresh', 0.7):.2f}"
          f"   (2x el azar con 4 roles)")
    print(f"    Fe (Fidelidad):              {contexto.get('fe_origen', 'n/d')}")
    if contexto["ie_disponible"]:
        print(f"    Ie (Información, P4):        disponible (proxy I(arista; val_loss))")
    else:
        print(f"    Ie (Información, P4):        NO EVALUABLE — faltan val_loss guardados "
              f"(reentrena con forzar_reentrenamiento=True para habilitarlo)")
    if contexto["be_disponible"]:
        print(f"    Jaccard(G, G_Pearson) (P5):  {contexto['jaccard_pearson']:.3f}  "
              f"→ Be {'Diferente' if contexto['jaccard_pearson'] < 0.5 else 'Idéntico'} de Pearson")
    else:
        print(f"    Be (Baseline, P5):           NO EVALUABLE — corre PARTE 10 (Paradigma 5)")

    counts = {}
    for p in profiles:
        counts[p.profile] = counts.get(p.profile, 0) + 1

    # ---------------------------------------------------------------------
    # PRUEBA DE INDEPENDENCIA + CLASIFICACIÓN CON LOS 3 INSTRUMENTOS
    # ---------------------------------------------------------------------
    try:
        _ser = {
            "Re Phi(e)": {(p.i, p.j): float(p.p_obs) for p in profiles},
            "Re -log p": {(p.i, p.j): float(-np.log10(max(p.p_value, 1e-300)))
                          for p in profiles},
            "Ne dMSE": {(p.i, p.j): float(p.delta_mse) for p in profiles
                        if p.delta_mse is not None},
            "P2 AUC ent": {(p.i, p.j): float(p.auc_degradation)
                           for p in profiles},
            "P4 MI(e;L)": {(p.i, p.j): float(p.ie_proxy) for p in profiles
                           if p.ie_proxy is not None},
            # Instrumentos nuevos (LINEAMIENTO_NUEVOS_PARADIGMAS.md): se
            # agregan a la MISMA prueba de independencia que ya descartó P4 —
            # ninguno se asume independiente, se mide antes de confiar.
            "P1 e-valor": {(p.i, p.j): float(np.log1p(p.e_value))
                           for p in profiles if p.e_value is not None},
            # La BRECHA es el estadistico (continuo, 56 valores distintos
            # de 66); el p-valor depende ademas de que columna toca el par
            # bajo el nulo de perfil de columna, asi que se miden los dos.
            "Te brecha": {(p.i, p.j): float(-p.te_brecha)
                          for p in profiles if p.te_brecha is not None},
            "Te -log p": {(p.i, p.j): float(-np.log10(
                              max(p.te_p_valor_khan, 1e-300)))
                          for p in profiles if p.te_p_valor_khan is not None},
        }
        _ind = matriz_independencia(_ser)
        reportar_independencia(_ind)
    except Exception as _e:
        print(f"  Prueba de independencia saltada: {type(_e).__name__}: {_e}")
        _ind = {}

    _c3 = {}
    for p in profiles:
        _c3[p.perfil_3d] = _c3.get(p.perfil_3d, 0) + 1
    print(f"\n  CLASIFICACIÓN CON LOS 3 INSTRUMENTOS INDEPENDIENTES")
    print(f"    Re reproducibilidad (BH sobre el nulo) · "
          f"Ne necesidad (ΔMSE vs no-aristas) · Se suficiencia (vs Pearson)")
    _sg = contexto.get("se_global_3d")
    _ra = contexto.get("ratio_attn_vs_pearson")
    print(f"    Se global: {'SUPERA a Pearson' if _sg else 'NO supera a Pearson'}"
          + (f"  (A/P = {_ra:.2f}x)" if isinstance(_ra, float) else ""))
    if not _sg:
        print(f"    Con Se en falso ninguna arista puede reclamar relación")
        print(f"    estructural: lo que haya lo explica la correlación lineal.")
    for _k, _v in sorted(_c3.items(), key=lambda x: -x[1]):
        print(f"      {_k:<34} {_v:>4}  ({100 * _v / max(len(profiles), 1):.1f}%)")
    _fuertes3 = [p for p in profiles if p.perfil_3d == "Relación Estructural"]
    if _fuertes3:
        print(f"\n    Aristas con Re y Ne y Se:")
        for p in sorted(_fuertes3, key=lambda x: -(x.delta_mse or 0))[:15]:
            print(f"      {p.label_i:>12} -> {p.label_j:<12} "
                  f"Phi={p.p_obs:.2f}  dMSE={p.delta_mse:+.6f}")

    print(f"\n  DISTRIBUCIÓN DE PERFILES:")
    print(f"    {'Perfil':32s} {'Cuenta':>8s} {'%':>8s}  Rango esperado (informe §22.5)")
    print(f"    {'─'*80}")
    rangos_esperados = {
        "Relación Estructural Fuerte": "3-10%",
        "Correlación Lineal Trivial": "0-5%",
        "Atajo Predictivo / Shortcut": "0-3%",
        "Artefacto / Attention Sink": "0-5%",
        "Ruido Estocástico": "70-85%",
        "Estructura Latente Oculta": "3-8%",
        "Relación Contextual": "0-5%",
    }
    total = len(profiles)
    for prof, n in sorted(counts.items(), key=lambda x: -x[1]):
        pct = 100 * n / total
        rango = rangos_esperados.get(prof, "n/a")
        print(f"    {prof:32s} {n:>8d} {pct:>7.1f}%  {rango}")

    # ΔMSE se mide sólo sobre el grafo Top-P de A_mean. Las aristas de
    # fuera entran con fe_pass=False y caen en "Ruido Estocástico" sin
    # haberse medido nunca. Decir "85 aristas son ruido" cuando 58 no se
    # evaluaron no se sostiene en una defensa, así que se desglosa.
    n_sin_eval = sum(1 for p in profiles if not p.fe_evaluado)
    if n_sin_eval:
        ruido = [p for p in profiles if p.profile == "Ruido Estocástico"]
        r_medidas = sum(1 for p in ruido if p.fe_evaluado)
        r_sin = len(ruido) - r_medidas
        print(f"    {'─'*80}")
        print(f"    ΔMSE medido en {total - n_sin_eval} de {total} aristas "
              f"(sólo las del grafo Top-P de A_mean).")
        print(f"    De las {len(ruido)} en 'Ruido Estocástico': "
              f"{r_medidas} medidas y sin degradación, "
              f"{r_sin} NUNCA evaluadas (fuera del grafo).")
        print(f"    En la tesis: '{r_medidas} de {total - n_sin_eval} "
              f"aristas evaluadas no degradan', no '{len(ruido)} son ruido'.")

    fuertes = [p for p in profiles if p.profile == "Relación Estructural Fuerte"]
    if fuertes:
        print(f"\n  🌟 ARISTAS 'RELACIÓN ESTRUCTURAL FUERTE' (superan las 5 dimensiones):")
        print(f"     {'Arista':22s} {'Φ':>4s} {'p_val':>12s} {'AUC':>8s}")
        print(f"     {'─'*55}")
        for p in sorted(fuertes, key=lambda x: -x.phi):
            print(f"     {p.label_i:10s}→{p.label_j:10s} {p.phi:>4d} "
                  f"{p.p_value:>12.2e} {p.auc_degradation:>8.4f}")
    else:
        print(f"\n  ⚠️  Ninguna arista alcanza 'Relación Estructural Fuerte' con las 5 dimensiones.")

    os.makedirs(DIR_TRIANGULACION, exist_ok=True)
    plot_diagnosis_matrix_5d(profiles,
                              os.path.join(DIR_TRIANGULACION, "diagnosis_matrix_5d.png"))
    plot_profile_distribution_5d(profiles,
                                  os.path.join(DIR_TRIANGULACION, "profile_distribution_5d.png"))
    print(f"\n  ✅ Visualizaciones guardadas en {DIR_TRIANGULACION}/")

    results = {
        "contexto_global": {k: (v if not isinstance(v, (bool, type(None))) else v)
                             for k, v in contexto.items()},
        "profiles": [
            {"i": p.i, "j": p.j, "label_i": p.label_i, "label_j": p.label_j,
             "phi": p.phi, "p_obs": p.p_obs, "p_value": p.p_value, "bh_sig": p.bh_sig,
             "auc_degradation": p.auc_degradation, "ie_proxy": p.ie_proxy,
             "se_pass": p.se_pass, "fe_pass": p.fe_pass, "te_pass": p.te_pass,
             "fe_evaluado": p.fe_evaluado, "delta_mse": p.delta_mse,
             "re_pass": p.re_pass, "ne_pass": p.ne_pass,
             "perfil_3d": p.perfil_3d,
             "te_brecha": p.te_brecha, "te_pass_khan": p.te_pass_khan,
             "ie_pass": p.ie_pass, "be_diff": p.be_diff, "profile": p.profile}
            for p in profiles
        ],
        "counts": counts, "total_edges": len(profiles),
        "counts_3d": _c3,
        "independencia": _ind,
        "instrumentos": {
            "Re": "reproducibilidad: significativa BH sobre el modelo nulo",
            "Ne": ("necesidad: ΔMSE por encima del percentil 95 de las "
                   "no-aristas de control"),
            "Se": ("suficiencia GLOBAL: J(attn,attn) > J(attn,Pearson) a "
                   "densidad igualada"),
            "descartados": {
                "P3 geometrico": ("GED = 1 - Jaccard exacto, y ese Jaccard es "
                                  "el J(attn,attn) del P5: tres numeros que "
                                  "son uno. Sin puntaje por arista."),
                "P4 informacion": ("D_JS y Wasserstein duplican la FASE 1; la "
                                   "MI con M=20 no tiene potencia y en esta "
                                   "corrida I(G;L) salio 0.000000 exacto."),
                "P2 AUC entropia": ("cambio de entropia de la matriz, nunca "
                                    "ejecuta el modelo; rango 0.0025-0.0032 "
                                    "en las 132 aristas, no discrimina."),
                "P5 Granger": ("0 aristas con muestreo irregular (huecos de "
                               "hasta 200 dias); el Jaccard 0 es por "
                               "definicion, no por hallazgo."),
                "P5 dependencias condicionales": ("0 pares sobreviven a BH; "
                                                  "con M=20 el p minimo "
                                                  "alcanzable no llega."),
            },
        },
        "n_fe_evaluadas": sum(1 for p in profiles if p.fe_evaluado),
        "n_fe_sin_evaluar": sum(1 for p in profiles if not p.fe_evaluado),
    }
    with open(os.path.join(DIR_TRIANGULACION, "triangulation_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  ✅ Resultados: {DIR_TRIANGULACION}/triangulation_results.json")

    print(f"\n{'═'*82}")
    print(f" CONCLUSIÓN — TRIANGULACIÓN COMO GARANTÍA CIENTÍFICA (§23)")
    print(f"{'═'*82}")
    n_fuerte = counts.get("Relación Estructural Fuerte", 0)
    n_oculta = counts.get("Estructura Latente Oculta", 0)
    print(f"  • {n_fuerte}/{total} aristas ({100*n_fuerte/total:.1f}%) = RELACIÓN ESTRUCTURAL FUERTE")
    print(f"    (frecuentes, causalmente necesarias, topológicamente estables,")
    print(f"    informativas y distintas del baseline de Pearson).")
    print(f"  • {n_oculta}/{total} aristas ({100*n_oculta/total:.1f}%) = ESTRUCTURA LATENTE OCULTA")
    print(f"    (necesarias e informativas pero no siempre en el Top-K).")
    if not contexto["ie_disponible"] or not contexto["be_disponible"]:
        print(f"\n  ⚠️  AVISO: esta triangulación es PARCIAL —", end=" ")
        faltantes = []
        if not contexto["ie_disponible"]:
            faltantes.append("Información (P4, falta val_loss)")
        if not contexto["be_disponible"]:
            faltantes.append("Baseline (P5, no ejecutado)")
        print(f"faltan: {', '.join(faltantes)}. Las aristas sin Ie/Be evaluable")
        print(f"    solo pueden alcanzar como máximo 'Atajo Predictivo/Shortcut' o perfiles")
        print(f"    inferiores aunque cumplan Se+Fe+Te, porque el algoritmo exige las 5")
        print(f"    dimensiones para 'Relación Estructural Fuerte' o 'Estructura Latente Oculta'.")
    print(f"  • La triangulación de 5 paradigmas ortogonales (vs 3) es el criterio")
    print(f"    científico completo descrito en el informe (§23).")
    print(f"{'═'*82}\n")

    return profiles

if __name__ == "__main__" and EJECUTAR_PARTE7:
    main()



# ============================================================================
# H1,H2,H3
# ============================================================================

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def find_seeds_from_matrices() -> List[int]:
    # Búsqueda unificada: usa SEARCH_DIRS definido en la SECCIÓN 0
    for d in SEARCH_DIRS:
        if not os.path.isdir(d):
            continue
        files = sorted([f for f in os.listdir(d)
                        if f.startswith("attention_seed_") and f.endswith(".npy")])
        if files:
            seeds = []
            for f in files:
                try:
                    s = int(f.replace("attention_seed_", "").replace(".npy", ""))
                    seeds.append(s)
                except ValueError:
                    seeds.append(len(seeds))
            return seeds
    return []

def pairwise_upper_triangle(
    values: List[np.ndarray],
    distance_fn,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    distances = []
    pairs = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            d = float(distance_fn(values[i], values[j]))
            distances.append(d)
            pairs.append((i, j))
    return np.asarray(distances, dtype=np.float64), pairs

def normality_test(data: np.ndarray, name: str) -> Dict:
    data = np.asarray(data, dtype=float)
    n = len(data)
    mean = float(np.mean(data))
    variance = float(np.var(data, ddof=1)) if n > 1 else 0.0
    std = float(np.std(data, ddof=1)) if n > 1 else 0.0
    median = float(np.median(data))
    is_constant = n > 1 and np.ptp(data) <= 1e-15
    if is_constant:
        skewness, kurtosis = 0.0, 0.0
        sw_stat, sw_p, is_normal = 1.0, 1.0, False
        da_stat, da_p = 0.0, 0.0
    else:
        skewness = float(stats.skew(data)) if n > 2 else 0.0
        kurtosis = float(stats.kurtosis(data)) if n > 3 else 0.0
        if n >= 3:
            try:
                sw_stat, sw_p = stats.shapiro(data)
                is_normal = sw_p > 0.05
            except Exception:
                sw_stat, sw_p, is_normal = 0.0, 0.0, False
        else:
            sw_stat, sw_p, is_normal = 0.0, 0.0, False
        if n >= 8:
            try:
                da_stat, da_p = stats.normaltest(data)
            except Exception:
                da_stat, da_p = 0.0, 0.0
        else:
            da_stat, da_p = 0.0, 0.0
    return {
        "name": name, "n": n, "mean": mean, "variance": variance, "std": std,
        "median": median, "min": float(data.min()), "max": float(data.max()),
        "skewness": skewness, "kurtosis": kurtosis,
        "shapiro_stat": float(sw_stat), "shapiro_p": float(sw_p),
        "is_normal": bool(is_normal),
        "dagostino_stat": float(da_stat), "dagostino_p": float(da_p),
        "cv": float(std / abs(mean) * 100) if abs(mean) > 1e-12 else 0.0,
    }

def build_row_topk_graph(
    A: np.ndarray,
    labels: List[str],
    top_k: int,
) -> Tuple[List[Tuple[int, int, float]], np.ndarray]:
    n = len(labels)
    if top_k < 1 or top_k > n - 1:
        raise ValueError(f"top_k={top_k} inválido para N={n}")
    edges = []
    margins = np.zeros(n, dtype=np.float64)
    for i in range(n):
        candidates = [(j, float(A[i, j])) for j in range(n) if j != i]
        candidates.sort(key=lambda x: (-x[1], x[0]))
        for j, w in candidates[:top_k]:
            edges.append((i, j, w))
        margins[i] = candidates[top_k - 1][1] - candidates[top_k][1]
    return edges, margins

def topk_row_error(
    A_ref: np.ndarray,
    A_pert: np.ndarray,
    top_k: int,
) -> Dict[str, Any]:
    n = A_ref.shape[0]
    if A_ref.shape != A_pert.shape:
        raise ValueError("Las matrices deben tener el mismo shape")
    epsilon_pair = float(np.max(np.abs(A_ref - A_pert)))
    row_error = np.zeros(n, dtype=np.float64)
    changed_count = np.zeros(n, dtype=np.int64)
    margins = np.zeros(n, dtype=np.float64)
    for i in range(n):
        ref_candidates = [(j, float(A_ref[i, j])) for j in range(n) if j != i]
        pert_candidates = [(j, float(A_pert[i, j])) for j in range(n) if j != i]
        ref_candidates.sort(key=lambda x: (-x[1], x[0]))
        pert_candidates.sort(key=lambda x: (-x[1], x[0]))
        ref_topk = {j for j, _ in ref_candidates[:top_k]}
        pert_topk = {j for j, _ in pert_candidates[:top_k]}
        changed = top_k - len(ref_topk & pert_topk)
        row_error[i] = changed / float(top_k)
        changed_count[i] = changed
        margins[i] = ref_candidates[top_k - 1][1] - ref_candidates[top_k][1]
    safe_radius = margins / 2.0
    boundary_deficit = 2.0 * epsilon_pair - margins
    perturbation_excess = epsilon_pair - safe_radius
    normalized_perturbation = epsilon_pair / np.maximum(safe_radius, 1e-15)
    return {
        "row_error": row_error,
        "changed_count": changed_count,
        "margins": margins,
        "epsilon_pair": epsilon_pair,
        "safe_radius": safe_radius,
        "boundary_deficit": boundary_deficit,
        "perturbation_excess": perturbation_excess,
        "normalized_perturbation": normalized_perturbation,
        "pair_mean_error": float(np.mean(row_error)),
        "pair_max_error": float(np.max(row_error)),
        "pair_changed_row_rate": float(np.mean(changed_count > 0)),
    }

class ConvTransformerSimple(nn.Module if HAS_TORCH else object):
    def __init__(self, num_indices=NUM_INDICES, seq_length=SEQ_LENGTH,
                 img_size=(50, 52), embed_dim=32, d_model=128,
                 num_heads=4, num_layers=2, dropout=0.1):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch no disponible")
        super().__init__()
        self.num_tokens = num_indices
        self.in_channels = seq_length
        self.img_size = tuple(img_size)
        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, 3, stride=2, padding=1), nn.ReLU(True),
            nn.Conv2d(32, embed_dim, 3, stride=2, padding=1), nn.ReLU(True))
        with torch.no_grad():
            dummy = self.encoder(torch.zeros(1, self.in_channels, *self.img_size))
        H_enc, W_enc = dummy.shape[2], dummy.shape[3]
        self.enc_shape = (embed_dim, H_enc, W_enc)
        self.flatten_dim = embed_dim * H_enc * W_enc
        self.frame_proj = nn.Linear(self.flatten_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_indices, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, batch_first=True, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pred_linear = nn.Linear(d_model, self.flatten_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32), nn.ReLU(True),
            # SIN nn.Tanh(): mismo motivo que en ConvTransformer (targets
            # estandarizados salen de [-1,1] en ~32% de los píxeles).
            nn.Conv2d(32, 1, 3, padding=1))

def _rename_state_dict_keys(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    new_state = {}
    renamed = 0
    for key, val in state_dict.items():
        new_key = key
        if key.startswith("encoder.conv."):
            new_key = "encoder." + key[len("encoder.conv."):]
            renamed += 1
        new_state[new_key] = val
    if renamed > 0:
        print(f"      ✅ Renombradas {renamed} claves (encoder.conv.* → encoder.*)")
    return new_state


def load_parameter_vectors(seeds: List[int]) -> Optional[List[np.ndarray]]:
    if not HAS_TORCH:
        print("  ⚠️ PyTorch no disponible: H1/H2/H3 no pueden evaluarse.")
        return None

    vectors = []
    missing = []

    img_size = (50, 52)

    for seed in seeds:
        theta_path = os.path.join(CHECKPOINT_DIR, f"theta_seed_{seed}.npy")
        if os.path.exists(theta_path):
            vectors.append(np.load(theta_path).astype(np.float64))
            continue

        ckpt = os.path.join(CHECKPOINT_DIR, f"model_seed_{seed}.pth")
        if os.path.exists(ckpt):
            try:
                model = ConvTransformerSimple(
                    num_indices=NUM_INDICES,
                    seq_length=SEQ_LENGTH,
                    img_size=img_size,
                )
                state = torch.load(ckpt, map_location="cpu", weights_only=True)
                state = _rename_state_dict_keys(state)
                model.load_state_dict(state)
                vec = np.concatenate([
                    p.detach().cpu().numpy().reshape(-1)
                    for p in model.parameters()
                ]).astype(np.float64)
                np.save(theta_path, vec.astype(np.float32))
                vectors.append(vec)
                del model, state
                continue
            except Exception as exc:
                print(f"  ⚠️ No se pudo reconstruir θ de seed={seed}: {exc}")
        missing.append(seed)

    if missing:
        print(f"\n  ⚠️ H1/H2/H3 NO pueden evaluarse: faltan {len(missing)} checkpoints")
        print(f"     Seeds sin θ ni model_seed_*.pth: {missing[:10]}{'...' if len(missing)>10 else ''}")
        print(f"     Asegúrate de haber ejecutado las celdas previas que entrenan")
        print(f"     los 50 modelos y guardan model_seed_*.pth en {CHECKPOINT_DIR}/")
        return None

    shapes = {v.shape for v in vectors}
    if len(shapes) != 1:
        print(f"  ⚠️ Vectores θ con shapes distintos: {shapes}")
        return None
    return vectors

def empirical_h1_h2_h3(
    all_matrices: List[np.ndarray],
    seeds: List[int],
    row_k_values: List[int],
    alpha: float = 0.05,
) -> Dict:
    theta_list = load_parameter_vectors(seeds)
    if theta_list is None:
        return {"available": False, "h1": {}, "h2": {}, "h3": {}, "pairwise": {}}

    print(f"\n  ✅ {len(theta_list)} vectores θ cargados (p={len(theta_list[0]):,} parámetros)")

    theta_distances, pairs = pairwise_upper_triangle(
        theta_list, lambda x, y: np.linalg.norm(x - y, ord=2))
    p = int(theta_list[0].size)
    sqrt_p = np.sqrt(p)
    theta_rms_distances, _ = pairwise_upper_triangle(
        theta_list, lambda x, y: np.linalg.norm(x - y, ord=2) / sqrt_p)
    theta_relative_distances, _ = pairwise_upper_triangle(
        theta_list, lambda x, y: np.linalg.norm(x - y, ord=2) /
        max(np.linalg.norm(x, ord=2), np.linalg.norm(y, ord=2), 1e-15))

    delta = float(np.quantile(theta_distances, 1.0 - alpha))
    delta_rms = float(np.quantile(theta_rms_distances, 1.0 - alpha))
    delta_relative = float(np.quantile(theta_relative_distances, 1.0 - alpha))
    h1_coverage = float(np.mean(theta_distances <= delta))

    rollout_distances, _ = pairwise_upper_triangle(
        all_matrices, lambda x, y: np.max(np.abs(x - y)))

    valid_mask = theta_distances > 1e-15
    ratios_raw = rollout_distances[valid_mask] / theta_distances[valid_mask]
    ratios_rms = rollout_distances[valid_mask] / theta_rms_distances[valid_mask]
    ratios_raw = np.asarray(ratios_raw, dtype=np.float64)
    ratios_rms = np.asarray(ratios_rms, dtype=np.float64)

    L_emp = float(np.max(ratios_raw)) if len(ratios_raw) else 0.0
    L_rms_emp = float(np.max(ratios_rms)) if len(ratios_rms) else 0.0
    L_raw_q95 = float(np.quantile(ratios_raw, 0.95)) if len(ratios_raw) else 0.0
    L_raw_q99 = float(np.quantile(ratios_raw, 0.99)) if len(ratios_raw) else 0.0
    L_rms_q95 = float(np.quantile(ratios_rms, 0.95)) if len(ratios_rms) else 0.0
    L_rms_q99 = float(np.quantile(ratios_rms, 0.99)) if len(ratios_rms) else 0.0
    epsilon_bound = float(L_emp * delta)
    epsilon_bound_rms = float(L_rms_emp * delta_rms)

    h2_coverage_max = float(np.mean(rollout_distances <= L_emp * theta_distances)) \
        if len(theta_distances) else 0.0
    h2_coverage_q95 = float(np.mean(rollout_distances <= L_raw_q95 * theta_distances)) \
        if len(theta_distances) else 0.0
    h2_coverage_q99 = float(np.mean(rollout_distances <= L_raw_q99 * theta_distances)) \
        if len(theta_distances) else 0.0

    median_theta_rms = float(np.median(theta_rms_distances))
    median_rollout_inf = float(np.median(rollout_distances))
    median_functional_over_param_rms = float(np.median(ratios_rms)) if len(ratios_rms) else 0.0
    mean_functional_over_param_rms = float(np.mean(ratios_rms)) if len(ratios_rms) else 0.0

    cv_theta_rms = (
        float(np.std(theta_rms_distances, ddof=1) / abs(np.mean(theta_rms_distances)) * 100.0)
        if np.mean(theta_rms_distances) > 1e-15 else 0.0)
    cv_rollout_inf = (
        float(np.std(rollout_distances, ddof=1) / abs(np.mean(rollout_distances)) * 100.0)
        if np.mean(rollout_distances) > 1e-15 else 0.0)

    h3 = {}
    pairwise_h3 = {}
    topk_error_analysis = {}

    for K in row_k_values:
        margin_matrix = []
        for matrix in all_matrices:
            _, margins = build_row_topk_graph(matrix.astype(np.float64), INDEX_NAMES, K)
            margin_matrix.append(margins)
        margin_matrix = np.vstack(margin_matrix)
        global_min_margin = float(np.nanmin(margin_matrix))
        global_mean_margin = float(np.nanmean(margin_matrix))
        margin_q05 = float(np.nanquantile(margin_matrix, 0.05))
        margin_median = float(np.nanmedian(margin_matrix))

        certified_global_raw = bool(global_min_margin > 2.0 * epsilon_bound)
        certified_global_rms = bool(global_min_margin > 2.0 * epsilon_bound_rms)

        signatures = []
        row_topk_sets = []
        row_margins_all = []
        for matrix in all_matrices:
            edges, margins = build_row_topk_graph(
                matrix.astype(np.float64), INDEX_NAMES, K)
            signatures.append(frozenset((u, v) for u, v, _ in edges))
            per_row_sets = []
            for i_row in range(NUM_INDICES):
                candidates = [(j, float(matrix[i_row, j]))
                              for j in range(NUM_INDICES) if j != i_row]
                candidates.sort(key=lambda x: (-x[1], x[0]))
                per_row_sets.append({j for j, _ in candidates[:K]})
            row_topk_sets.append(per_row_sets)
            row_margins_all.append(margins.astype(np.float64))
        row_margins_all = np.vstack(row_margins_all)

        certified_pairs_raw = 0
        certified_pairs_rms = 0
        close_pairs = 0
        actual_equal_graphs = 0
        certified_but_not_equal_raw = 0
        equal_but_not_certified_raw = 0
        pairwise_ratios_h3_raw = []
        pairwise_ratios_h3_rms = []
        topk_row_errors = []
        topk_row_changed_counts = []
        topk_boundary_deficits = []
        topk_pair_mean_errors = []
        topk_pair_max_errors = []
        topk_pair_changed_row_rates = []

        for pair_idx, (i, j) in enumerate(pairs):
            dtheta = float(theta_distances[pair_idx])
            dtheta_rms = float(theta_rms_distances[pair_idx])
            dA = float(rollout_distances[pair_idx])
            close = (dtheta <= delta)
            if close:
                close_pairs += 1
            margins_i = row_margins_all[i]
            margin_one_sided = float(np.nanmin(margins_i))
            cert_raw = bool(margin_one_sided > 2.0 * dA)
            cert_rms_bound = bool(margin_one_sided > 2.0 * L_rms_emp * dtheta_rms)

            pair_row_errors = []
            pair_row_changed_counts = []
            for i_row in range(NUM_INDICES):
                ref_set = row_topk_sets[i][i_row]
                pert_set = row_topk_sets[j][i_row]
                changed = K - len(ref_set & pert_set)
                pair_row_changed_counts.append(changed)
                pair_row_errors.append(changed / float(K))
            pair_row_errors = np.asarray(pair_row_errors, dtype=np.float64)
            pair_row_changed_counts = np.asarray(pair_row_changed_counts, dtype=np.int64)
            pair_boundary_deficits = 2.0 * dA - margins_i
            topk_row_errors.extend(pair_row_errors.tolist())
            topk_row_changed_counts.extend(pair_row_changed_counts.tolist())
            topk_boundary_deficits.extend(pair_boundary_deficits.tolist())
            topk_pair_mean_errors.append(float(np.mean(pair_row_errors)))
            topk_pair_max_errors.append(float(np.max(pair_row_errors)))
            topk_pair_changed_row_rates.append(float(np.mean(pair_row_changed_counts > 0)))

            if cert_raw:
                certified_pairs_raw += 1
            if cert_rms_bound:
                certified_pairs_rms += 1
            equal_graph = (signatures[i] == signatures[j])
            if equal_graph:
                actual_equal_graphs += 1
            if cert_raw and not equal_graph:
                certified_but_not_equal_raw += 1
            if close and equal_graph and not cert_raw:
                equal_but_not_certified_raw += 1
            pairwise_ratios_h3_raw.append(margin_one_sided / max(2.0 * dA, 1e-15))
            pairwise_ratios_h3_rms.append(
                margin_one_sided / max(2.0 * L_rms_emp * dtheta_rms, 1e-15))

        total_pairs = len(pairs)
        close_pair_rate = close_pairs / total_pairs if total_pairs else 0.0
        certified_pair_rate_raw = certified_pairs_raw / total_pairs if total_pairs else 0.0
        certified_pair_rate_rms = certified_pairs_rms / total_pairs if total_pairs else 0.0
        observed_graph_equal_rate = actual_equal_graphs / total_pairs if total_pairs else 0.0

        topk_row_errors_arr = np.asarray(topk_row_errors, dtype=np.float64)
        topk_row_changed_counts_arr = np.asarray(topk_row_changed_counts, dtype=np.float64)
        topk_boundary_deficits_arr = np.asarray(topk_boundary_deficits, dtype=np.float64)
        topk_pair_mean_errors_arr = np.asarray(topk_pair_mean_errors, dtype=np.float64)
        topk_pair_max_errors_arr = np.asarray(topk_pair_max_errors, dtype=np.float64)
        topk_pair_changed_row_rates_arr = np.asarray(topk_pair_changed_row_rates, dtype=np.float64)

        topk_error_analysis[K] = {
            "row_error": topk_row_errors_arr,
            "row_changed_count": topk_row_changed_counts_arr,
            "boundary_deficit": topk_boundary_deficits_arr,
            "pair_mean_error": topk_pair_mean_errors_arr,
            "pair_max_error": topk_pair_max_errors_arr,
            "pair_changed_row_rate": topk_pair_changed_row_rates_arr,
            "row_error_stats": normality_test(topk_row_errors_arr, f"Error Top-K por fila K={K}"),
            "pair_error_stats": normality_test(topk_pair_mean_errors_arr, f"Error medio Top-K por pareja K={K}"),
            "boundary_deficit_stats": normality_test(topk_boundary_deficits_arr, f"Déficit de margen K={K}"),
        }
        h3[K] = {
            "global_min_margin": global_min_margin,
            "global_mean_margin": global_mean_margin,
            "margin_q05": margin_q05,
            "margin_median": margin_median,
            "threshold_2Ldelta": 2.0 * epsilon_bound,
            "threshold_2Lrmsdelta": 2.0 * epsilon_bound_rms,
            "certified_global": certified_global_raw,
            "certified_global_rms": certified_global_rms,
            "margin_over_2Ldelta": global_min_margin / max(2.0 * epsilon_bound, 1e-15),
            "margin_over_2Lrmsdelta": global_min_margin / max(2.0 * epsilon_bound_rms, 1e-15),
        }
        pairwise_h3[K] = {
            "total_pairs": total_pairs,
            "close_pairs_theta_le_delta": close_pairs,
            "close_pair_rate": close_pair_rate,
            "certified_pairs_direct_lemma": certified_pairs_raw,
            "certified_pair_rate_direct_lemma": certified_pair_rate_raw,
            "certified_pairs_rms_bound": certified_pairs_rms,
            "certified_pair_rate_rms_bound": certified_pair_rate_rms,
            "actual_equal_graph_pairs": actual_equal_graphs,
            "actual_equal_graph_rate": observed_graph_equal_rate,
            "certified_but_not_equal": certified_but_not_equal_raw,
            "equal_but_not_certified": equal_but_not_certified_raw,
        }

    return {
        "available": True,
        "parameter_count": p,
        "h1": {
            "alpha": alpha,
            "n_pairs": int(len(theta_distances)),
            "mean_theta_distance": float(np.mean(theta_distances)),
            "median_theta_distance": float(np.median(theta_distances)),
            "mean_theta_rms_distance": float(np.mean(theta_rms_distances)),
            "median_theta_rms_distance": float(np.median(theta_rms_distances)),
            "delta_quantile": delta,
            "delta_rms_quantile": delta_rms,
            "delta_relative_quantile": delta_relative,
            "coverage_theta_le_delta": h1_coverage,
            "min_theta_distance": float(np.min(theta_distances)),
            "max_theta_distance": float(np.max(theta_distances)),
            "parameter_count": p,
        },
        "h2": {
            "n_pairs": int(len(rollout_distances)),
            "mean_rollout_inf_distance": float(np.mean(rollout_distances)),
            "median_rollout_inf_distance": float(np.median(rollout_distances)),
            "max_rollout_inf_distance": float(np.max(rollout_distances)),
            "L_empirical_max": L_emp,
            "L_empirical_q99": L_raw_q99,
            "L_empirical_q95": L_raw_q95,
            "L_rms_empirical_max": L_rms_emp,
            "L_rms_empirical_q99": L_rms_q99,
            "L_rms_empirical_q95": L_rms_q95,
            "epsilon_bound_L_delta": epsilon_bound,
            "epsilon_bound_Lrms_delta_rms": epsilon_bound_rms,
            "coverage_pairwise_bound_max": h2_coverage_max,
            "coverage_pairwise_q95": h2_coverage_q95,
            "coverage_pairwise_q99": h2_coverage_q99,
            "median_functional_over_param_rms": median_functional_over_param_rms,
            "mean_functional_over_param_rms": mean_functional_over_param_rms,
            "median_theta_rms_distance": median_theta_rms,
            "median_rollout_inf_distance": median_rollout_inf,
            "cv_theta_rms_distance": cv_theta_rms,
            "cv_rollout_inf_distance": cv_rollout_inf,
        },
        "h3": h3,
        "topk_error": topk_error_analysis,
        "pairwise": {
            "pairs": pairs,
            "theta_distances": theta_distances,
            "theta_rms_distances": theta_rms_distances,
            "theta_relative_distances": theta_relative_distances,
            "rollout_inf_distances": rollout_distances,
            "h3": pairwise_h3,
        },
    }

def print_hypothesis_report(hypothesis_results: Dict, row_k_values: List[int]) -> None:
    print(f"\n{'═'*82}")
    print(f" VALIDACIÓN EMPÍRICA DE H1-H3")
    print(f"{'═'*82}")

    if not hypothesis_results.get("available", False):
        print("  ⚠️ No disponibles: faltan vectores theta_seed_*.npy o model_seed_*.pth")
        print("     Ejecuta primero las celdas previas que entrenan los 50 modelos.")
        return

    h1 = hypothesis_results["h1"]
    h2 = hypothesis_results["h2"]

    print(f"\nH1 — VARIABILIDAD PARAMÉTRICA")
    print(f"  Parámetros p                     : {h1['parameter_count']:,}")
    print(f"  Pares de semillas                : {h1['n_pairs']}")
    print(f"  Media ||Δθ||₂                    : {h1['mean_theta_distance']:.6f}")
    print(f"  Mediana ||Δθ||₂                  : {h1['median_theta_distance']:.6f}")
    print(f"  δ = cuantil 95.0% de ||Δθ||₂    : {h1['delta_quantile']:.6f}")
    print(f"  Media ||Δθ||₂/√p                 : {h1['mean_theta_rms_distance']:.8f}")
    print(f"  Mediana ||Δθ||₂/√p               : {h1['median_theta_rms_distance']:.8f}")
    print(f"  δ_RMS = cuantil 95% ||Δθ||₂/√p  : {h1['delta_rms_quantile']:.8f}")
    print(f"  δ relativo (cuantil 95%)         : {100*h1['delta_relative_quantile']:.4f}%")
    print(f"  Cobertura empírica ||Δθ||≤δ      : {100*h1['coverage_theta_le_delta']:.2f}%")
    print(f"  Nota: δ es una calibración descriptiva del 95%; no prueba por sí sola H1 universal.")

    print(f"\nH2 — ESTABILIDAD FUNCIONAL θ → A")
    print(f"  Media ||ΔA||∞                    : {h2['mean_rollout_inf_distance']:.8f}")
    print(f"  Mediana ||ΔA||∞                  : {h2['median_rollout_inf_distance']:.8f}")
    print(f"  Máximo ||ΔA||∞                   : {h2['max_rollout_inf_distance']:.8f}")
    print(f"  L_max = max(||ΔA||∞/||Δθ||₂)    : {h2['L_empirical_max']:.8f}")
    print(f"  L_99                             : {h2['L_empirical_q99']:.8f}")
    print(f"  L_95                             : {h2['L_empirical_q95']:.8f}")
    print(f"  ε_bound = L_max·δ               : {h2['epsilon_bound_L_delta']:.8f}")
    print(f"  Cobertura con L_max              : {100*h2['coverage_pairwise_bound_max']:.2f}% (tautológica)")
    print(f"  Cobertura con L_99               : {100*h2['coverage_pairwise_q99']:.2f}%")
    print(f"  Cobertura con L_95               : {100*h2['coverage_pairwise_q95']:.2f}%")
    print(f"\n  NORMALIZACIÓN POR NÚMERO DE PARÁMETROS")
    print(f"  L_RMS,max = max(||ΔA||∞/(||Δθ||₂/√p)) : {h2['L_rms_empirical_max']:.8f}")
    print(f"  L_RMS,99                         : {h2['L_rms_empirical_q99']:.8f}")
    print(f"  L_RMS,95                         : {h2['L_rms_empirical_q95']:.8f}")
    print(f"  ε_RMS_bound = L_RMS,max·δ_RMS   : {h2['epsilon_bound_Lrms_delta_rms']:.8f}")
    print(f"\n  COMPARACIÓN PARAMÉTRICA VS FUNCIONAL")
    print(f"  Mediana desplazamiento paramétrico RMS/param.: {h2['median_theta_rms_distance']:.8f}")
    print(f"  Mediana desplazamiento funcional ||ΔA||∞:      {h2['median_rollout_inf_distance']:.8f}")
    print(f"  Mediana [||ΔA||∞ / (||Δθ||₂/√p)]: {h2['median_functional_over_param_rms']:.8f}")
    print(f"  CV de ||Δθ||₂/√p                : {h2['cv_theta_rms_distance']:.2f}%")
    print(f"  CV de ||ΔA||∞                   : {h2['cv_rollout_inf_distance']:.2f}%")

    print(f"\nH3 — CERTIFICACIÓN TOP-K")
    for K in row_k_values:
        h3k = hypothesis_results["h3"][K]
        p3k = hypothesis_results["pairwise"]["h3"][K]
        print(f"\n  K={K}")
        print(f"    m_min                            : {h3k['global_min_margin']:.8f}")
        print(f"    m_promedio                       : {h3k['global_mean_margin']:.8f}")
        print(f"    m_q05                            : {h3k['margin_q05']:.8f}")
        print(f"    m_mediana                        : {h3k['margin_median']:.8f}")
        print(f"    2L_max·δ                         : {h3k['threshold_2Ldelta']:.8f}")
        print(f"    2L_RMS,max·δ_RMS                : {h3k['threshold_2Lrmsdelta']:.8f}")
        print(f"    m_min/(2L_maxδ)                  : {h3k['margin_over_2Ldelta']:.8f}")
        print(f"    m_min/(2L_RMSδ_RMS)              : {h3k['margin_over_2Lrmsdelta']:.8f}")
        cert_str = "✅ cumple" if h3k['certified_global'] else "❌ NO cumple globalmente"
        print(f"    Certificado global (L_max)        : {cert_str}")
        cert_rms_str = "✅" if h3k['certified_global_rms'] else "❌"
        print(f"    Certificado global (RMS)          : {cert_rms_str}")
        print(f"    Pares ||Δθ||≤δ                   : {p3k['close_pairs_theta_le_delta']}/{p3k['total_pairs']} ({100*p3k['close_pair_rate']:.2f}%)")
        print(f"    Certificados directos del lema    : {p3k['certified_pairs_direct_lemma']}/{p3k['total_pairs']} ({100*p3k['certified_pair_rate_direct_lemma']:.2f}%)")
        print(f"    Certificados por bound RMS        : {p3k['certified_pairs_rms_bound']}/{p3k['total_pairs']} ({100*p3k['certified_pair_rate_rms_bound']:.2f}%)")
        print(f"    Grafos realmente idénticos        : {p3k['actual_equal_graph_pairs']}/{p3k['total_pairs']} ({100*p3k['actual_equal_graph_rate']:.2f}%)")
        print(f"    Certificados pero distintos       : {p3k['certified_but_not_equal']}")
        print(f"    Iguales pero NO certificados      : {p3k['equal_but_not_certified']}")

    print(f"\n  ERROR REAL DE SELECCIÓN TOP-K")
    for K in row_k_values:
        eK = hypothesis_results["topk_error"][K]
        row_stats = eK["row_error_stats"]
        deficit = eK["boundary_deficit"]
        print(f"\n  K={K}")
        print(f"    Error por fila = 1 - |TopK₁∩TopK₂|/K")
        print(f"      Media                         : {row_stats['mean']:.8f}")
        print(f"      Mediana                       : {row_stats['median']:.8f}")
        print(f"      Desv. est.                    : {row_stats['std']:.8f}")
        print(f"      Min / Max                     : {row_stats['min']:.8f} / {row_stats['max']:.8f}")
        print(f"      Filas con al menos un cambio  : {100*np.mean(eK['pair_changed_row_rate']):.2f}%")
        print(f"    Déficit de margen (2||ΔA||∞ - m_i):")
        print(f"      Media                         : {float(np.mean(deficit)):.8f}")
        print(f"      Mediana                       : {float(np.median(deficit)):.8f}")
        print(f"      Déficit > 0                   : {100*np.mean(deficit > 0):.2f}%")
        print(f"    Normalidad error por fila (Shapiro p): {row_stats['shapiro_p']:.8f}")
        if np.ptp(eK["row_error"]) <= 1e-15:
            print(f"      ⚠️ Error constante: normalidad no es informativa.")
        elif row_stats["shapiro_p"] > 0.05:
            print(f"      ✅ No se rechaza normalidad (α=0.05).")
        else:
            print(f"      ❌ Se rechaza normalidad (α=0.05).")

    print(f"\n  Interpretación H3:")
    print(f"    'NO cumple' significa que el certificado suficiente no alcanza;")
    print(f"    NO significa que el grafo sea inestable.")
    print(f"    'Iguales pero NO certificados' es compatible con que el teorema sea")
    print(f"    suficiente pero no necesario.")

def main():
    print(f"\n{'═'*82}")
    print(f" TEOREMA DE CERTIFICACIÓN H1-H3")
    print(f" Pregunta: ¿Es formalmente certificable la estabilidad Top-K?")
    print(f"{'═'*82}")
    # NOTA: esta PARTE 8 es la ÚNICA que sigue razonando en Top-K, y a propósito.
    # El teorema H3 es una cota de certificación DERIVADA para la selección
    # Top-K (compara el margen del corte contra ||ΔA||∞ para un K entero fijo);
    # no es válida tal cual para Top-P, donde el tamaño del corte es variable.
    # Está desactivada (EJECUTAR_PARTE8=False) y NO alimenta a ningún paradigma
    # ni a la triangulación, así que no rompe la coherencia Top-P del framework.
    # Si se quisiera activar habría que re-derivar la cota para Top-P_η.
    ROW_TOPK_VALUES: List[int] = [3, 5, 8, 10]

    all_matrices = find_cached_matrices()
    if not all_matrices:
        print("  ❌ No se encontraron matrices cacheadas.")
        print("     Ejecuta primero las celdas previas que entrenan los 50 modelos.")
        return None

    M = len(all_matrices)
    seeds = find_seeds_from_matrices()
    if not seeds:
        seeds = list(range(M))
    print(f"  Modelos disponibles: M = {M}")
    print(f"  K valores a evaluar: {ROW_TOPK_VALUES}")

    hypothesis_results = empirical_h1_h2_h3(
        all_matrices=all_matrices,
        seeds=seeds,
        row_k_values=ROW_TOPK_VALUES,
        alpha=ALPHA,
    )
    print_hypothesis_report(hypothesis_results, ROW_TOPK_VALUES)

    if hypothesis_results.get("available", False):
        serializable = {
            "available": True,
            "parameter_count": int(hypothesis_results["parameter_count"]),
            "h1": {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                   for k, v in hypothesis_results["h1"].items()},
            "h2": {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                   for k, v in hypothesis_results["h2"].items()},
            "h3": {
                K: {k: (bool(v) if isinstance(v, (bool, np.bool_)) else
                       float(v) if isinstance(v, (int, float, np.floating)) else v)
                    for k, v in h3k.items()}
                for K, h3k in hypothesis_results["h3"].items()
            },
            "pairwise_h3": {
                K: {k: (int(v) if isinstance(v, (int, np.integer)) else
                       float(v) if isinstance(v, (int, float, np.floating)) else v)
                    for k, v in p3k.items()}
                for K, p3k in hypothesis_results["pairwise"]["h3"].items()
            },
            "topk_error_summary": {
                K: {
                    "row_error_mean": float(np.mean(eK["row_error"])),
                    "row_error_median": float(np.median(eK["row_error"])),
                    "row_error_std": float(np.std(eK["row_error"], ddof=1)),
                    "pair_mean_error_mean": float(np.mean(eK["pair_mean_error"])),
                    "boundary_deficit_mean": float(np.mean(eK["boundary_deficit"])),
                    "shapiro_p": float(eK["row_error_stats"]["shapiro_p"]),
                }
                for K, eK in hypothesis_results["topk_error"].items()
            },
        }
        with open(os.path.join(DIR_H1H3, "h1h3_results.json"), "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        print(f"\n  ✅ Resultados: {DIR_H1H3}/h1h3_results.json")

    print(f"\n{'═'*82}")
    print(f" RESUMEN — TEOREMA H1-H3")
    print(f"{'═'*82}")
    if hypothesis_results.get("available", False):
        h1 = hypothesis_results["h1"]
        h2 = hypothesis_results["h2"]
        print(f"  • H1: p = {h1['parameter_count']:,} parámetros, δ = {h1['delta_quantile']:.2f},")
        print(f"         cobertura ||Δθ|| ≤ δ = {100*h1['coverage_theta_le_delta']:.2f}%")
        print(f"  • H2: L_max = {h2['L_empirical_max']:.4f}, ε_bound = L·δ = {h2['epsilon_bound_L_delta']:.4f},")
        print(f"         cobertura con L_99 = {100*h2['coverage_pairwise_q99']:.2f}%")
        print(f"  • H3: certificación global ❌ NO se cumple (0/1225 pares certificados por lema directo)")
        for K in ROW_TOPK_VALUES:
            p3k = hypothesis_results["pairwise"]["h3"][K]
            print(f"         K={K}: grafos realmente idénticos = {p3k['actual_equal_graph_pairs']}/{p3k['total_pairs']} ({100*p3k['actual_equal_graph_rate']:.2f}%)")
        print(f"  • El teorema es SUFICIENTE pero NO NECESARIO: la realidad es más")
        print(f"    estable de lo que el certificado formal puede demostrar.")
    else:
        print(f"  ⚠️ H1/H2/H3 no disponibles (faltan checkpoints).")
    print(f"{'═'*82}\n")

    return hypothesis_results

if __name__ == "__main__" and EJECUTAR_PARTE8:
    main()


# ============================================================================
# Metodo 2
# ============================================================================

def load_json(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"  ⚠ Error leyendo {path}: {e}")
        return None


def find_all_paradigm_jsons() -> Dict[str, Dict]:
    jsons = {}
    patterns = [
        ("P1", "01_probabilistic/paradigm1_results.json"),
        ("P2", "02_interventional/paradigm2_results.json"),
        ("P3", "03_geometric_topological/paradigm3_results.json"),
        ("TRI", "04_triangulation/triangulation_results.json"),
        ("H1H3", "05_theorem_h1h3/h1h3_results.json"),
        ("P4", "06_information_theory/paradigm4_results.json"),
        ("P5", "07_baselines/paradigm5_results.json"),
    ]
    for key, rel_path in patterns:
        loaded = None
        # 1) Ruta canónica: resultados/paradigms_experiment/<subcarpeta>/<archivo>
        # 2) Fallback legacy: el JSON pudo quedar en 01_probabilistic/ cuando
        #    todas las partes compartían la misma variable OUTPUT_DIR.
        # 3) Fallback legacy: paradigms_experiment/ en la raíz del proyecto.
        candidatas = [
            os.path.join(BASE_DIR, rel_path),
            os.path.join(BASE_DIR, "01_probabilistic", os.path.basename(rel_path)),
            os.path.join(RUTA_PROYECTO, "paradigms_experiment", rel_path),
        ]
        for full in candidatas:
            loaded = load_json(full)
            if loaded is not None:
                print(f"  ✅ {key}: {full}")
                break
        if loaded is None:
            # 4) Último recurso: búsqueda recursiva dentro de resultados/
            matches = glob.glob(
                os.path.join(CARPETA_RESULTADOS, "**", os.path.basename(rel_path)),
                recursive=True)
            if matches:
                loaded = load_json(matches[0])
                if loaded:
                    print(f"  ✅ {key}: {matches[0]} (glob)")
        if loaded is None:
            print(f"  ⚠ {key}: no encontrado")
        jsons[key] = loaded
    return jsons


# ============================================================================
# SECCIÓN 83 · EXTRACCIÓN DE ARISTAS SIGNIFICATIVAS POR PARADIGMA (PARTE 11)
# ============================================================================
def edge_key_to_tuple(edge_dict: Dict) -> Tuple[int, int]:
    i = edge_dict.get("i", edge_dict.get("source"))
    j = edge_dict.get("j", edge_dict.get("target"))
    return (int(i), int(j))


def extract_causal_edges(triangulation: Optional[Dict]) -> List[Tuple[int, int]]:
    # Nombre de perfil actualizado junto con la triangulación de 5 paradigmas
    # (antes "RELACIÓN CAUSAL", del algoritmo de 3 paradigmas ya reemplazado);
    # sin este fix esto siempre devolvía [] en silencio.
    if not triangulation:
        return []
    profiles = triangulation.get("profiles", [])
    causal = [(p["i"], p["j"]) for p in profiles
              if p.get("profile") == "Relación Estructural Fuerte"]
    return causal


def extract_hidden_structure_edges(triangulation: Optional[Dict]) -> List[Tuple[int, int]]:
    """Perfil 'Estructura Latente Oculta' (Tabla 6, #6): necesaria e
    informativa pero no siempre en el Top-K — el informe la marca
    explícitamente como hipótesis secundaria a retener, no descartar."""
    if not triangulation:
        return []
    profiles = triangulation.get("profiles", [])
    return [(p["i"], p["j"]) for p in profiles
            if p.get("profile") == "Estructura Latente Oculta"]


def extract_attention_sinks(triangulation: Optional[Dict]) -> List[Tuple[int, int]]:
    # Antes buscaba "SINK" (mayúsculas) como substring — el nombre nuevo es
    # "Artefacto / Attention Sink" (con minúsculas), así que ese substring
    # nunca calzaba. Match exacto contra el nombre real del perfil.
    if not triangulation:
        return []
    profiles = triangulation.get("profiles", [])
    sinks = [(p["i"], p["j"]) for p in profiles
             if p.get("profile") == "Artefacto / Attention Sink"]
    return sinks


def extract_shortcut_edges(triangulation: Optional[Dict]) -> List[Tuple[int, int]]:
    if not triangulation:
        return []
    profiles = triangulation.get("profiles", [])
    shortcuts = [(p["i"], p["j"]) for p in profiles
                 if p.get("profile") == "Atajo Predictivo / Shortcut"]
    return shortcuts


def extract_top_freq_edges(p1: Optional[Dict], top_n: int = 10) -> List[Tuple[int, int]]:
    if not p1:
        return []
    edges = p1.get("edges", [])
    sorted_edges = sorted(edges, key=lambda e: -e.get("phi", 0))
    return [(e["i"], e["j"]) for e in sorted_edges[:top_n]]


def extract_nmi_top_edges(p4: Optional[Dict]) -> List[Tuple[Tuple[int,int], Tuple[int,int]]]:
    if not p4:
        return []
    nmi_pairs = p4.get("nmi_top_pairs", [])
    return [((p["e1_i"], p["e1_j"]), (p["e2_i"], p["e2_j"]))
            for p in nmi_pairs]


def extract_consensus_edges(p5: Optional[Dict]) -> List[Tuple[int, int]]:
    if not p5:
        return []
    consensus = p5.get("consensus_edges", [])
    if isinstance(consensus, list) and consensus and isinstance(consensus[0], dict):
        return [(e["i"], e["j"]) for e in consensus]
    elif isinstance(consensus, list):
        return [tuple(e) for e in consensus if isinstance(e, (list, tuple)) and len(e) == 2]
    return []


# ============================================================================
# SECCIÓN 84 · CONSTRUCCIÓN DE LA LISTA CANÓNICA (PARTE 11)
# ============================================================================
@dataclass
class SignificantEdge:
    i: int
    j: int
    label_i: str
    label_j: str
    categories: List[str]
    source_paradigms: List[str]

    @property
    def label(self) -> str:
        return f"{self.label_i}→{self.label_j}"


def build_significant_edges(jsons: Dict[str, Dict]) -> List[SignificantEdge]:
    causal = extract_causal_edges(jsons.get("TRI"))
    hidden = extract_hidden_structure_edges(jsons.get("TRI"))
    sinks = extract_attention_sinks(jsons.get("TRI"))
    shortcuts = extract_shortcut_edges(jsons.get("TRI"))
    top_freq = extract_top_freq_edges(jsons.get("P1"), top_n=15)
    consensus = extract_consensus_edges(jsons.get("P5"))

    print(f"\n  Aristas extraídas:")
    print(f"    Relación Estructural Fuerte (TRI): {len(causal)}")
    print(f"    Estructura Latente Oculta (TRI): {len(hidden)}")
    print(f"    Attention Sinks (TRI): {len(sinks)}")
    print(f"    Atajos predictivos (TRI): {len(shortcuts)}")
    print(f"    Top frecuentes (P1): {len(top_freq)}")
    print(f"    Grafo consenso (P5): {len(consensus)}")

    edge_categories: Dict[Tuple[int,int], Tuple[set, set]] = {}
    def add_edge(edge, category, paradigm):
        if edge not in edge_categories:
            edge_categories[edge] = (set(), set())
        edge_categories[edge][0].add(category)
        edge_categories[edge][1].add(paradigm)

    for e in causal:
        add_edge(e, "causal", "TRI")
    for e in hidden:
        add_edge(e, "hidden_structure", "TRI")
    for e in sinks:
        add_edge(e, "sink", "TRI")
    for e in shortcuts:
        add_edge(e, "shortcut", "TRI")
    for e in top_freq:
        add_edge(e, "top_freq", "P1")
    for e in consensus:
        add_edge(e, "consensus", "P5")

    edges = []
    for (i, j), (cats, paradigms) in edge_categories.items():
        edges.append(SignificantEdge(
            i=i, j=j,
            label_i=INDEX_NAMES[i], label_j=INDEX_NAMES[j],
            categories=sorted(cats),
            source_paradigms=sorted(paradigms),
        ))

    priority = {"causal": 0, "hidden_structure": 1, "sink": 2, "shortcut": 3,
                "top_freq": 4, "consensus": 5}
    edges.sort(key=lambda e: (min(priority[c] for c in e.categories), -len(e.categories)))

    return edges

def main():
    print(f"\n{'═'*82}")
    print(f" EXTRACTOR DINÁMICO DE ARISTAS SIGNIFICATIVAS")
    print(f"{'═'*82}")

    jsons = find_all_paradigm_jsons()
    edges = build_significant_edges(jsons)

    print(f"\n  ✅ Total aristas significativas a probar: {len(edges)}")
    print(f"\n  Lista canónica (ordenada por prioridad):")
    print(f"  {'#':>3s} {'Arista':22s} {'Categorías':35s} {'Fuentes':15s}")
    print(f"  {'─'*80}")
    for idx, e in enumerate(edges, 1):
        cats = ", ".join(e.categories)
        sources = ", ".join(e.source_paradigms)
        print(f"  {idx:3d} {e.label:22s} {cats:35s} {sources:15s}")

    os.makedirs(BASE_DIR, exist_ok=True)
    out = {
        "total_edges": len(edges),
        "index_names": INDEX_NAMES,
        "edges": [asdict(e) for e in edges],
        "sources_found": {k: bool(v) for k, v in jsons.items()},
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  ✅ Guardado: {OUTPUT_PATH}")

    print(f"\n  RESUMEN POR CATEGORÍA:")
    for cat in ["causal", "hidden_structure", "sink", "shortcut", "top_freq", "consensus"]:
        n = sum(1 for e in edges if cat in e.categories)
        print(f"    {cat:15s}: {n} aristas")

    print(f"\n{'═'*82}\n")
    return edges

EJECUTAR_PARTE11: bool = True

if __name__ == "__main__" and EJECUTAR_PARTE11:
    SIGNIFICANT_EDGES = main()
    N_SIG_EDGES = len(SIGNIFICANT_EDGES)
    print(f"Variable global SIGNIFICANT_EDGES: {N_SIG_EDGES} aristas listas para usar.")


plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# SHP_PATH, SIG_EDGES_PATH y DIR_M2 (OUTPUT_DIR_M2) ya quedaron definidos en
# la SECCIÓN 0. Fallback adicional para el shapefile por si vive en upload/:
if not os.path.exists(SHP_PATH):
    for p in [os.path.join(RUTA_PROYECTO, "upload", "viveros_2024_h19.shp"),
              os.path.join(os.getcwd(), "viveros_2024_h19.shp")]:
        if os.path.exists(p):
            SHP_PATH = p
            break

os.makedirs(OUTPUT_DIR_M2, exist_ok=True)

@dataclass
class SpeciesProfileReal:
    name: str
    sp1_matches: List[str]
    water_need: str
    water_need_score: int
    color: str

SPECIES_TARGETS_REAL = [
    SpeciesProfileReal("Palto", ["Palto"], "Muy alta", 5, "#2ecc71"),
    SpeciesProfileReal("Cerezo", ["Cerezo", "Cerezo Dulce"], "Media-alta", 4, "#e74c3c"),
    SpeciesProfileReal("Vid", ["Vid Vinífera", "Vid de Mesa", "Vid",
                                "Vid Pisquera"], "Media", 3, "#9b59b6"),
    SpeciesProfileReal("Arándano", ["Arándano"], "Alta", 4, "#3498db"),
    SpeciesProfileReal("Manzano", ["Manzano"], "Media", 3, "#f39c12"),
    SpeciesProfileReal("Olivo", ["Olivo"], "Baja (xerófito)", 1, "#95a5a6"),
    SpeciesProfileReal("Limonero", ["Limonero"], "Alta (cítrico)", 4, "#16a085"),
    SpeciesProfileReal("Naranjo", ["Naranjo"], "Alta (cítrico)", 4, "#e67e22"),
    SpeciesProfileReal("Nogal", ["Nogal"], "Media-alta", 4, "#7f8c8d"),
    SpeciesProfileReal("Duraznero", ["Duraznero"], "Media", 3, "#d35400"),
    SpeciesProfileReal("Peral", ["Peral"], "Media", 3, "#bdc3c7"),
    SpeciesProfileReal("Frambueso", ["Frambueso"], "Alta (arbusto)", 4, "#c0392b"),
]

def download_stack_at_coords(coords, start_date="2018-01-01",
                              end_date="2025-12-31",
                              tamano_area=TAMANO_AREA, max_cloud=60):
    if not HAS_EE:
        return None
    try:
        ee.Initialize(project="ee-patricio21")
    except Exception:
        try:
            init_earth_engine("ee-patricio21")
        except Exception:
            return None
    aoi = calcular_area_interes(coords, tamano_area)
    coleccion = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                 .filterBounds(aoi).filterDate(start_date, end_date)
                 .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
                 .map(mask_clouds_qa60_scl))
    tile = obtener_tile_mas_frecuente(coleccion)
    if tile:
        coleccion = coleccion.filter(ee.Filter.eq("MGRS_TILE", tile))
    coleccion_indices = coleccion.map(compute_12_indices)
    n = coleccion_indices.size().getInfo()
    if n == 0:
        return None
    from concurrent.futures import ThreadPoolExecutor, as_completed
    image_list = coleccion_indices.toList(coleccion_indices.size())
    results = [None] * n
    def proc(idx):
        img = ee.Image(image_list.get(idx))
        return descargar_indices(img, aoi, 10), idx
    with ThreadPoolExecutor(max_workers=min(4, n)) as ex:
        futures = {ex.submit(proc, i): i for i in range(n)}
        for f in as_completed(futures):
            arr, idx = f.result()
            if arr is not None:
                results[idx] = arr
    valid = [r for r in results if r is not None]
    if not valid:
        return None
    ref = valid[0].shape
    valid = [r for r in valid if r.shape == ref]
    return np.stack(valid, axis=0).astype(np.float32)


def train_models_on_stack(stack, n_models, seed_base=42, label=""):
    if not HAS_TORCH:
        return []
    (X_tr, Y_tr), (X_va, Y_va), img_size, scaler = process_indices_data(
        stack, seq_length=SEQ_LENGTH)
    X_val_sample = torch.from_numpy(
        np.ascontiguousarray(np.transpose(X_va[:16], (0, 4, 1, 2, 3)))
    ).float()
    matrices = []
    t0 = time.time()
    for i in range(n_models):
        rollout, val_loss, _ = train_one_model(stack, seed_base + i, X_val_sample)
        matrices.append(rollout)
        elapsed = time.time() - t0
        rem = elapsed / (i + 1) * (n_models - i - 1)
        print(f"\r  {label} {i+1}/{n_models} (loss={val_loss:.4f}, "
              f"~{rem:.0f}s rest)", end="", flush=True)
    print()
    return matrices


def compute_edge_strengths(matrices, edges):
    return {e: [float(A[e[0], e[1]]) for A in matrices] for e in edges}


def compute_edge_frequencies(matrices, edges, p_top: Optional[float] = None):
    """Φ(e): frecuencia de cada arista sobre las M matrices, discretizando con
    Top-P (antes usaba Top-K con DEFAULT_K)."""
    M = len(matrices)
    counter = Counter()
    p_top = p_efectivo(p_top)
    for A in matrices:
        for e in discretizar(A, p_top):
            counter[e] += 1
    return {e: counter.get(e, 0) / max(M, 1) for e in edges}
def plot_species_map(gdf, species_targets, output_path):
    fig, ax = plt.subplots(figsize=(13, 14))
    classified = set()
    for sp in species_targets:
        sub = gdf[gdf["sp_1"].isin(sp.sp1_matches) |
                   gdf["sp_2"].isin(sp.sp1_matches) |
                   gdf["sp_3"].isin(sp.sp1_matches)]
        classified.update(sub.index.tolist())
    other = gdf.loc[~gdf.index.isin(classified)]
    if len(other) > 0:
        other.plot(ax=ax, color="#ecf0f1", markersize=2, alpha=0.2)
    for sp in species_targets:
        sub = gdf[gdf["sp_1"].isin(sp.sp1_matches) |
                   gdf["sp_2"].isin(sp.sp1_matches) |
                   gdf["sp_3"].isin(sp.sp1_matches)]
        if len(sub) > 0:
            sub.plot(ax=ax, color=sp.color, markersize=12, alpha=0.8,
                     edgecolors="black", linewidths=0.3, label=f"{sp.name} ({len(sub)})")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.set_title("Método 2 (v4 real) — 12 especies objetivo\nEntrenamiento real sobre Sentinel-2",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_strength_heatmap(species_results, output_path):
    names = list(species_results.keys())
    n_sp = len(names)
    n_e = len(SIG_EDGES)
    matrix = np.zeros((n_e, n_sp))
    for j, s in enumerate(names):
        for i, e in enumerate(SIG_EDGES):
            matrix[i, j] = species_results[s]["strengths"][str(e)]["mean"]
    order = np.argsort(-matrix.mean(axis=1))
    fig, ax = plt.subplots(figsize=(14, max(8, n_e * 0.3)))
    im = ax.imshow(matrix[order], aspect="auto", cmap="YlOrRd",
                    vmin=0, vmax=max(matrix.max(), 0.3))
    ax.set_xticks(range(n_sp))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_e))
    ax.set_yticklabels([SIG_LABELS[i] for i in order], fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax.set_title("Fuerza de atención por arista × especie (entrenamiento real)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_kruskal_per_edge(species_results, output_path):
    names = list(species_results.keys())
    n_e = len(SIG_EDGES)
    H_vals = []
    p_vals = []
    for e in SIG_EDGES:
        groups = [species_results[s]["strengths"][str(e)]["values"] for s in names]
        try:
            H, p = stats.kruskal(*groups)
        except Exception:
            H, p = float('nan'), float('nan')
        H_vals.append(H if not np.isnan(H) else 0)
        p_vals.append(p if not np.isnan(p) else 1)
    order = np.argsort(-np.array(H_vals))
    colors = []
    for e in sig_data["edges"]:
        if "causal" in e["categories"]: colors.append("#27ae60")
        elif "sink" in e["categories"]: colors.append("#9b59b6")
        elif "shortcut" in e["categories"]: colors.append("#f39c12")
        else: colors.append("#3498db")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    x = np.arange(n_e)
    axes[0].bar(x, np.array(H_vals)[order], color=colors, alpha=0.85,
                edgecolor="black", linewidth=0.5)
    axes[0].set_ylabel("Kruskal-Wallis H")
    axes[0].set_title("Diferencias entre especies por arista (entrenamiento real)",
                      fontsize=12, fontweight="bold")
    axes[0].grid(True, alpha=0.3, axis="y")
    neg_log_p = -np.log10(np.maximum(np.array(p_vals), 1e-300))
    axes[1].bar(x, neg_log_p[order], color=colors, alpha=0.7)
    axes[1].axhline(y=-np.log10(ALPHA), color="red", linestyle="--", label=f"α={ALPHA}")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([SIG_LABELS[i] for i in order], rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("-log₁₀(p-valor)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")
    legend = [
        Patch(facecolor="#27ae60", label="Causal"),
        Patch(facecolor="#9b59b6", label="Sink"),
        Patch(facecolor="#f39c12", label="Shortcut"),
        Patch(facecolor="#3498db", label="Top/Consenso"),
    ]
    axes[0].legend(handles=legend, loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
def run_method2_real():
    print(f"\n{'═'*82}")
    print(f" MÉTODO 2 (v4 real) — 12 ESPECIES × {N_MODELS_PER_SPECIES} MODELOS REALES")
    print(f"{'═'*82}")

    print(f"\n[1/4] Cargando catastro…")
    if not shapefile_disponible():
        print(f"  Sube viveros_2024_h19.shp/.shx/.dbf/.prj/.cpg a "
              f"{RUTA_PROYECTO} y vuelve a correr esta parte.")
        return
    gdf = gpd.read_file(SHP_PATH)
    print(f"  ✅ {len(gdf)} viveros")

    plot_species_map(gdf, SPECIES_TARGETS_REAL,
                      os.path.join(OUTPUT_DIR_M2, "species_map.png"))

    print(f"\n[2/4] Descargando stacks y entrenando modelos por especie…")
    from pyproj import Transformer
    inv_tr = Transformer.from_crs("EPSG:32719", "EPSG:4326", always_xy=True)

    species_results = {}
    t_total = time.time()

    for sp_idx, sp in enumerate(SPECIES_TARGETS_REAL):
        print(f"\n{'─'*70}")
        print(f"  [{sp_idx+1}/12] {sp.name} (demanda hídrica: {sp.water_need})")
        print(f"{'─'*70}")

        sub = gdf[gdf["sp_1"].isin(sp.sp1_matches) |
                   gdf["sp_2"].isin(sp.sp1_matches) |
                   gdf["sp_3"].isin(sp.sp1_matches)]
        if len(sub) == 0:
            print(f"  ⚠ Sin viveros — omitiendo")
            continue
        print(f"  Viveros disponibles: {len(sub)}")

        vivero = sub.iloc[0]
        v_x, v_y = float(vivero.geometry.x), float(vivero.geometry.y)
        v_lon, v_lat = inv_tr.transform(v_x, v_y)
        print(f"  Vivero seleccionado: ({v_lon:.4f}, {v_lat:.4f})")

        print(f"  Descargando stack Sentinel-2…")
        stack = download_stack_at_coords([v_lon, v_lat], tamano_area=TAMANO_AREA)
        if stack is None:
            print(f"  ⚠ Sin imágenes — omitiendo")
            continue
        print(f"  Stack shape: {stack.shape}")

        print(f"  Entrenando {N_MODELS_PER_SPECIES} modelos…")
        seed_base = 42 + sp_idx * 1000
        mats = train_models_on_stack(stack, N_MODELS_PER_SPECIES,
                                      seed_base=seed_base,
                                      label=f"  {sp.name}")
        if not mats:
            continue

        str_dict = compute_edge_strengths(mats, SIG_EDGES)
        freq_dict = compute_edge_frequencies(mats, SIG_EDGES)

        strengths_clean = {}
        for e in SIG_EDGES:
            vals = str_dict[e]
            strengths_clean[str(e)] = {
                "values": vals,
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }
        species_results[sp.name] = {
            "n_viveros": int(len(sub)),
            "water_need": sp.water_need,
            "water_need_score": sp.water_need_score,
            "color": sp.color,
            "vivero_coords": [float(v_lon), float(v_lat)],
            "strengths": strengths_clean,
            "freqs": {str(e): float(freq_dict[e]) for e in SIG_EDGES},
        }

    elapsed = time.time() - t_total
    print(f"\n  ⏱ Tiempo total: {elapsed/60:.1f} min")

    print(f"\n[3/4] Tests Kruskal-Wallis por arista…")
    names = list(species_results.keys())
    per_edge_tests = {}
    for edge in SIG_EDGES:
        groups = [species_results[s]["strengths"][str(edge)]["values"] for s in names]
        try:
            H, p = stats.kruskal(*groups)
        except Exception:
            H, p = float('nan'), float('nan')
        if "Palto" in species_results and "Olivo" in species_results:
            v_p = species_results["Palto"]["strengths"][str(edge)]["values"]
            v_o = species_results["Olivo"]["strengths"][str(edge)]["values"]
            try:
                u, p_mwu = stats.mannwhitneyu(v_p, v_o, alternative="greater")
            except Exception:
                u, p_mwu = float('nan'), float('nan')
        else:
            u, p_mwu = float('nan'), float('nan')
        per_edge_tests[str(edge)] = {
            "label": INDEX_NAMES[edge[0]] + "→" + INDEX_NAMES[edge[1]],
            "H_kruskal": float(H) if not np.isnan(H) else None,
            "p_kruskal": float(p) if not np.isnan(p) else None,
            "p_mannwhitney_palto_vs_olivo": float(p_mwu) if not np.isnan(p_mwu) else None,
        }

    sorted_t = sorted(per_edge_tests.items(),
                       key=lambda x: -(x[1]["H_kruskal"] or 0))
    print(f"\n  Top 10 aristas con mayor variabilidad entre especies:")
    for e_str, t in sorted_t[:10]:
        H = t["H_kruskal"] or 0
        p = t["p_kruskal"] or 1
        p_mwu = t["p_mannwhitney_palto_vs_olivo"] or 1
        print(f"    {t['label']:22s} H={H:>8.2f}  p={p:.4e}  p_MWU(P>O)={p_mwu:.4e}")

    print(f"\n[4/4] Generando visualizaciones…")
    plot_strength_heatmap(species_results,
                           os.path.join(OUTPUT_DIR_M2, "strength_heatmap.png"))
    plot_kruskal_per_edge(species_results,
                           os.path.join(OUTPUT_DIR_M2, "kruskal_wallis.png"))

    clean = {}
    for s, r in species_results.items():
        clean[s] = {**r,
                    "strengths": {e: {"mean": v["mean"], "std": v["std"]}
                                   for e, v in r["strengths"].items()}}
    results = {
        "method": "Method 2 v4 real — real training per species",
        "n_species": len(species_results),
        "n_models_per_species": N_MODELS_PER_SPECIES,
        "total_time_min": elapsed / 60,
        "n_significant_edges_tested": N_SIG,
        "species": clean,
        "per_edge_tests": per_edge_tests,
    }
    with open(os.path.join(OUTPUT_DIR_M2, "method2_v4_real_results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    n_sig = sum(1 for t in per_edge_tests.values()
                 if t["p_kruskal"] is not None and t["p_kruskal"] < ALPHA)
    print(f"\n{'═'*82}")
    print(f" RESUMEN — MÉTODO 2 (v4 real)")
    print(f"{'═'*82}")
    print(f"  • Especies: {len(species_results)}")
    print(f"  • Modelos por especie: {N_MODELS_PER_SPECIES}")
    print(f"  • Tiempo: {elapsed/60:.1f} min")
    print(f"  • Aristas con diferencias significativas: {n_sig}/{N_SIG}")
    print(f"{'═'*82}\n")

    return results

EJECUTAR_PARTE12: bool = False  # Método 2 (especies CIREN) — 60 entrenamientos, no relacionado con los 5 paradigmas; activar explícito si se necesita

if __name__ == "__main__" and EJECUTAR_PARTE12:
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        HAS_GEO = True
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "geopandas"])
        import geopandas as gpd
        HAS_GEO = True
    try:
        import ee
        HAS_EE = True
    except ImportError:
        HAS_EE = False
    try:
        import torch
        HAS_TORCH = True
    except ImportError:
        HAS_TORCH = False
    with open(SIG_EDGES_PATH, "r") as f:
        sig_data = json.load(f)
    SIG_EDGES = [(e["i"], e["j"]) for e in sig_data["edges"]]
    SIG_LABELS = [f"{e['label_i']}→{e['label_j']}" for e in sig_data["edges"]]
    N_SIG = len(SIG_EDGES)
    print(f"  ✅ {N_SIG} aristas significativas cargadas")
    total_models = 12 * N_MODELS_PER_SPECIES
    print(f"  📊 Config: 12 especies × {N_MODELS_PER_SPECIES} modelos = {total_models} entrenamientos")
    print(f"     Tiempo estimado: ~{total_models * 10 / 60:.0f} min en GPU T4")
    results_m2_real = run_method2_real()