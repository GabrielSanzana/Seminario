#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests de la particion por fase y del sobremuestreo.

POR QUE EXISTE ESTE ARCHIVO
---------------------------
El control de exposicion de entrenar_balanceado.py fallo dos veces seguidas por
errores que ningun resultado dejaba ver:

  1. la particion train/val era cronologica, asi que el bloque de validacion
     tenia 17 fechas de crecimiento y 14 de maduracion contra 1 de dormancia y
     1 de postcosecha, y el criterio de parada quedaba sesgado justo contra lo
     que el experimento medía;

  2. el arreglo de esa particion tenia una rama, va.append(idx[-1:]), que
     metia la misma fecha en train y en val cuando una fase se quedaba corta.

Ninguno de los dos aparece en las cifras de salida: el entrenamiento corre, el
val baja, los checkpoints se guardan. Solo se ven mirando la composicion de los
bloques. Por eso las propiedades que tienen que cumplirse se comprueban aqui y
no a ojo.

Cada test contesta una pregunta que, si se contesta mal, invalida una cifra del
README sin dar ningun error.

USO
    .venv/bin/python test_particion.py
"""
from __future__ import annotations

import os
import sys
import traceback

import numpy as np

sys.path.insert(0, "/srv/pfigueroa")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import entrenar_balanceado as EB  # noqa: E402
from analisis_fenologico import FASES  # noqa: E402

# ---------------------------------------------------------------- infraestructura

FALLOS = []
CORRIDOS = []


def prueba(fn):
    """Corre un test, captura el fallo y sigue. Un fallo no puede esconder
    a los demas: la primera version de este archivo usaba asserts sueltos y el
    primer error dejaba sin correr todo lo que venia detras."""
    nombre = fn.__name__.replace("t_", "").replace("_", " ")
    try:
        detalle = fn()
        CORRIDOS.append((nombre, True, detalle or ""))
    except Exception as e:  # noqa: BLE001
        CORRIDOS.append((nombre, False, str(e)))
        FALLOS.append((nombre, traceback.format_exc()))
    return fn


def igual(a, b, msg=""):
    if a != b:
        raise AssertionError("%s: %r != %r" % (msg or "distinto", a, b))


def cierto(cond, msg):
    if not cond:
        raise AssertionError(msg)


def calendario(cuentas, n_anios=3):
    """Meses de una serie de escenas, EN ORDEN CRONOLOGICO.

    El orden importa y no es un detalle: el sesgo que se esta probando nace de
    que el ultimo 20% de una serie temporal cae en los meses del final del
    recorrido, no en una muestra representativa. Un calendario con las fechas
    barajadas haria pasar los tests sin que prueben nada, porque un corte al
    final de una lista barajada es una muestra al azar.

    Cada fase reparte sus fechas entre las veces que sus meses aparecen a lo
    largo de n_anios, y al final todo se ordena por instante.
    """
    puntos = []
    for nombre, ms in FASES:
        n = cuentas.get(nombre, 0)
        if not n:
            continue
        activos = sorted(anio * 12 + (mes - 1)
                         for anio in range(n_anios) for mes in ms)
        for k in range(n):
            # varias escenas dentro del mismo mes se separan con un desfase
            # pequeno para que el orden quede definido
            puntos.append((activos[k % len(activos)]
                           + 0.001 * (k // len(activos)),
                           ms[k % len(ms)]))
    puntos.sort()
    return np.array([m for _t, m in puntos])


REAL = {n: c for n, c in [("dormancia (may-ago)", 8), ("brotacion (sep-oct)", 17),
                          ("crecimiento (nov-dic)", 40),
                          ("maduracion (ene-mar)", 76),
                          ("postcosecha (abr)", 7)]}

# ------------------------------------------------------------------- las fases


@prueba
def t_fases_particionan_el_ano():
    """Si un mes no esta en FASES, sus frames desaparecen del entrenamiento sin
    aviso, porque _indices_balanceados concatena solo los grupos que encuentra.
    """
    vistos = []
    for _n, ms in FASES:
        vistos += list(ms)
    igual(sorted(vistos), list(range(1, 13)), "FASES no cubre los 12 meses")
    igual(len(vistos), len(set(vistos)), "hay meses repetidos entre fases")
    return "12 meses, sin solape"


@prueba
def t_mes_no_cubierto_es_error():
    """Un mes fuera de FASES tiene que reventar, no perderse en silencio."""
    try:
        EB._cobertura_fases(np.array([1, 2, 99]))
    except ValueError as e:
        cierto("99" in str(e), "el error no dice que mes falta")
        return "levanta ValueError y nombra el mes"
    raise AssertionError("un mes no cubierto paso sin error")


# -------------------------------------------------------- particion estratificada


@prueba
def t_split_sin_fuga():
    """Ninguna fecha puede estar en train y en val. Este es el bug que tenia
    la rama va.append(idx[-1:])."""
    for cuentas in (REAL, {n: 5 for n in REAL}, dict(REAL, **{
            "postcosecha (abr)": 3})):
        m = calendario(cuentas)
        tr, va = EB._split_estratificado(m)
        igual(int(np.intersect1d(tr, va).size), 0, "fuga train/val")
    return "3 calendarios, interseccion vacia"


@prueba
def t_split_cubre_todo():
    m = calendario(REAL)
    tr, va = EB._split_estratificado(m)
    igual(len(tr) + len(va), len(m), "faltan o sobran fechas")
    igual(sorted(np.concatenate([tr, va]).tolist()), list(range(len(m))),
          "la union no son todos los indices")
    return "%d train + %d val = %d" % (len(tr), len(va), len(m))


@prueba
def t_split_da_val_a_cada_fase():
    """El punto entero del arreglo: que las cinco fases pesen en el criterio de
    parada. Si alguna se queda con cero fechas de val, el sesgo sigue ahi."""
    m = calendario(REAL)
    _tr, va = EB._split_estratificado(m)
    comp = EB._composicion(m[va])
    for nombre, n in comp.items():
        cierto(n >= 1, "la fase " + nombre + " se quedo sin val")
    return str(comp)


@prueba
def t_split_es_proporcional():
    """Cada fase cede alrededor del 20%, no una fraccion que dependa de su
    tamano. Con tolerancia de una fecha, que es lo que da el redondeo."""
    m = calendario(REAL)
    _tr, va = EB._split_estratificado(m, frac_val=0.2)
    comp = EB._composicion(m[va])
    for nombre, ms in FASES:
        n_tot = int(np.isin(m, ms).sum())
        esperado = min(max(round(0.2 * n_tot), 1), n_tot - 1)
        cierto(abs(comp[nombre] - esperado) <= 1,
               "%s cede %d de %d, se esperaba %d"
               % (nombre, comp[nombre], n_tot, esperado))
    return "todas dentro de +-1 fecha del 20%"


@prueba
def t_split_respeta_el_tiempo_dentro_de_fase():
    """El val de una fase tiene que ser posterior a su train. Con muestreo
    aleatorio una fecha de train y su vecina de val serian casi la misma
    escena, y el val dejaria de medir generalizacion."""
    m = calendario(REAL)
    tr, va = EB._split_estratificado(m)
    for _nombre, ms in FASES:
        t = tr[np.isin(m[tr], ms)]
        v = va[np.isin(m[va], ms)]
        if len(t) and len(v):
            cierto(t.max() < v.min(),
                   "en " + _nombre + " el val no va despues del train")
    return "val posterior al train en las cinco fases"


@prueba
def t_fase_corta_no_se_duplica():
    """Regresion del bug de fuga: una fase con 1 o 2 fechas va entera a train y
    se declara, en vez de aparecer en los dos lados."""
    m = calendario(dict(REAL, **{"postcosecha (abr)": 1}))
    tr, va = EB._split_estratificado(m, min_fechas=3)
    igual(int(np.intersect1d(tr, va).size), 0, "la fase corta se duplico")
    igual(EB._composicion(m[va])["postcosecha (abr)"], 0,
          "la fase corta no debia aportar val")
    igual(EB._composicion(m[tr])["postcosecha (abr)"], 1,
          "la fase corta tiene que estar entera en train")
    cierto(any(n == "postcosecha (abr)" for n, _c in EB.OMITIDAS_DEL_VAL),
           "la omision no quedo declarada en OMITIDAS_DEL_VAL")
    return "fase de 1 fecha: entera a train, declarada"


@prueba
def t_ningun_bloque_queda_vacio():
    """Con frac_val extremos el corte no puede vaciar un lado."""
    m = calendario(REAL)
    for frac in (0.01, 0.5, 0.99):
        tr, va = EB._split_estratificado(m, frac_val=frac)
        cierto(len(tr) > 0 and len(va) > 0, "bloque vacio con frac=%g" % frac)
        igual(int(np.intersect1d(tr, va).size), 0, "fuga con frac=%g" % frac)
    return "frac_val de 0.01 a 0.99 sin bloque vacio"


def _fraccion_en_val(m, va):
    """Que parte de cada fase cae en validacion.

    Es la cifra que decide, y no el numero absoluto: la perdida de validacion
    es un promedio sobre sus frames, asi que lo que sesga el criterio de parada
    es que unas fases entreguen la mitad de sus fechas y otras un decimo. Con
    una particion estratificada todas entregan la misma fraccion por
    construccion.
    """
    out = {}
    for nombre, ms in FASES:
        n_tot = int(np.isin(m, ms).sum())
        if n_tot:
            out[nombre] = float(np.isin(m[va], ms).sum()) / n_tot
    return out


@prueba
def t_el_split_cronologico_si_esta_sesgado():
    """El test que justifica todo el arreglo, sobre un calendario ordenado en
    el tiempo. El corte cronologico entrega al val fracciones muy distintas de
    cada fase; el estratificado entrega la misma de todas.

    Si este test empieza a fallar hay que mirar el calendario antes que el
    codigo: mide una propiedad del reparto de fechas."""
    m = calendario(REAL)
    n_tr = int(round(0.8 * len(m)))
    f_cron = _fraccion_en_val(m, np.arange(n_tr, len(m)))
    _tr, va = EB._split_estratificado(m)
    f_estr = _fraccion_en_val(m, va)
    des = lambda f: max(f.values()) / max(min(f.values()), 1e-9)  # noqa: E731
    cierto(des(f_cron) >= 2.5,
           "el corte cronologico deberia repartir muy desigual, dio %.1fx"
           % des(f_cron))
    cierto(des(f_estr) <= 2.0,
           "el estratificado deberia repartir parejo, dio %.1fx" % des(f_estr))
    cierto(des(f_estr) < des(f_cron), "el arreglo no mejora el reparto")
    return ("fraccion de cada fase que llega al val: cronologico %.1fx de "
            "desigualdad, estratificado %.1fx" % (des(f_cron), des(f_estr)))


# ------------------------------------------------------------- sobremuestreo


@prueba
def t_balanceo_iguala_las_fases():
    m = calendario(REAL)
    idx = EB._indices_balanceados(m)
    comp = EB._composicion(m[idx])
    igual(len(set(comp.values())), 1,
          "las fases no quedaron con el mismo peso: " + str(comp))
    igual(max(comp.values()), max(REAL.values()),
          "el objetivo no es el tamano de la fase mayor")
    return "cinco fases a %d frames cada una" % max(comp.values())


@prueba
def t_balanceo_no_pierde_frames():
    """Sobremuestrear es anadir copias. Si un indice original desaparece es que
    el recorte [:objetivo] se comio un grupo."""
    m = calendario(REAL)
    idx = EB._indices_balanceados(m)
    igual(sorted(set(idx.tolist())), list(range(len(m))),
          "el sobremuestreo perdio frames")
    return "los %d frames originales siguen presentes" % len(m)


@prueba
def t_balanceo_no_inventa_indices():
    m = calendario(REAL)
    idx = EB._indices_balanceados(m)
    cierto(idx.min() >= 0 and idx.max() < len(m), "indice fuera de rango")
    return "rango [0, %d)" % len(m)


@prueba
def t_balanceo_reporta_los_factores():
    m = calendario(REAL)
    rep = {}
    EB._indices_balanceados(m, rep)
    for nombre, v in rep.items():
        igual(v["originales"], REAL[nombre], "originales mal contados")
        esperado = round(max(REAL.values()) / REAL[nombre], 2)
        igual(v["factor"], esperado, "factor mal calculado para " + nombre)
    return "factores: " + ", ".join("%s x%.2f" % (n[:4], v["factor"])
                                    for n, v in rep.items())


@prueba
def t_balanceo_es_determinista():
    m = calendario(REAL)
    a = EB._indices_balanceados(m)
    b = EB._indices_balanceados(m)
    cierto(np.array_equal(a, b), "dos llamadas dan resultados distintos")
    return "dos llamadas identicas"


@prueba
def t_los_dos_reportes_no_se_pisan():
    """El bug de reporte: la version anterior tenia un solo diccionario global
    y el balanceo del val sobrescribia el del train, asi que balanceo.json
    mentia sobre lo que se le hizo al entrenamiento."""
    m_tr = calendario(REAL)
    m_va = calendario({n: max(1, c // 5) for n, c in REAL.items()})
    rt, rv = {}, {}
    EB._indices_balanceados(m_tr, rt)
    EB._indices_balanceados(m_va, rv)
    cierto(rt != rv, "los dos reportes salieron iguales")
    igual(rt["dormancia (may-ago)"]["originales"], REAL["dormancia (may-ago)"],
          "el reporte de train quedo pisado por el de val")
    return "train y val reportan por separado"


# ------------------------------------------------- el parche de carga completo


def _fake_pid(n_tr, n_va, dim=3):
    """Sustituto de process_indices_data que devuelve frames identificables:
    el frame k lleva el valor k, asi que se puede seguir a donde va cada uno."""
    def pid(stack, seq_length=1, dates_millis=None, **kw):
        X = np.arange(n_tr + n_va, dtype=np.float32)[:, None].repeat(dim, 1)
        return (X[:n_tr], None), (X[n_tr:], None), (1, 1), None
    return pid


class _Ctx:
    """Pone las globales de EB y las devuelve como estaban."""

    def __init__(self, **kw):
        self.kw = kw

    def __enter__(self):
        self.old = {k: getattr(EB, k) for k in
                    ("MESES", "BALANCEAR", "ESTRATIFICADO", "BALANCEAR_VAL",
                     "_ORIG_PID")}
        for k, v in self.kw.items():
            setattr(EB, k, v)
        return self

    def __exit__(self, *e):
        for k, v in self.old.items():
            setattr(EB, k, v)
        return False


@prueba
def t_desalineacion_frames_fechas_es_error():
    """Con SEQ_LENGTH > 1 process_indices_data devuelve ventanas y el frame k
    ya no es la fecha k. Sin esta comprobacion cada frame quedaria etiquetado
    con el mes de otra fecha y el balanceo seria falso sin dar ningun error."""
    m = calendario(REAL)
    with _Ctx(MESES=m, BALANCEAR=True, ESTRATIFICADO=False,
              BALANCEAR_VAL=False, _ORIG_PID=_fake_pid(100, 20)):
        try:
            EB._pid_balanceado(None, seq_length=3)
        except ValueError as e:
            cierto("desalineacion" in str(e), "el error no explica la causa")
            return "levanta ValueError con 120 frames y %d fechas" % len(m)
    raise AssertionError("la desalineacion paso sin error")


@prueba
def t_sin_banderas_no_toca_nada():
    """Con todo apagado el parche tiene que ser la identidad, o el brazo base
    dejaria de ser comparable con los publicados antes."""
    m = calendario(REAL)
    n_tr = len(m) - 30
    with _Ctx(MESES=m, BALANCEAR=False, ESTRATIFICADO=False,
              BALANCEAR_VAL=False, _ORIG_PID=_fake_pid(n_tr, 30)):
        (Xtr, _), (Xva, _), _i, _s = EB._pid_balanceado(None)
        igual(len(Xtr), n_tr, "el train cambio de tamano")
        igual(len(Xva), 30, "el val cambio de tamano")
        cierto(np.array_equal(Xtr[:, 0], np.arange(n_tr)),
               "el train cambio de contenido")
    return "identidad exacta"


@prueba
def t_estratificado_no_mezcla_contenido():
    """A nivel de arrays, no solo de indices: ningun frame del val puede
    aparecer tambien en el train."""
    m = calendario(REAL)
    n_tr = len(m) - 30
    with _Ctx(MESES=m, BALANCEAR=False, ESTRATIFICADO=True,
              BALANCEAR_VAL=False, _ORIG_PID=_fake_pid(n_tr, 30)):
        (Xtr, _), (Xva, _), _i, _s = EB._pid_balanceado(None)
        a, b = set(Xtr[:, 0].tolist()), set(Xva[:, 0].tolist())
        igual(len(a & b), 0, "hay frames en los dos bloques")
        igual(len(a) + len(b), len(m), "se perdieron o duplicaron frames")
    return "%d train y %d val, sin solape" % (len(Xtr), len(Xva))


@prueba
def t_balancear_train_no_toca_el_val():
    m = calendario(REAL)
    n_tr = len(m) - 30
    with _Ctx(MESES=m, BALANCEAR=True, ESTRATIFICADO=True,
              BALANCEAR_VAL=False, _ORIG_PID=_fake_pid(n_tr, 30)):
        (Xtr, _), (Xva, _), _i, _s = EB._pid_balanceado(None)
        n_val_sin = len(Xva)
    with _Ctx(MESES=m, BALANCEAR=False, ESTRATIFICADO=True,
              BALANCEAR_VAL=False, _ORIG_PID=_fake_pid(n_tr, 30)):
        (Xtr2, _), (Xva2, _), _i, _s = EB._pid_balanceado(None)
        igual(len(Xva2), n_val_sin, "balancear el train cambio el val")
        cierto(len(Xtr) > len(Xtr2), "el train balanceado no crecio")
    return "train %d -> %d, val intacto en %d" % (len(Xtr2), len(Xtr),
                                                  n_val_sin)


@prueba
def t_val_balanceado_pesa_igual_las_fases():
    """Lo que arregla el sesgo del criterio de parada: despues de balancear, el
    val tiene el mismo numero de frames de cada fase."""
    m = calendario(REAL)
    n_tr = len(m) - 30
    with _Ctx(MESES=m, BALANCEAR=False, ESTRATIFICADO=True,
              BALANCEAR_VAL=True, _ORIG_PID=_fake_pid(n_tr, 30)):
        EB._pid_balanceado(None)
        comp = EB.SPLIT["val_balanceado"]
    igual(len(set(comp.values())), 1,
          "el val balanceado sigue desigual: " + str(comp))
    return "val a %d frames por fase" % max(comp.values())


# ------------------------------------------------------- contra el dato real


@prueba
def t_calendario_real_reproduce_el_sesgo():
    """Con las fechas de verdad, no con un calendario sintetico. Si el dato no
    esta disponible el test se salta en vez de fallar."""
    try:
        import hello as H
        fechas = H.cargar_fechas_stack(len(np.load(H.OUTPUT_NPY, mmap_mode="r")))
    except Exception as e:  # noqa: BLE001
        return "saltado, sin acceso al stack (" + type(e).__name__ + ")"
    import datetime as dt
    m = np.array([dt.datetime.utcfromtimestamp(int(x) / 1000).month
                  for x in fechas])
    n_tr = int(round(0.8 * len(m)))
    cron = EB._composicion(m[n_tr:])
    _tr, va = EB._split_estratificado(m)
    estr = EB._composicion(m[va])
    f_cron = _fraccion_en_val(m, np.arange(n_tr, len(m)))
    f_estr = _fraccion_en_val(m, va)
    d = lambda f: max(f.values()) / max(min(f.values()), 1e-9)  # noqa: E731
    cierto(min(estr.values()) >= 1,
           "con el estratificado alguna fase se quedo sin val: " + str(estr))
    cierto(d(f_estr) < d(f_cron),
           "el estratificado no mejora el reparto del val")
    return ("%d fechas | val cronologico %s (%.1fx desigual) | estratificado "
            "%s (%.1fx)" % (len(m), list(cron.values()), d(f_cron),
                            list(estr.values()), d(f_estr)))


def main() -> int:
    W = 78
    print("=" * W)
    print(" TESTS DE LA PARTICION POR FASE Y DEL SOBREMUESTREO")
    print("=" * W)
    for nombre, ok, detalle in CORRIDOS:
        print("  %s  %-46s %s" % ("ok  " if ok else "FALLA", nombre[:46],
                                  detalle[:80]))
    print("-" * W)
    print("  %d de %d pasan" % (len(CORRIDOS) - len(FALLOS), len(CORRIDOS)))
    for nombre, tb in FALLOS:
        print("\n  FALLA " + nombre + "\n" + tb)
    print("=" * W)
    return 1 if FALLOS else 0


if __name__ == "__main__":
    sys.exit(main())
