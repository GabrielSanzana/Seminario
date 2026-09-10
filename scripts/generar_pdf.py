#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compila tesis/main.tex a tesis/main.pdf.

Hace lo mismo que scripts/compilar.sh pero en Python, para correrlo directo
con "python scripts/generar_pdf.py" sin pasar por bash.

Secuencia de compilacion (el documento usa biblatex con backend biber, no
bibtex ni thebibliography):

    pdflatex -> biber -> pdflatex -> pdflatex

La primera pasada de pdflatex deja el archivo .bcf que biber necesita para
resolver las citas; las dos pasadas finales resuelven las citas ya
procesadas, el indice y las referencias cruzadas (numeros de pagina,
figuras, tablas).

Requiere TinyTeX instalado (o cualquier distribucion de LaTeX con pdflatex
y biber en el PATH). Si no los encuentra ahi, prueba la ruta tipica de
instalacion de TinyTeX en Windows.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_RAIZ = Path(__file__).resolve().parent.parent
CARPETA_TESIS = REPO_RAIZ / "tesis"
ARCHIVO_PRINCIPAL = "main.tex"

# Ruta tipica de TinyTeX en Windows, por si pdflatex/biber no estan en el
# PATH del sistema. TinyTeX se instala por usuario, no por maquina.
RUTA_TINYTEX = Path.home() / "AppData" / "Roaming" / "TinyTeX" / "bin" / "windows"


def resolver_binario(nombre):
    """Busca un ejecutable en el PATH; si no aparece, prueba la ruta de TinyTeX."""
    encontrado = shutil.which(nombre)
    if encontrado:
        return encontrado
    candidato = RUTA_TINYTEX / f"{nombre}.exe"
    if candidato.exists():
        return str(candidato)
    return None


def correr(comando, cwd):
    """Ejecuta un comando y devuelve (codigo_salida, salida_combinada)."""
    resultado = subprocess.run(
        comando,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return resultado.returncode, resultado.stdout


def main():
    pdflatex = resolver_binario("pdflatex")
    biber = resolver_binario("biber")

    if not pdflatex:
        print("FALLO: pdflatex no esta en el PATH ni en la ruta de TinyTeX.", file=sys.stderr)
        print(f"  Se busco en: {RUTA_TINYTEX}", file=sys.stderr)
        print("  Instalar TinyTeX: https://yihui.org/tinytex/", file=sys.stderr)
        return 1

    if not biber:
        print("FALLO: biber no esta en el PATH ni en la ruta de TinyTeX.", file=sys.stderr)
        print("  Instalar con: tlmgr install biber biblatex biblatex-apa", file=sys.stderr)
        return 1

    if not (CARPETA_TESIS / ARCHIVO_PRINCIPAL).exists():
        print(f"FALLO: no se encontro {CARPETA_TESIS / ARCHIVO_PRINCIPAL}", file=sys.stderr)
        return 1

    comando_pdflatex = [pdflatex, "-interaction=nonstopmode", ARCHIVO_PRINCIPAL]
    comando_biber = [biber, "main"]

    print("Compilando (pasada 1/4: pdflatex)...")
    correr(comando_pdflatex, cwd=CARPETA_TESIS)

    print("Resolviendo bibliografia (pasada 2/4: biber)...")
    codigo_biber, salida_biber = correr(comando_biber, cwd=CARPETA_TESIS)
    if codigo_biber != 0:
        print("ADVERTENCIA: biber devolvio un error, revisar main.blg", file=sys.stderr)
        print(salida_biber[-2000:], file=sys.stderr)

    print("Resolviendo citas e indice (pasada 3/4: pdflatex)...")
    correr(comando_pdflatex, cwd=CARPETA_TESIS)

    print("Resolviendo referencias cruzadas (pasada 4/4: pdflatex)...")
    _, salida_final = correr(comando_pdflatex, cwd=CARPETA_TESIS)

    errores = [linea for linea in salida_final.splitlines() if linea.startswith("!")]
    if errores:
        print("FALLO: errores de LaTeX", file=sys.stderr)
        for linea in errores:
            print(f"  {linea}", file=sys.stderr)
        return 1

    coincidencia = re.search(r"Output written on main\.pdf \((\d+) pages", salida_final)
    if not coincidencia:
        print("FALLO: no se genero main.pdf", file=sys.stderr)
        return 1

    paginas = coincidencia.group(1)
    print(f"OK: main.pdf compilado, {paginas} paginas")
    print(f"  -> {CARPETA_TESIS / 'main.pdf'}")

    colgantes = re.findall(
        r"LaTeX Warning: (?:Reference|Citation).*undefined", salida_final
    )
    if colgantes:
        print(f"ADVERTENCIA: {len(colgantes)} referencias o citas sin resolver (saldran como ?? en el PDF)")
        for linea in colgantes:
            print(f"  {linea}")
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
