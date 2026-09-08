"""Genera el informe en DOCX a partir de las fuentes LaTeX.

Aplana los \\include, entrega el resultado a pandoc con citeproc + ieee.csl,
y usa un reference.docx ajustado al formato de la Escuela (Times New Roman 12,
margenes 2,5 cm, sangria 1 cm, interlineado sencillo, sin espacio extra antes
de los encabezados, numero de pagina abajo a la derecha).
"""
import os
import re
import subprocess
import shutil

TESIS = r"C:\Users\patru\Downloads\archivos tesis\Seminario-claude-1\tesis"
SCRATCH = r"C:\Users\patru\AppData\Local\Temp\claude\C--Users-patru-Downloads-archivos-tesis-Seminario-main\3e009d46-31a1-4eb3-a0ec-2a923d717f71\scratchpad"
PANDOC = os.path.join(SCRATCH, "pandoc-3.1.11", "pandoc.exe")
CSL = os.path.join(SCRATCH, "ieee.csl")

ORDEN = [
    "Resumen/resumen.tex",
    "secciones/01_introduccion.tex",
    "secciones/02_objetivos.tex",
    "secciones/03_estado_arte.tex",
    "secciones/04_marco_teorico.tex",
    "secciones/05_plan_trabajo.tex",
    "secciones/06_propuesta.tex",
    "secciones/07_conclusiones.tex",
]


def limpiar(texto):
    """Quita comentarios y comandos que solo tienen sentido en el PDF."""
    salida = []
    for linea in texto.splitlines():
        if linea.lstrip().startswith("%"):
            continue
        salida.append(linea)
    t = "\n".join(salida)
    t = t.replace("\\addcontentsline{toc}{chapter}{\\texorpdfstring{Resumen/Abstract\\vspace{-0.5cm}}{Resumen/Abstract}}", "")
    t = t.replace("\\begingroup", "").replace("\\endgroup", "")
    t = t.replace("\\let\\clearpage\\relax", "")
    t = re.sub(r"\\vspace\{[^}]*\}", "", t)
    t = re.sub(r"\\label\{[^}]*\}", "", t)
    # El tipo de columna P{x} es propio del preambulo del template; pandoc no
    # lo conoce, asi que para el DOCX se reemplaza por columnas simples.
    def simplifica(m):
        spec = m.group(1)
        n = len(re.findall(r"P\{[^}]*\}", spec)) or spec.count("l") or 1
        return "\\begin{longtable}{" + "l" * n + "}"
    t = re.sub(r"\\begin\{longtable\}\{([^\n]*?)\}\s*$", simplifica, t, flags=re.M)
    t = re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}\{", "{", t)
    return t


def construir_tex():
    partes = [
        "\\documentclass{report}",
        "\\usepackage{amsmath,amssymb,booktabs,longtable,array,graphicx}",
        "\\begin{document}",
    ]
    for rel in ORDEN:
        with open(os.path.join(TESIS, rel), encoding="utf-8") as fh:
            partes.append(limpiar(fh.read()))
    partes.append("\\end{document}")
    destino = os.path.join(SCRATCH, "informe_plano.tex")
    with open(destino, "w", encoding="utf-8") as fh:
        fh.write("\n\n".join(partes))
    return destino


def preparar_referencia():
    ref = os.path.join(SCRATCH, "reference.docx")
    with open(ref, "wb") as fh:
        salida = subprocess.run(
            [PANDOC, "--print-default-data-file", "reference.docx"],
            stdout=subprocess.PIPE, check=True)
        fh.write(salida.stdout)
    ajustar_referencia(ref)
    return ref


def ajustar_referencia(ruta):
    """Aplica el formato de la Escuela sobre el reference.docx de pandoc."""
    from docx import Document
    from docx.shared import Pt, Cm
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING

    doc = Document(ruta)

    # Margenes 2,5 cm en todos los lados, tamano carta.
    for sec in doc.sections:
        sec.top_margin = Cm(2.5)
        sec.bottom_margin = Cm(2.5)
        sec.left_margin = Cm(2.5)
        sec.right_margin = Cm(2.5)
        sec.page_width = Cm(21.6)
        sec.page_height = Cm(27.9)

    def fija(nombre, tam, negrita, mayus=False, antes=0, despues=0, sangria=None):
        try:
            estilo = doc.styles[nombre]
        except KeyError:
            return
        fuente = estilo.font
        fuente.name = "Times New Roman"
        fuente.size = Pt(tam)
        fuente.bold = negrita
        if mayus:
            fuente.all_caps = True
        parrafo = estilo.paragraph_format
        parrafo.space_before = Pt(antes)
        parrafo.space_after = Pt(despues)
        parrafo.line_spacing_rule = WD_LINE_SPACING.SINGLE
        if sangria is not None:
            parrafo.first_line_indent = Cm(sangria)

    # Cuerpo: 12 pt, sangria 1 cm, sin espacio entre parrafos, justificado.
    fija("Normal", 12, False, sangria=1.0)
    doc.styles["Normal"].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    fija("Body Text", 12, False, sangria=1.0)
    # Encabezados: capitulo 14 en mayuscula, seccion 14, subseccion 12.
    fija("Heading 1", 14, True, mayus=True, antes=0, despues=8, sangria=0)
    fija("Heading 2", 14, True, antes=10, despues=0, sangria=0)
    fija("Heading 3", 12, True, antes=8, despues=0, sangria=0)
    fija("Title", 18, True, sangria=0)
    fija("Compact", 12, False, sangria=0)
    fija("Author", 12, False, sangria=0)
    fija("Date", 12, False, sangria=0)
    fija("Bibliography", 12, False, sangria=0)

    doc.save(ruta)


def convertir(tex, ref):
    salida = os.path.join(SCRATCH, "Informe_avance.docx")
    cmd = [
        PANDOC, tex,
        "--from", "latex",
        "--to", "docx",
        "--reference-doc", ref,
        "--citeproc",
        "--bibliography", os.path.join(TESIS, "referencias.bib"),
        "--csl", CSL,
        "--toc", "--toc-depth=2",
        "--metadata", "lang=es",
        "-o", salida,
    ]
    subprocess.run(cmd, check=True)
    return salida


if __name__ == "__main__":
    tex = construir_tex()
    ref = preparar_referencia()
    docx = convertir(tex, ref)
    print("OK ->", docx, os.path.getsize(docx), "bytes")
