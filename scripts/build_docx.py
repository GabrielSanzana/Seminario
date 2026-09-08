"""Genera el informe en DOCX a partir de las fuentes LaTeX.

Aplana los \\include, entrega el resultado a pandoc con citeproc + ieee.csl,
y usa un reference.docx ajustado al formato de la Escuela (Times New Roman 12,
margenes 2,5 cm, sangria 1 cm, interlineado sencillo, sin espacio extra antes
de los encabezados, numero de pagina abajo a la derecha).

Formato exigido por "Formato_Informes_Proyecto_Titulo-2024.pdf" (subido por el
autor humano el 2026-09-08) que este script debe seguir siempre:
  - Portada, con imagen institucional, titulo 18pt negrita, autores, nombre de
    la asignatura y fecha. No lleva numero de pagina.
  - Numeracion: paginas previas a la Introduccion en romano minuscula (la
    portada no cuenta), desde la Introduccion en arabigo, ambas alineadas
    abajo a la derecha.
  - Cada capitulo (Heading 1: Introduccion, Objetivos, Estado del arte, Marco
    teorico, Plan de trabajo, Propuesta, Conclusiones) empieza en pagina nueva.
"""
import os
import re
import subprocess
import shutil
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TESIS = os.path.join(SCRIPT_DIR, "..", "tesis")
SCRATCH = os.environ.get("BUILD_DOCX_SCRATCH") or os.path.join(tempfile.gettempdir(), "build_docx")
os.makedirs(SCRATCH, exist_ok=True)

# pandoc: PATH primero, despues la ruta tipica del instalador de winget
# (JohnMacFarlane.Pandoc no siempre queda en el PATH de una sesion de bash).
_pandoc_winget = os.path.join(
    os.environ.get("LOCALAPPDATA", ""), "Pandoc", "pandoc.exe"
)
PANDOC = (
    os.environ.get("PANDOC_BIN")
    or shutil.which("pandoc")
    or (_pandoc_winget if os.path.isfile(_pandoc_winget) else None)
    or os.path.join(SCRATCH, "pandoc-3.1.11", "pandoc.exe")
)

# ieee.csl vive en el repo (scripts/ieee.csl) para no depender de que cada
# maquina lo descargue por su cuenta; ver CLAUDE.md para la fuente original.
_csl_repo = os.path.join(SCRIPT_DIR, "ieee.csl")
CSL = (
    os.environ.get("IEEE_CSL")
    or (_csl_repo if os.path.isfile(_csl_repo) else None)
    or os.path.join(SCRATCH, "ieee.csl")
)

if not os.path.isfile(PANDOC):
    raise SystemExit(
        f"FALLO: no se encontro pandoc en '{PANDOC}'. Instalarlo y/o fijar "
        "PANDOC_BIN a la ruta del ejecutable."
    )
if not os.path.isfile(CSL):
    raise SystemExit(
        f"FALLO: no se encontro ieee.csl en '{CSL}'. Descargarlo desde "
        "https://github.com/citation-style-language/styles/blob/master/ieee.csl "
        "y/o fijar IEEE_CSL a su ruta."
    )

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

# Nombre del primer capitulo: donde empieza la numeracion arabiga (1.3 del
# formato). Debe coincidir con el \chapter{...} real de 01_introduccion.tex.
PRIMER_CAPITULO = "Introducción"


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


def _limpia_latex_simple(s):
    """Para texto de portada: quita saltos de linea LaTeX y espacios extra."""
    s = re.sub(r"\\\\\s*(\[[^\]]*\])?", "\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def leer_portada():
    """Extrae titulo/autores/fecha de Portadas/portada_principal.tex.

    No se le pasa a pandoc: usa \\begin{titlepage}, \\makeatletter y macros
    (\\@title, \\@author) que son validos en LaTeX puro pero que el lector de
    pandoc no resuelve de forma confiable. Se reconstruye la portada a mano
    con python-docx en vez de arriesgar que pandoc la deje a medias.
    """
    ruta = os.path.join(TESIS, "Portadas", "portada_principal.tex")
    t = open(ruta, encoding="utf-8").read()
    titulo_m = re.search(r"\\title\{(.*?)\}\s*\n", t, re.S)
    autor_m = re.search(r"\\author\{(.*?)\}\s*\n", t, re.S)
    fecha_m = re.search(r"\\date\{(.*?)\}", t, re.S)
    titulo = _limpia_latex_simple(titulo_m.group(1)) if titulo_m else "Título del informe"
    autores = _limpia_latex_simple(autor_m.group(1)).split("\n") if autor_m else []
    fecha = _limpia_latex_simple(fecha_m.group(1)) if fecha_m else ""
    return {"titulo": titulo, "autores": [a for a in autores if a], "fecha": fecha}


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

    def fija(nombre, tam, negrita, mayus=False, antes=0, despues=0, sangria=None,
             salto_pagina=False):
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
        if salto_pagina:
            parrafo.page_break_before = True

    # Cuerpo: 12 pt, sangria 1 cm, sin espacio entre parrafos, justificado,
    # con control de lineas viudas/huerfanas (1.2, ultimo punto del formato).
    fija("Normal", 12, False, sangria=1.0)
    doc.styles["Normal"].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    doc.styles["Normal"].paragraph_format.widow_control = True
    fija("Body Text", 12, False, sangria=1.0)
    # Encabezados: capitulo 14 en mayuscula (cada uno en pagina nueva, 1.2),
    # seccion 14, subseccion 12 -- ninguna de las dos en mayuscula (1.2).
    fija("Heading 1", 14, True, mayus=True, antes=0, despues=8, sangria=0,
         salto_pagina=True)
    fija("Heading 2", 14, True, antes=10, despues=0, sangria=0)
    fija("Heading 3", 12, True, antes=8, despues=0, sangria=0)
    fija("Title", 18, True, sangria=0)
    fija("Compact", 12, False, sangria=0)
    fija("Author", 12, False, sangria=0)
    fija("Date", 12, False, sangria=0)
    fija("Bibliography", 12, False, sangria=0)

    doc.save(ruta)


def convertir(tex, ref):
    # El destino final es tesis/, no el scratch: es el archivo que se versiona
    # en el repo (ver .gitignore / CLAUDE.md), asi que build_docx.py debe
    # dejarlo ahi directamente y no depender de un paso manual de copia.
    salida = os.path.join(TESIS, "Informe_avance.docx")
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


# ---------------------------------------------------------------------------
# Post-procesamiento: portada + numeracion de pagina (secciones 1.3 y 2 del
# formato). Pandoc no genera ninguna de las dos.
#
# Mecanismo OOXML: cada seccion, salvo la ultima del documento, cierra con un
# <w:p> cuyo <w:pPr> contiene el <w:sectPr> de ESA seccion (no de la
# siguiente). La ultima seccion del documento no tiene ese parrafo: sus
# propiedades son el <w:sectPr> que cuelga directo de <w:body>. Por eso
# "insertar un salto de seccion antes del parrafo X" es, en la practica,
# insertar un <w:p> con una copia del <w:sectPr> vigente justo antes de X: lo
# que quede antes hereda esas propiedades (numeracion incluida) y lo que
# sigue las hereda del siguiente marcador (o del <w:sectPr> final del body).
# python-docx no expone esto como metodo -- doc.add_section() solo puede
# partir al final del documento -- asi que se arma a mano con lxml.
# ---------------------------------------------------------------------------

def _campo_pagina(paragraph):
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    run = paragraph.add_run()
    for tag, attrs, texto in [
        ("w:fldChar", {"w:fldCharType": "begin"}, None),
        ("w:instrText", {"xml:space": "preserve"}, "PAGE"),
        ("w:fldChar", {"w:fldCharType": "separate"}, None),
        ("w:fldChar", {"w:fldCharType": "end"}, None),
    ]:
        el = OxmlElement(tag)
        for k, v in attrs.items():
            el.set(qn(k), v)
        if texto is not None:
            el.text = texto
        run._r.append(el)


def _fijar_numeracion(section, fmt, inicio=None):
    """fmt: 'lowerRoman' o 'decimal'. w:pgNumType no es una propiedad de
    alto nivel en python-docx."""
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    sectPr = section._sectPr
    pgNumType = sectPr.find(qn("w:pgNumType"))
    if pgNumType is None:
        pgNumType = OxmlElement("w:pgNumType")
        sectPr.append(pgNumType)
    pgNumType.set(qn("w:fmt"), fmt)
    if inicio is not None:
        pgNumType.set(qn("w:start"), str(inicio))


def _pie_con_numero(section):
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    footer = section.footer
    footer.is_linked_to_previous = False
    for p in list(footer.paragraphs):
        p.clear()
    p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    _campo_pagina(p)


def _pie_vacio(section):
    footer = section.footer
    footer.is_linked_to_previous = False
    for p in list(footer.paragraphs):
        p.clear()


def _insertar_salto_seccion_antes(doc, parrafo_referencia_element, tipo="nextPage"):
    """Cierra la seccion actual justo antes de `parrafo_referencia_element`:
    inserta un <w:p> con una copia del <w:sectPr> vigente (el del final del
    body). Devuelve ese <w:sectPr> copiado para que el llamador fije su
    numeracion (rige la seccion que TERMINA ahi, no la que sigue)."""
    import copy
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    body = doc.element.body
    body_sectPr = body.find(qn("w:sectPr"))
    sectPr_copia = copy.deepcopy(body_sectPr)
    tipo_el = sectPr_copia.find(qn("w:type"))
    if tipo_el is None:
        tipo_el = OxmlElement("w:type")
        sectPr_copia.append(tipo_el)
    tipo_el.set(qn("w:val"), tipo)

    nuevo_p = OxmlElement("w:p")
    pPr = OxmlElement("w:pPr")
    pPr.append(sectPr_copia)
    nuevo_p.append(pPr)
    parrafo_referencia_element.addprevious(nuevo_p)
    return sectPr_copia


def _agregar_parrafos_portada(doc, portada):
    """Devuelve la lista de elementos <w:p> de la portada, ya en el documento
    (al final, que es donde add_paragraph los deja) pero todavia sin mover."""
    from docx.shared import Pt, Cm
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn

    # Se cuenta por posicion, no por identidad de objeto: lxml puede devolver
    # envoltorios Python nuevos (con id() distinto) para el mismo nodo XML en
    # cada llamada a iterchildren(), asi que comparar id()s de "antes" contra
    # "despues" es un error real (encontrado el 2026-09-08: capturaba ~50
    # parrafos de mas, todo el resto del documento se corrompia).
    body = doc.element.body
    cuenta_antes = len(list(body.iterchildren(qn("w:p"))))

    def parrafo(texto, tam, negrita=False,
                alineacion=WD_ALIGN_PARAGRAPH.CENTER, despues=0):
        p = doc.add_paragraph()
        p.alignment = alineacion
        p.paragraph_format.space_after = Pt(despues)
        run = p.add_run(texto)
        run.font.name = "Times New Roman"
        run.font.size = Pt(tam)
        run.font.bold = negrita
        return p

    imagen = os.path.join(TESIS, "Portadas", "imagenes", "encabezado.png")
    if os.path.isfile(imagen):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.add_run().add_picture(imagen, width=Cm(16))
    else:
        parrafo("ESCUELA DE INGENIERÍA INFORMÁTICA", 12, negrita=True)
        parrafo("PONTIFICIA UNIVERSIDAD CATÓLICA DE VALPARAÍSO", 12,
                 negrita=True, despues=24)

    for _ in range(4):
        parrafo("", 12)
    parrafo(portada["titulo"], 18, negrita=True, despues=24)
    for autor in portada["autores"]:
        parrafo(autor, 14)
    parrafo("", 12)
    parrafo("Seminario de Título", 12, negrita=True)
    parrafo("Informe de avance", 12, negrita=True)
    if portada["fecha"]:
        parrafo(portada["fecha"], 12, negrita=True)

    return list(body.iterchildren(qn("w:p")))[cuenta_antes:]


def postprocesar(ruta_docx, portada):
    """Antepone la portada y divide el documento en tres secciones:
    portada (sin numero), Resumen..Objetivos (romano desde i) e
    Introduccion..Conclusiones (arabigo desde 1) -- formato 1.3 y 2."""
    from docx import Document
    from docx.oxml.ns import qn

    doc = Document(ruta_docx)
    body = doc.element.body

    primer_parrafo_original = body.find(qn("w:p"))  # "Resumen" (Heading 1)

    # 1) Construir la portada (queda al final del body por ahora) y
    #    trasplantarla al principio, en orden, antes del contenido original.
    parrafos_portada = _agregar_parrafos_portada(doc, portada)
    for el in parrafos_portada:
        body.remove(el)
        primer_parrafo_original.addprevious(el)

    # 2) Cerrar la seccion "portada" justo antes de "Resumen".
    sectPr_portada = _insertar_salto_seccion_antes(doc, primer_parrafo_original)

    # 3) Ubicar el primer capitulo (arranca la numeracion arabiga) y cerrar
    #    ahi la seccion "Resumen..Objetivos".
    parrafo_primer_capitulo = None
    for p in doc.paragraphs:
        estilo = p.style.name if p.style else ""
        if estilo == "Heading 1" and p.text.strip().lower().startswith(PRIMER_CAPITULO.lower()):
            parrafo_primer_capitulo = p._p
            break
    if parrafo_primer_capitulo is None:
        raise SystemExit(
            f"FALLO: no se encontro el capitulo '{PRIMER_CAPITULO}' (Heading 1) "
            "para partir la numeracion de pagina; revisar PRIMER_CAPITULO."
        )
    sectPr_frontmatter = _insertar_salto_seccion_antes(doc, parrafo_primer_capitulo)

    doc.save(ruta_docx)

    # 4) Releer: los <w:sectPr> insertados ya son doc.sections reales.
    doc = Document(ruta_docx)
    secciones = doc.sections
    if len(secciones) != 3:
        raise SystemExit(
            f"FALLO: se esperaban 3 secciones (portada, romano, arabigo), "
            f"salieron {len(secciones)}. Revisar _insertar_salto_seccion_antes."
        )
    _pie_vacio(secciones[0])                              # portada: sin numero
    _pie_con_numero(secciones[1])
    _fijar_numeracion(secciones[1], "lowerRoman", inicio=1)  # Resumen..Objetivos
    _pie_con_numero(secciones[2])
    _fijar_numeracion(secciones[2], "decimal", inicio=1)     # Introduccion..fin
    doc.save(ruta_docx)

if __name__ == "__main__":
    portada = leer_portada()
    tex = construir_tex()
    ref = preparar_referencia()
    docx = convertir(tex, ref)
    postprocesar(docx, portada)
    print("OK ->", docx, os.path.getsize(docx), "bytes")
