# Reglas de trabajo — tesis multiagente

Este repositorio se edita por dos agentes Claude que corren en cuentas distintas y en
sesiones separadas. **No pueden verse ni mensajearse entre sí.** El único canal de
comunicación es este repositorio: las ramas, `TASKS.md` y los pull requests. Si algo no
queda escrito en el repositorio, el otro agente no se enterará.

## Proyecto

Tesis: framework metodológico que transforma la representación latente de un modelo
profundo ya entrenado en una hipótesis relacional compacta (un grafo dirigido y ponderado
sobre variables nombradas), más el protocolo de validación que decide cuándo una relación
extraída merece tratarse como hipótesis científica.

**El alcance no se agota en explicar el modelo.** El objetivo final es producir una
hipótesis relacional sobre el fenómeno subyacente a los datos; explicar el modelo (qué
relaciones aprendió) es un paso instrumental hacia ese fin, no el fin en sí mismo.
Introducción, objetivos y propuesta deben mantener esta jerarquía sin invertirla —
ninguna debe presentar la explicabilidad del modelo como el objetivo terminal del
trabajo. Instrucción explícita del autor humano, 2026-09-08 (ver Task-022 en `TASKS.md`).

Fuentes primarias, de solo lectura:

- `Presentación preliminar del tema de investigación.pdf` — sus secciones 3 a 8
  (formalización matemática: la matriz relacional $A$, $\Phi: A \to G$, $\Psi: G \to H$,
  análisis estructural del grafo, $F(H,M) \to R$, análisis espectral del operador
  relacional, y la composición $X \to M_\theta \to A \to G \to H \to R$) son la única
  formalización matemática vigente del framework y la fuente de autoridad para cualquier
  ecuación que se cite — aunque no todas se incluyan en cada entrega (ver historial de
  `TASKS.md` sobre qué se excluyó explícitamente en la entrega vigente). El resto del
  documento (secciones 1-2 y 9-17: contexto, la arquitectura ConvTransformer del caso de
  estudio, diferenciación con el estado del arte, aplicabilidad a otros dominios,
  objetivos, hipótesis, diseño experimental, resultados y hoja de ruta) es información de
  motivación **aún no formalizada**: no tratarla como axioma ni derivar de ahí reglas
  nuevas que el documento no plantea como tales (p. ej., no inventar condiciones fijas de
  selección de dominio a partir de un ejemplo ilustrativo — ver Task-022).
- `Protocolo PRISMA/` — revisión sistemática, exports de Zotero (WOS, Scopus, PUBMED).
- `Framework.py` — implementación del framework (sucesor de `Prototipo_Preliminar.ipynb`,
  reemplazado por el autor humano el 2026-09-07). Es código fuente puro, sin salidas de
  celda: las cifras experimentales ya citadas en la prosa (Jaccard, % de arista
  dominante, sensibilidad a K) se verificaron contra las celdas ejecutadas del notebook
  anterior, que ya no está en el repositorio. Antes de citar una cifra nueva desde
  `Framework.py`, confirmar que efectivamente se ejecutó (no solo que el código existe)
  y con qué parámetros; varias fórmulas de índices cambiaron respecto al notebook
  (MARI, ARI, CHL\_REDEDGE, PSRI), así que una cifra del notebook viejo no se
  reafirma automáticamente ejecutando este archivo. **Es una instancia concreta del
  framework (el caso de estudio de índices espectrales con el ConvTransformer), no su
  definición general**: no usar sus detalles de implementación (encoder convolucional,
  stride, kernel, etc.) para describir el framework en abstracto, y en particular no
  para la sección Propuesta (ver Task-022 en `TASKS.md`).

Ningún agente escribe en esas rutas.

### Corpus de referencias: cuál usar y cuál no

Dentro de `Protocolo PRISMA/` hay dos generaciones del cribado. Solo la segunda es
válida:

- **Vigente, usar esta:** `Protocolo PRISMA/Referencias_seminario.xlsx`, hoja
  `Referencias` — 25 referencias, cada una con DOI y estado `Leído`. La hoja
  `Aporte por sección` marca con `X` a qué sección de la tesis aporta cada referencia
  (Resumen, Introducción, Objetivos, Estado del arte, Marco Teórico, Plan de trabajo,
  Propuesta, Conclusiones); úsala para decidir dónde citar cada una. Los PDF completos
  de estas 25 referencias, numerados 1-25 en el mismo orden que la hoja, están en
  `Protocolo PRISMA/PDF seleccionados para el seminario/`.
- **Obsoleto, ignorar por completo:** `Protocolo PRISMA/Analisis 137 referencias.xlsx`
  y `Protocolo PRISMA/PDF primera iteración/`. Corresponden a una iteración anterior
  del cribado, superada por `Referencias_seminario.xlsx`. Ningún agente los lee ni cita
  desde ellos, aunque aparezcan en `Bítacora_revision_sistematica.docx` o en el corpus
  núcleo original de 22 referencias que trae `secciones/99_bibliografia.tex` desde antes
  de que este flujo multiagente existiera.

Toda entrada nueva en `referencias.bib` debe corresponder a una fila de
`Referencias_seminario.xlsx` (mismo DOI). Una entrada que no resuelve contra esa hoja es
del corpus obsoleto y debe reemplazarse o justificarse explícitamente, no darse por
buena solo porque ya estaba en el documento heredado. Desde el 2026-09-08 el corpus está
además **cerrado en exactamente 25**: el informe de avance cita las 25 filas de la hoja
y nada más (instrucción explícita del autor humano, ver historial de `TASKS.md`). Si el
documento vuelve a crecer más allá del informe de avance, esta restricción de "cerrado
en 25" se levanta primero con el autor humano; el resto de la regla (toda cita nueva
sale de esa hoja) sigue aplicando siempre.

Nota de calidad de datos: la fila 5 de la hoja trae un título ("...Faithfulness of
Causality in Saliency-Based Explanations...") que no coincide con el título real del
DOI que la acompaña (sin "Causality"). Usar el título que resuelve el DOI, no el de la
celda, cuando difieran — ya se verificó para esa fila (`rizzo2022faithfulness` en
`referencias.bib` usa el título correcto).

Procedencia de las citas, a declarar explícitamente donde se describa la metodología de
revisión (síntesis PRISMA en `03_estado_arte.tex` o similar): la mayoría de las 25
referencias vigentes salió del protocolo PRISMA, pero no todas — algunas se incorporaron
por búsqueda propia del autor humano y con asistencia de IA, al margen del cribado
sistemático. No presentar las 25 como resultado exclusivo del protocolo. Instrucción
explícita del autor humano, 2026-09-08 (ver Task-024 en `TASKS.md`).

Fuera de alcance, declarado como trabajo futuro: esta tesis no analiza las citas de las
referencias utilizadas (a quién citan esos 25 trabajos, redes de citación). Donde se
describa el alcance de la revisión bibliográfica, declararlo como línea de trabajo
futuro, no como limitación pendiente de esta entrega. Instrucción explícita del autor
humano, 2026-09-08 (ver Task-024 en `TASKS.md`).

## Estructura editable

Desde el 2026-09-08 el documento usa el template institucional PUCV (Task-019,
instrucción directa del autor humano — ver `TASKS.md` para el detalle completo de la
migración). La estructura anterior (`secciones/00_resumen.tex`...`99_bibliografia.tex`,
`thebibliography`) ya no existe; no asumir que sigue vigente si esta sección parece
desactualizada, verificar contra `tesis/main.tex`.

```
tesis/
  main.tex                        solo el autor humano lo edita (ver excepción abajo)
  pucv_inf_2024.sty               copia literal del template PUCV, no se edita
  TASKS.md                        panel de control y mecanismo de reserva
  Portadas/portada_principal.tex  portada; datos de autor/asignatura, no prosa
  Resumen/resumen.tex             resumen + abstract + palabras clave
  referencias.bib                 bibliografía biblatex, ver regla de corpus arriba
  secciones/
    01_introduccion.tex
    02_objetivos.tex
    03_estado_arte.tex
    04_marco_teorico.tex           absorbe el contenido de la antigua formalización
    05_plan_trabajo.tex
    06_propuesta.tex               su §6.2 absorbe el contenido de la antigua arquitectura
    07_conclusiones.tex
```

El documento usa **biblatex con backend biber** (estilo APA), no BibTeX ni
`thebibliography`. Las claves de `\cite` se resuelven contra las entradas `@...{clave,...}`
de `referencias.bib`. Compilar exige la secuencia `pdflatex → biber → pdflatex → pdflatex`;
`./scripts/compilar.sh` ya la implementa, no compilar a mano con solo `pdflatex`.

`referencias.bib` funciona con la misma disciplina que tenía el `thebibliography`
anterior: es de claude-2, y aunque técnicamente no es "append-only" en el sentido de
BibTeX (las entradas no tienen un orden que preservar), se trata igual como tal para
evitar colisiones — agregar entradas nuevas al final, no reordenar ni reformatear las
existentes al agregar una.

## Toolchain

LaTeX local instalado: TinyTeX en `$HOME/AppData/Roaming/TinyTeX/bin/windows` (por
máquina; `./scripts/compilar.sh` ya lo detecta así, no usa un usuario hardcodeado).

Compilación completa desde `tesis/`:

```bash
export PATH="$HOME/AppData/Roaming/TinyTeX/bin/windows:$PATH"
pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
```

Dos pasadas, no cuatro: la segunda resuelve el índice y las referencias cruzadas. No se
ejecuta `bibtex`. El número de páginas del documento vigente cambia con el contenido;
no asumir un número fijo, verificar con `./scripts/compilar.sh`.

El template PUCV necesita paquetes que una instalación mínima de TinyTeX no trae por
defecto: `titlesec`, `lipsum`, `fancyhdr`, `algorithms`, `algorithmicx`, `glossaries`,
`nomencl`, `biblatex`, `biblatex-apa`, `biblatex-ieee`, `biber`, `csquotes`, `caption`,
`enumitem`, `etoolbox`, `koma-script`, `psnfss`, `hyphen-spanish`. `biblatex-ieee` es el
que trae el estilo `ieee` que usa `pucv_inf_2024.sty` desde Task-021 (citas numéricas);
sin él, biblatex falla con "Style 'ieee' not found" y en cascada "Command '\cite'
undefined" en todo el documento — verificado y corregido en esta máquina, 2026-09-08.
Si falta alguno, instalarlo con
`tlmgr install <paquete>` (el binario es `tlmgr.bat` dentro de la carpeta de TinyTeX;
en bash se puede invocar por ruta completa, no hace falta PowerShell).

**Ningún push sin compilar antes.** Un commit que rompe la compilación bloquea al otro agente.

### Generar el DOCX (`scripts/build_docx.py`)

Aplana las secciones y llama a `pandoc` con `--citeproc` y el estilo `scripts/ieee.csl`
(ya está en el repo, no hace falta descargarlo) para producir `Informe_avance.docx` con
el formato de la Escuela. Necesita `pandoc` y el paquete Python `python-docx`
(`pip install python-docx`); revisa PATH y, si no lo encuentra ahí, la ruta típica del
instalador de winget (`%LOCALAPPDATA%\Pandoc\pandoc.exe`) — instalar con
`winget install JohnMacFarlane.Pandoc` si falta. Corre sin variables de entorno en la
máquina de claude-2 (verificado 2026-09-08); si hace falta forzar una ruta,
`PANDOC_BIN`, `IEEE_CSL` y `BUILD_DOCX_SCRATCH` la sobrescriben.

**El formato exacto que debe seguir el DOCX está en
`Formato_Informes_Proyecto_Título-2024.pdf`** (raíz del repo, subido por el autor
humano el 2026-09-08). Es la fuente de autoridad para cómo se ve el documento —
`pucv_inf_2024.sty` cubre el PDF, este script cubre el DOCX, y ninguno de los dos se
asume correcto solo porque compila: hay que verificar contra ese PDF si algo del
formato cambia. Lo que `build_docx.py` ya implementa siguiendo ese documento (sección
del PDF entre paréntesis):

- Márgenes 2,5 cm, papel carta, Times New Roman 12, sangría de 1 cm, interlineado
  sencillo, control de líneas viudas/huérfanas (1.1–1.2).
- Encabezados de capítulo en mayúscula negrita 14, cada uno en página nueva; secciones
  y subsecciones en negrita 14/12 sin mayúscula (1.2).
- Portada reconstruida a mano con `python-docx` (imagen institucional, título 18pt
  negrita, autores, "Seminario de Título" + "Informe de avance" + fecha), **no** vía
  pandoc: `Portadas/portada_principal.tex` usa `\begin{titlepage}`, `\makeatletter` y
  macros (`\@title`, `\@author`) que el lector LaTeX de pandoc no resuelve de forma
  confiable (2).
- Numeración de página abajo a la derecha: la portada no lleva número; de Resumen a
  Objetivos va en romano minúsculo empezando en "i"; de Introducción en adelante, en
  arábigo empezando en 1 (1.3). Esto exige tres secciones DOCX reales (no solo
  saltos de página) con `w:pgNumType` distinto cada una — python-docx no lo expone
  como propiedad de alto nivel, `build_docx.py` lo arma con XML crudo
  (`_insertar_salto_seccion_antes`, `_fijar_numeracion`).

  **Bug real encontrado y corregido el 2026-09-08, para no repetirlo:** identificar
  "qué párrafos son nuevos" comparando `id()` de objetos `lxml` antes/después de
  insertarlos corrompe el documento — lxml puede devolver un envoltorio Python con
  `id()` distinto para el mismo nodo XML en cada llamada a `iterchildren()`, así que la
  comparación por identidad captura decenas de párrafos de más (todo el resto del
  documento, en la práctica) y termina moviendo contenido a un lugar equivocado sin
  lanzar ningún error. La forma segura es contar por posición (cuántos `<w:p>` había
  antes, tomar los que sobran después), no por identidad de objeto.

No implementado todavía, pendiente si hace falta más precisión: la regla de 1.2 sobre
trasladar una sección completa a la página siguiente cuando su primer párrafo no
alcanza dos líneas (se aproxima con `widow_control`, que no es exactamente lo mismo).
La extensión máxima de 30 páginas (Introducción a Conclusiones, sección 4 del formato)
es responsabilidad de quien escribe la prosa, no de este script.

## Roles

### claude-1 — Extractor y redactor

Lee las fuentes primarias y produce prosa en `secciones/*.tex`. Dueño de la narrativa:
resumen (`Resumen/resumen.tex`), introducción, objetivos, estado del arte, síntesis del
protocolo PRISMA, plan de trabajo, conclusiones, y de §6.1/§6.3+ de `06_propuesta.tex`
(§6.2 es de claude-2, ver abajo).

Toda referencia que cite debe salir de `Protocolo PRISMA/Referencias_seminario.xlsx`
(ver "Corpus de referencias" más arriba), leyendo el PDF correspondiente en
`PDF seleccionados para el seminario/`. Si necesita citar una referencia de esa hoja que
todavía no tiene entrada en `referencias.bib`, no la agrega él mismo: ese archivo es de
claude-2. La pide en `TASKS.md` (título, DOI, y en qué frase la va a usar) para que
claude-2 la agregue. **Toda referencia debe usarse al menos tres veces** en el
documento (`\cite`/`\textcite`/`\parencite`, en apariciones argumentativas distintas):
una cita que solo sirve para una ocasión no se pide ni se agrega tal cual — instrucción
explícita del autor humano, 2026-09-08 (ver Task-023 en `TASKS.md`).

Al resumir un trabajo referenciado (estado del arte, o cualquier comparación en marco
teórico o propuesta), **no se reporta la precisión ni el desempeño predictivo del
algoritmo del artículo citado**. Lo único relevante es el resultado obtenido con el
método de explicabilidad aplicado en ese trabajo: qué reveló, qué error tuvo, sus pros y
contras. Instrucción explícita del autor humano, 2026-09-08 (ver Task-023 en
`TASKS.md`).

No toca `main.tex` ni `pucv_inf_2024.sty` salvo instrucción explícita del autor humano
(como ocurrió con la migración de Task-019). No corrige formato en archivos ajenos.

### claude-2 — Auditor de consistencia y LaTeX

Revisa lo que produjo claude-1. Verifica que compile, corrige `\label`, `\ref` y `\cite`
colgantes, entornos de figura y tabla, y coherencia conceptual: que las cifras y métricas
citadas en la prosa coincidan con las de `Framework.py` cuando haya evidencia de que se
ejecutó, o con el registro histórico de resultados ya verificados en `TASKS.md` cuando
la fuente original que los produjo (el notebook) ya no esté en el repositorio. Dueño de
`04_marco_teorico.tex` (formalización), `referencias.bib` (bibliografía) y §6.2 de
`06_propuesta.tex` (arquitectura instrumental). La antigua matriz de revisión
sistemática en apéndice (`A_matriz_literatura.tex`) salió del documento en Task-019; si
se repone, sigue siendo de claude-2.

Agrega las entradas que claude-1 pida en `TASKS.md` a `referencias.bib`, y audita que
todo `\cite` de la prosa resuelva contra una referencia de `Referencias_seminario.xlsx`,
no contra el corpus obsoleto. Un `\cite` que no resuelve se marca `NEED_REWRITE` con el
DOI esperado.

**No reescribe prosa ajena.** Si el contenido está mal, lo marca `NEED_REWRITE` en
`TASKS.md` con el motivo y lo devuelve a claude-1.

## Ciclo de trabajo obligatorio

1. `git pull` y leer `TASKS.md` completo.
2. Elegir la primera tarea asignada al propio rol que no esté bloqueada.
3. Si el archivo está `IN_PROGRESS` por el otro agente, **no tocarlo**. Pasar a la
   siguiente tarea, o detenerse y avisar al autor humano.
4. Reservar: marcar el archivo `IN_PROGRESS` en `TASKS.md`, commit y push. Esto ocurre
   **antes** de editar, y es lo que evita la colisión.
5. Editar únicamente el archivo reservado.
6. Compilar con `./scripts/compilar.sh`. Si no compila, arreglar antes de seguir.
7. Marcar `NEED_REVIEW` en `TASKS.md` y hacer commit del trabajo y del cambio de estado
   **en el mismo commit**. Estados y contenido separados hacen que el otro agente lea
   `APPROVED` de contenido que todavía no llegó.
8. `./scripts/sincronizar.sh` para rebasar sobre `origin/main`, y abrir el PR con
   `./scripts/gh.sh pr-abrir "<titulo>" <cuerpo.md>`.
9. Avisar en `TASKS.md`, en la sección `PR abiertos`, que hay un PR esperando revisión del
   otro agente. Ese es el único aviso que el otro va a recibir.

## Reglas de aislamiento

- Nunca dos agentes en el mismo archivo. La reserva en `TASKS.md` es la autoridad.
- `main` es la rama sincronizada con Overleaf. Ningún agente escribe directo en `main`.
- Ramas: `claude-1/drafting` para contenido, `claude-2/review` para auditoría y formato.
- `main.tex` lo edita solo el autor humano. Si una sección nueva necesita un `\include`,
  pedirlo en `TASKS.md` en vez de agregarlo. La excepción del 2026-09-08 (Task-019,
  migración al template PUCV) fue por instrucción directa del autor humano y no
  establece precedente: sigue haciendo falta esa misma instrucción explícita para
  volver a tocarlo.
- `referencias.bib` se trata como append-only en la práctica (ver "Estructura editable"):
  agregar entradas al final, no reordenar ni reformatear las existentes.
- Los archivos auxiliares de LaTeX (`.aux`, `.log`, `.toc`, `.out`, `.bbl`, `.blg`,
  `.bcf`, `.run.xml`, `.nlo`, `.nls`, `.glo`, `.gls`, `.ist`) están en `.gitignore`. No
  forzar su commit: cambian en cada compilación y colisionan siempre.
- `tesis/main.pdf` no se versiona (vuelve a la regla general el 2026-09-08: entre el
  8 y el 8 de septiembre se versionó, pero el autor humano pidió revertirlo). El
  entregable que sí se versiona es **`tesis/Informe_avance.docx`**
  (`./scripts/build_docx.py`, ver más arriba), para que se pueda ver el documento
  actualizado en GitHub sin compilar localmente. Regenerar y comitear el DOCX junto
  con el `.tex` que lo generó, en el mismo commit — nunca un commit de solo texto
  seguido de un commit de solo DOCX, porque es un binario y cualquier commit
  intermedio de otro agente sobre el mismo archivo genera conflicto sin nada que
  conservar de ambos lados (a diferencia de `TASKS.md`, un binario no se puede
  "conservar ambos lados"). Por eso: regenerar y comitear el DOCX solo al cerrar una
  tarea (paso 7 del ciclo obligatorio), nunca en los commits de reserva o intermedios.

## Protocolo de contradicciones

El propósito de los pull requests no es solo revisar formato: es detectar contradicciones
entre lo que escribió un agente y lo que escribió el otro. Antes de aprobar un PR, el
agente revisor verifica explícitamente:

1. **Cifras.** Toda cantidad en la prosa (número de semillas, de referencias, de
   paradigmas, de eslabones, valores de métricas) coincide con la fuente primaria y con
   lo afirmado en las demás secciones.
2. **Alcance.** Lo que la introducción promete es lo que la formalización entrega. Si la
   introducción afirma que algo fue validado y la sección de resultados dice que está
   diseñado pero no ejecutado, es una contradicción y bloquea el merge.
3. **Nomenclatura.** Los símbolos y nombres se usan igual en todas las secciones. Si una
   sección llama a la transformación `Phi` y otra la llama `F`, se unifica antes de
   mergear.
4. **Referencias.** Todo `\cite` resuelve a una entrada existente en `referencias.bib`,
   esa entrada corresponde a una fila de `Protocolo PRISMA/Referencias_seminario.xlsx`
   (mismo DOI, no al corpus obsoleto de `Analisis 137 referencias.xlsx`), y la
   afirmación que sostiene corresponde a lo que ese trabajo dice de verdad. Además, toda
   clave se usa (`\cite`/`\textcite`/`\parencite`) al menos **tres veces** en el
   documento: una referencia citada una sola vez no se aprueba tal cual, se marca
   `NEED_REWRITE` (ampliarla a al menos 3 apariciones argumentativas distintas, o
   retirarla).

Las contradicciones se reportan en el PR con
`./scripts/gh.sh pr-comentar <n> <comentario.md>` y se anotan en la sección
`Contradicciones` de `TASKS.md`. Una contradicción abierta bloquea el merge de las dos
secciones implicadas, no solo de una.

## Merge automático con revisión cruzada

El autor humano no está en el circuito de merge. Los agentes mergean, con una única regla
que lo hace seguro:

**Ningún agente mergea su propio PR. Cada uno mergea el del otro.**

Esa regla es lo que sostiene todo el esquema. Si un agente mergeara lo propio, nadie
revisaría nada y los pull requests serían un trámite vacío. Al mergear el del otro, la
revisión cruzada se conserva y el humano sale del circuito sin que se pierda el control.

Antes de mergear el PR ajeno, el revisor:

1. `./scripts/sincronizar.sh` y `./scripts/gh.sh pr-ver <n>` para ver qué archivos toca.
2. Verifica que solo toque archivos cuyo dueño sea el autor del PR. Un PR que modifica
   una sección ajena es un fallo del protocolo de reserva: no se mergea, se comenta. La
   única excepción es una migración de formato que el autor humano ordenó explícitamente
   (como Task-019): ahí sí toca archivos ajenos, siempre que el PR lo documente
   (instrucción del autor humano, qué se movió y por qué, contenido técnico migrado sin
   reescribir) y que el revisor confirme ambas cosas antes de mergear.
3. Hace checkout de la rama del PR y corre `./scripts/compilar.sh`. **Esta compuerta
   reemplaza la revisión humana.** Si falla, no se mergea.
4. Recorre los cuatro puntos del protocolo de contradicciones sobre el diff.
5. Si todo pasa: `./scripts/gh.sh pr-mergear <n>`, y marca `APPROVED` en `TASKS.md`.
6. Si algo falla: `pr-comentar` con el motivo concreto, y `NEED_REWRITE` en `TASKS.md`.
   No se mergea y no se arregla por cuenta propia: el arreglo es del autor.

## Conflictos

Los resuelve el **autor** del PR, no el revisor, y sin intervención humana:

```bash
./scripts/sincronizar.sh          # rebasa sobre origin/main
# si reporta conflicto: resolver, luego
git add <archivos> && git rebase --continue
git push --force-with-lease origin <rama-propia>
```

`--force-with-lease` sobre la **rama propia** durante un rebase de PR está permitido: es
la única forma de rebasar una rama ya empujada, y falla sola si alguien más la tocó.

Un conflicto en `tesis/TASKS.md` se resuelve **conservando ambos lados**: son filas
distintas de la misma tabla, no versiones rivales del mismo hecho. Es el conflicto
esperado y es barato.

Un conflicto en un `.tex` significa que dos agentes editaron la misma sección, lo que el
protocolo de reserva debía impedir. Ahí sí hay que detenerse y avisar al autor humano,
porque el problema no es el conflicto sino que la reserva falló.

## Nunca sin autorización del autor humano

- Mergear el PR propio.
- `git push --force` sobre `main` o sobre la rama del otro agente.
- Reescribir la historia de `main`, o hacer `commit --amend` de commits ya mergeados.
- Borrar ramas.
- Editar las fuentes primarias en solo lectura.
- Cambiar el preámbulo de `main.tex` o la clase del documento.

## Estilo de la prosa

Español académico, tercera persona, sin primera persona del plural retórica. Los términos
técnicos en inglés que no tienen traducción establecida se dejan en inglés y en redonda
(`attention rollout`, `baseline`, `token`). Las afirmaciones sobre el estado de avance son
literales: lo que está diseñado pero no ejecutado se declara como tal. Desde Task-019, por
condición explícita del autor humano para esta entrega: sin rayas largas (`--`) en la
prosa (los rangos numéricos del `.bib`, p. ej. páginas, no cuentan), y esta entrega en
particular no reporta cifras de ejecución del prototipo (Jaccard, persistencia entre
semillas, sensibilidad a K) — el estado declarado es "diseñado, no ejecutado" sin
matices, aunque el registro de esas cifras y de dónde salieron sigue en el historial de
`TASKS.md` para cuando la entrega que sí las incluya se retome.

Antes de aprobar o mergear un PR con prosa nueva (o al escribir prosa propia en las
secciones que sí posee), pasar una auditoría de tells de escritura de IA — es parte de
la revisión igual que el resto del protocolo de contradicciones, no un paso opcional.
`thesis-prose-audit` (una skill instalada solo en el entorno de claude-2, con el mapeo
de esos patrones al español académico) es una forma de hacerlo; `humanizer` a secas
cubre el mismo conjunto de patrones y sirve igual si la primera no está disponible en
la máquina de quien revisa. Ninguna de las dos vive en el repositorio, así que no
asumir que el otro agente la tiene instalada solo porque esta regla la nombra.

**Sin código, umbrales ni métricas de ejecución en esta entrega.** La prosa se ciñe a
la formulación matemática del framework ($\Phi$, $\Psi$, la composición completa) y a
la defensa conceptual de la idea: por qué la estructura emerge de la representación
aprendida y no de un supuesto previo, por qué hace falta un criterio de validación de
varias etapas y qué distingue una relación genuina de un atajo predictivo, un artefacto
de arquitectura o ruido. No corresponde a esta entrega: detalles de implementación o
arquitectura (ConvTransformer, encoder/decoder, capas, dimensiones — eso ya lo excluía
la nota sobre `Framework.py` de más arriba, para toda la prosa y no solo para la
Propuesta), parámetros operativos concretos (el valor de $K$ del criterio Top-$K$,
métodos específicos de perturbación) y nombres de métricas que se aplicarán en la
ejecución (Jaccard, fidelidad, etc.), en la misma línea que la exclusión de cifras de
ejecución del prototipo ya vigente desde Task-019. La cadena de preguntas del criterio
de validación y la taxonomía de resultados (relación genuina / atajo / artefacto /
ruido) sí se mantienen: son la idea que esta entrega defiende, no un detalle operativo.
Instrucción explícita del autor humano, 2026-09-08 (ver Task-025 en `TASKS.md`).
