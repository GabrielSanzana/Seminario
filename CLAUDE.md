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

Fuentes primarias, de solo lectura:

- `Presentación preliminar del tema de investigación.pdf` — introducción, objetivos, justificación.
- `Protocolo PRISMA/` — revisión sistemática, exports de Zotero (WOS, Scopus, PUBMED).
- `Framework.py` — implementación del framework (sucesor de `Prototipo_Preliminar.ipynb`,
  reemplazado por el autor humano el 2026-09-07). Es código fuente puro, sin salidas de
  celda: las cifras experimentales ya citadas en la prosa (Jaccard, % de arista
  dominante, sensibilidad a K) se verificaron contra las celdas ejecutadas del notebook
  anterior, que ya no está en el repositorio. Antes de citar una cifra nueva desde
  `Framework.py`, confirmar que efectivamente se ejecutó (no solo que el código existe)
  y con qué parámetros; varias fórmulas de índices cambiaron respecto al notebook
  (MARI, ARI, CHL\_REDEDGE, PSRI), así que una cifra del notebook viejo no se
  reafirma automáticamente ejecutando este archivo.

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

Toda entrada nueva en `secciones/99_bibliografia.tex` debe corresponder a una fila de
`Referencias_seminario.xlsx` (mismo DOI). Una entrada que no resuelve contra esa hoja es
del corpus obsoleto y debe reemplazarse o justificarse explícitamente, no darse por
buena solo porque ya estaba en el documento heredado.

## Estructura editable

```
tesis/
  main.tex                        solo el autor humano lo edita
  TASKS.md                        panel de control y mecanismo de reserva
  secciones/*.tex                 una sección por archivo, un dueño por archivo
  secciones/99_bibliografia.tex   append-only, ver regla abajo
```

El documento no usa BibTeX: la bibliografía es un entorno `thebibliography` que vive en
`secciones/99_bibliografia.tex`. Las claves de `\cite` se resuelven contra los
`\bibitem` de ese archivo, no contra un `.bib`.

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

Si falta un paquete, instalarlo con `tlmgr install <paquete>` (usar `tlmgr.bat` desde
PowerShell; no está en el PATH de bash).

**Ningún push sin compilar antes.** Un commit que rompe la compilación bloquea al otro agente.

## Roles

### claude-1 — Extractor y redactor

Lee las fuentes primarias y produce prosa en `secciones/*.tex`. Dueño de la narrativa:
resumen, introducción, estado del arte, síntesis del protocolo PRISMA, aplicabilidad,
conclusiones.

Toda referencia que cite debe salir de `Protocolo PRISMA/Referencias_seminario.xlsx`
(ver "Corpus de referencias" más arriba), leyendo el PDF correspondiente en
`PDF seleccionados para el seminario/`. Si necesita citar una referencia de esa hoja que
todavía no tiene `\bibitem` en `secciones/99_bibliografia.tex`, no la agrega él mismo:
ese archivo es de claude-2 y es append-only. La pide en `TASKS.md` (título, DOI, y en qué
frase la va a usar) para que claude-2 la agregue.

No toca `main.tex`. No corrige formato en archivos ajenos.

### claude-2 — Auditor de consistencia y LaTeX

Revisa lo que produjo claude-1. Verifica que compile, corrige `\label`, `\ref` y `\cite`
colgantes, entornos de figura y tabla, y coherencia conceptual: que las cifras y métricas
citadas en la prosa coincidan con las de `Framework.py` cuando haya evidencia de que se
ejecutó, o con el registro histórico de resultados ya verificados en `TASKS.md` cuando
la fuente original que los produjo (el notebook) ya no esté en el repositorio. Dueño de
las secciones técnicas:
arquitectura, formalización, bibliografía y matriz de literatura.

Agrega los `\bibitem` que claude-1 pida en `TASKS.md` (append-only, ver regla más abajo),
y audita que todo `\cite` de la prosa resuelva contra una referencia de
`Referencias_seminario.xlsx`, no contra el corpus obsoleto de 22 que traía el documento
heredado. Un `\cite` que no resuelve contra esa hoja se marca `NEED_REWRITE` con el DOI
esperado.

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
- `main.tex` lo edita solo el autor humano. Si una sección nueva necesita un `\input`,
  pedirlo en `TASKS.md` en vez de agregarlo.
- `secciones/99_bibliografia.tex` es **append-only**: agregar `\bibitem` nuevos al final,
  nunca reordenar ni reformatear los existentes. Reordenarlo genera conflictos enormes sin
  ningún beneficio, y renumera las citas de todo el documento.
- Los archivos auxiliares de LaTeX (`.aux`, `.log`, `.toc`, `.out`, `.bbl`, `.blg`) están
  en `.gitignore`. No forzar su commit: cambian en cada compilación y colisionan siempre.

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
4. **Referencias.** Todo `\cite` resuelve a un `\bibitem` existente en
   `secciones/99_bibliografia.tex`, ese `\bibitem` corresponde a una fila de
   `Protocolo PRISMA/Referencias_seminario.xlsx` (mismo DOI, no al corpus obsoleto de
   `Analisis 137 referencias.xlsx`), y la afirmación que sostiene corresponde a lo que ese
   trabajo dice de verdad.

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
   una sección ajena es un fallo del protocolo de reserva: no se mergea, se comenta.
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
literales: lo que está diseñado pero no ejecutado se declara como tal.
