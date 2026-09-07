# Panel de control — tesis multiagente

Núcleo de verdad del proyecto. Ningún agente edita un archivo `.tex` sin haber leído
este panel primero, y sin haber reservado el archivo aquí.

Reglas completas en [`../CLAUDE.md`](../CLAUDE.md).

## Convención de estados

| Estado | Significado |
|---|---|
| `TODO` | Sin asignar, disponible para tomar |
| `IN_PROGRESS` | Reservado. El otro agente NO lo toca |
| `NEED_REVIEW` | Contenido listo, espera auditoría de claude-2 |
| `NEED_REWRITE` | Auditoría lo rechazó. Vuelve a claude-1 con motivo |
| `APPROVED` | Auditado y mergeado a `main` |
| `BLOCKED` | Depende de otra tarea o de una decisión humana |

Protocolo de reserva: cambiar el estado a `IN_PROGRESS`, hacer commit y push, **y después**
editar. El cambio de estado y el trabajo terminado van en el mismo commit al cerrar.

## Estado de secciones

| Archivo | Estado | Dueño | Líneas | Nota |
|---|---|---|---|---|
| `secciones/00_resumen.tex` | `APPROVED` | claude-1 | 3 | Split verificado |
| `secciones/01_introduccion.tex` | `APPROVED` | claude-1 | 43 | Split verificado |
| `secciones/02_estado_arte.tex` | `APPROVED` | claude-1 | 76 | Task-010 mergeado a `main` en PR #5 |
| `secciones/03_objetivos.tex` | `APPROVED` | claude-1 | 18 | Split verificado |
| `secciones/04_arquitectura.tex` | `APPROVED` | claude-2 | 42 | Split verificado |
| `secciones/05_formalizacion.tex` | `APPROVED` | claude-2 | 35 | Split verificado |
| `secciones/06_aplicabilidad.tex` | `APPROVED` | claude-1 | 31 | Incluye `tab:dominios` |
| `secciones/07_conclusiones.tex` | `APPROVED` | claude-1 | 10 | Split verificado |
| `secciones/99_bibliografia.tex` | `APPROVED` | claude-2 | 47 | 22 `\bibitem`, append-only |
| `secciones/A_matriz_literatura.tex` | `NEED_REVIEW` | claude-2 | 41 | Task-013: corrige overclaim de validación en fila `chefer2021transformer` |

`APPROVED` aquí significa que el contenido corresponde exactamente al `.tex` canónico y
que compila. No significa que el contenido esté auditado: esa es la cola de abajo.

## Estado del split

Resuelto. El `.tex` canónico se partió en diez archivos y el resultado se verificó
comparando el texto extraído del PDF del monolito contra el del `main.tex` dividido:
**idéntico**, 18 páginas, mismo tamaño de PDF. El split no perdió ni alteró contenido.

Dos hechos del documento que condicionan el trabajo:

1. **No usa BibTeX.** La bibliografía es un entorno `thebibliography` con 22 `\bibitem`.
   No existe `referencias.bib` en el flujo de compilación.
2. **El preámbulo es el real**, no uno provisional: incluye `\newcolumntype{P}` para
   columnas proporcionales y `LTchunksize` para estabilidad de `longtable`. Las tres
   tablas del documento dependen de ambos, así que no se toca el preámbulo sin revisar
   las tablas.

## Cola de tareas

- **Task-010** — Auditar la síntesis del estado del arte en
  `secciones/02_estado_arte.tex` contra el `Protocolo PRISMA/` (137 referencias). El
  documento declara un corpus núcleo de 22; verificar que el criterio de reducción de 137
  a 22 esté enunciado en alguna parte, y si no lo está, señalarlo. [claude-1]
  **Resultado (NEED_REVIEW):** el criterio no estaba enunciado en ninguna parte. La
  `Bítacora_revision_sistematica.docx` documenta con precisión la cadena 319→221→201→137
  (deduplicación, filtro DOI, disponibilidad de texto completo), pero su Etapa 6
  ("Extracción de datos") solo remite a las planillas de trabajo y su Etapa 7
  ("Redacción del estudio") figura `Pendiente`; ninguna de las dos planillas
  (`Analisis 137 referencias.xlsx`, `Referencias_seminario.xlsx`) registra un umbral o
  regla explícita para la reducción de 137 a los 22 `\bibitem` de
  `secciones/99_bibliografia.tex` — solo una justificación cualitativa por referencia.
  Se agregó un párrafo a `secciones/02_estado_arte.tex` que declara la cadena numérica
  documentada, y declara explícitamente como pendiente (no ejecutada) la formalización
  del criterio de relevancia temática que redujo 137 a 22, en vez de omitir el vacío.
  Compila en 18 páginas. Ver PR abajo.
- **Task-011** — Validar que las cifras y métricas de `Prototipo_Preliminar.ipynb`
  coincidan con lo afirmado en `secciones/05_formalizacion.tex` y en
  `secciones/04_arquitectura.tex`. [claude-2]
- **Task-015** — Resolver la contradicción aritmética de paradigmas registrada como
  **C-001** más abajo. Requiere editar prosa en `secciones/00_resumen.tex`,
  `secciones/01_introduccion.tex` y `secciones/03_objetivos.tex` (las tres son
  propiedad de claude-1; claude-2 no las reescribe). Ver motivo detallado en C-001.
  [claude-1]

### Cerradas por claude-2 en esta sesión

- **Task-012** — Verificado: 22 claves de `\cite` distintas en uso, 22 `\bibitem` en
  `secciones/99_bibliografia.tex`, correspondencia 1:1. Cero `\cite` colgantes, cero
  `\bibitem` sin citar. Sin acción requerida.
- **Task-013** — Revisado el estado de avance en resumen, introducción, arquitectura,
  formalización y conclusiones: todos declaran el sistema de validación como diseñado y
  no ejecutado, de forma consistente. Se encontró una excepción en
  `secciones/A_matriz_literatura.tex` (fila `chefer2021transformer`), propiedad de
  claude-2: la columna "Aporte a la tesis" decía que el framework produce un "grafo
  relacional validado", lo que sugiere validación ya ejecutada. Corregido a "grafo
  relacional sujeto a validación multicriterio". Compilado y verificado (18 páginas).
- **Task-014** — Auditado. No se borra ningún número; ver **C-001** para el detalle y
  **Task-015** para la resolución, que corresponde a claude-1 por ser prosa ajena.

## PR abiertos

Un PR aquí es una petición de revisión dirigida al **otro** agente. Nadie mergea lo
propio. Ninguno de los dos recibe notificaciones, así que este listado es el único aviso
que existe: si no se anota, el PR queda esperando para siempre.

- PR #4 | rama: claude-2/review | autor: claude-2 | revisa: claude-1
  Toca: tesis/TASKS.md, secciones/A_matriz_literatura.tex
  Task: Task-012, Task-013, Task-014

Formato:

```
- PR #N | rama: claude-1/drafting | autor: claude-1 | revisa: claude-2
  Toca: secciones/02_estado_arte.tex
  Task: Task-010
```

## Contradicciones abiertas

- [C-001] Secciones implicadas: 00_resumen.tex, 01_introduccion.tex, 03_objetivos.tex,
  02_estado_arte.tex (`tab:familias`, fila (d))
  Descripción: el resumen y la introducción afirman una triangulación de **cinco**
  paradigmas (probabilístico, intervencional, geométrico-topológico, informacional,
  baselines deterministas). El objetivo específico 4 y la fila (d) de `tab:familias`
  afirman **cuatro** paradigmas "ortogonales al enfoque frecuentista" (intervencional,
  geométrico-topológico, informacional, baselines deterministas), sin el probabilístico.
  Auditoría: no es un error aritmético sino una distinción real no explicitada. El
  paradigma probabilístico/frecuentista ya se emplea en el eslabón 3 (modelo nulo
  binomial con corrección de Bonferroni/Benjamini-Hochberg, objetivo específico 3); el
  objetivo específico 4 triangula con cuatro paradigmas *adicionales* a ese, ortogonales
  a él. "Cinco" cuenta el total de la triangulación completa; "cuatro" cuenta los que se
  suman al probabilístico ya usado. Ambos números son correctos en su propio contexto,
  pero el texto no lo dice, por lo que un lector no puede reconstruir la distinción.
  Corrección propuesta (para quien la aplique, ver Task-015): en el objetivo específico 4
  y en la fila (d) de `tab:familias`, agregar una cláusula breve del tipo "...cuatro
  paradigmas ortogonales al enfoque frecuentista ya empleado en el eslabón de consenso
  estadístico (objetivo específico 3), que sumados a este completan la triangulación de
  cinco paradigmas declarada en el resumen y la introducción". No borrar "cinco" de
  00_resumen.tex/01_introduccion.tex ni "cuatro" de 03_objetivos.tex/02_estado_arte.tex.
  Detectada por: claude-2, auditoría de Task-014, 2026-09-07.
  Estado: ABIERTA — bloquea el merge de 00_resumen.tex, 01_introduccion.tex,
  03_objetivos.tex y 02_estado_arte.tex hasta que claude-1 aplique Task-015. No bloquea
  el resto del documento, que ya está mergeado a `main`.

Formato de registro:

```
- [C-001] Secciones implicadas: 01_introduccion.tex, 05_formalizacion.tex
  Descripción: la introduccion afirma X, la formalizacion afirma Y.
  Detectada por: claude-2 en PR #N
  Estado: ABIERTA | RESUELTA
```

Una contradicción abierta bloquea el merge de **todas** las secciones implicadas.

## Historial de decisiones

- **2026-09-07** — El repositorio se conecta a `GabrielSanzana/Seminario`. Se confirma
  permiso de escritura para el autor.
- **2026-09-07** — Fuente canónica elegida: el `.tex` de Overleaf que genera el PDF de 18
  páginas. `tesis_actualizada.tex` queda como referencia histórica, no como base.
- **2026-09-07** — Se instala TinyTeX local para que la auditoría pueda verificar
  compilación real y no solo sintaxis.
- **2026-09-07** — Los dos agentes trabajan en el mismo repositorio con ramas separadas y
  pull requests, para que las contradicciones se detecten en la revisión cruzada.
