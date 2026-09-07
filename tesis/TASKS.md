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
| `secciones/02_estado_arte.tex` | `IN_PROGRESS` | claude-1 | 68 | Task-010: auditoría de la síntesis PRISMA |
| `secciones/03_objetivos.tex` | `APPROVED` | claude-1 | 18 | Split verificado |
| `secciones/04_arquitectura.tex` | `APPROVED` | claude-2 | 42 | Split verificado |
| `secciones/05_formalizacion.tex` | `APPROVED` | claude-2 | 35 | Split verificado |
| `secciones/06_aplicabilidad.tex` | `APPROVED` | claude-1 | 31 | Incluye `tab:dominios` |
| `secciones/07_conclusiones.tex` | `APPROVED` | claude-1 | 10 | Split verificado |
| `secciones/99_bibliografia.tex` | `APPROVED` | claude-2 | 47 | 22 `\bibitem`, append-only |
| `secciones/A_matriz_literatura.tex` | `APPROVED` | claude-2 | 41 | Incluye `tab:matriz` |

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
- **Task-011** — Validar que las cifras y métricas de `Prototipo_Preliminar.ipynb`
  coincidan con lo afirmado en `secciones/05_formalizacion.tex` y en
  `secciones/04_arquitectura.tex`. [claude-2]
- **Task-012** — Verificar que cada clave de `\cite` resuelva a un `\bibitem` de
  `secciones/99_bibliografia.tex` y que no haya `\bibitem` sin citar. [claude-2]
- **Task-013** — Revisar consistencia del estado de avance entre resumen, introducción,
  formalización y conclusiones: el sistema de validación está diseñado y especificado, no
  ejecutado. Cualquier frase que sugiera lo contrario es una contradicción. [claude-2]
- **Task-014** — Revisar la aritmética de los paradigmas. El resumen y la introducción
  hablan de **cinco** paradigmas de triangulación; el objetivo específico 4 habla de
  **cuatro** ortogonales al enfoque frecuentista, y la fila (d) de `tab:familias` dice
  cuatro. Determinar si es la distinción entre "cinco en total incluyendo el
  probabilístico" y "cuatro además de él", y si lo es, hacerla explícita en el texto.
  [claude-2]

## PR abiertos

Un PR aquí es una petición de revisión dirigida al **otro** agente. Nadie mergea lo
propio. Ninguno de los dos recibe notificaciones, así que este listado es el único aviso
que existe: si no se anota, el PR queda esperando para siempre.

Ninguno abierto.

Formato:

```
- PR #N | rama: claude-1/drafting | autor: claude-1 | revisa: claude-2
  Toca: secciones/02_estado_arte.tex
  Task: Task-010
```

## Contradicciones abiertas

Ninguna registrada.

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
