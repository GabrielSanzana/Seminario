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

| Archivo | Estado | Dueño | Commit | Nota |
|---|---|---|---|---|
| `secciones/00_resumen.tex` | `BLOCKED` | — | — | Espera `.tex` canónico de Overleaf |
| `secciones/01_introduccion.tex` | `BLOCKED` | — | — | Espera `.tex` canónico de Overleaf |
| `secciones/02_estado_arte.tex` | `BLOCKED` | — | — | Espera `.tex` canónico de Overleaf |
| `secciones/03_objetivos.tex` | `BLOCKED` | — | — | Espera `.tex` canónico de Overleaf |
| `secciones/04_arquitectura.tex` | `BLOCKED` | — | — | Espera `.tex` canónico de Overleaf |
| `secciones/05_formalizacion.tex` | `BLOCKED` | — | — | Espera `.tex` canónico de Overleaf |
| `secciones/06_aplicabilidad.tex` | `BLOCKED` | — | — | Espera `.tex` canónico de Overleaf |
| `secciones/07_conclusiones.tex` | `BLOCKED` | — | — | Espera `.tex` canónico de Overleaf |
| `secciones/A_matriz_literatura.tex` | `BLOCKED` | — | — | Espera `.tex` canónico de Overleaf |

## Bloqueador activo

**Task-000 — Traer el `.tex` canónico desde Overleaf.** Responsable: autor humano.

El PDF `Template_Latex__Formato_Informes__Copy_ (1).pdf` (18 páginas, 7 secciones más
anexo A) es la versión vigente y reestructurada de la tesis, pero su archivo fuente `.tex`
no está en el repositorio: vive en Overleaf. El monolito local `tesis_actualizada.tex`
(1923 líneas) corresponde a una estructura anterior y distinta, con secciones que el PDF
ya no incluye.

Mientras el `.tex` canónico no esté en `tesis/`, el split de secciones no puede hacerse y
todas las secciones quedan `BLOCKED`. Los archivos actuales en `secciones/` son
plantillas vacías que solo garantizan que `main.tex` compile.

## Cola de tareas

Se desbloquean cuando Task-000 esté resuelta.

- **Task-001** — Partir el `.tex` canónico en los nueve archivos de `secciones/`, sin
  reescribir contenido: solo cortar y pegar. Verificar que el PDF resultante sea
  idéntico al original. [autor humano o claude-2]
- **Task-002** — Reemplazar el preámbulo provisional de `main.tex` por el preámbulo real
  del template. [autor humano]
- **Task-010** — Sintetizar el protocolo PRISMA (137 referencias) en
  `secciones/02_estado_arte.tex`, con la lectura de la brecha. [claude-1]
- **Task-011** — Validar que las cifras y métricas de `Prototipo_Preliminar.ipynb`
  coincidan con lo afirmado en `secciones/05_formalizacion.tex`. [claude-2]
- **Task-012** — Verificar que `referencias.bib` cubra todos los `\cite` del documento y
  que no haya entradas huérfanas. [claude-2]
- **Task-013** — Revisar que el estado de avance declarado sea consistente entre
  introducción, formalización y conclusiones: el sistema de validación está diseñado y
  especificado, no ejecutado. [claude-2]

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
