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
| `secciones/02_estado_arte.tex` | `APPROVED` | claude-1 | 76 | Task-015 mergeado a `main` en PR #6; C-001 resuelta |
| `secciones/03_objetivos.tex` | `APPROVED` | claude-1 | 24 | Task-015 mergeado a `main` en PR #6; C-001 resuelta |
| `secciones/04_arquitectura.tex` | `APPROVED` | claude-2 | 42 | Split verificado |
| `secciones/05_formalizacion.tex` | `APPROVED` | claude-2 | 35 | Split verificado |
| `secciones/06_aplicabilidad.tex` | `APPROVED` | claude-1 | 31 | Incluye `tab:dominios` |
| `secciones/07_conclusiones.tex` | `APPROVED` | claude-1 | 10 | Split verificado |
| `secciones/99_bibliografia.tex` | `APPROVED` | claude-2 | 47 | 22 `\bibitem`, append-only |
| `secciones/A_matriz_literatura.tex` | `APPROVED` | claude-2 | 41 | Task-013 mergeado a `main` en PR #4 |

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

- **Task-016** — Resolver la contradicción de alcance registrada como **C-002** más
  abajo: el resumen, la introducción y las conclusiones declaran que la cadena de
  evidencia completa está "sin ejecutar", pero `Prototipo_Preliminar.ipynb` ya ejecutó
  el primer eslabón con resultados cuantitativos. Requiere editar prosa en
  `secciones/00_resumen.tex`, `secciones/01_introduccion.tex` y
  `secciones/07_conclusiones.tex` (propiedad de claude-1; claude-2 no las reescribe).
  Ver motivo detallado en C-002. [claude-1]
- **Task-017** — Resolver la contradicción de corpus registrada como **C-003** más
  abajo: reemplazar en la prosa las referencias que no pertenecen al corpus vigente de
  `Protocolo PRISMA/Referencias_seminario.xlsx` por las que sí, leyendo los PDF en
  `PDF seleccionados para el seminario/`. Afecta sobre todo
  `secciones/02_estado_arte.tex` (propiedad de claude-1); revisar también
  `secciones/00_resumen.tex` y `secciones/01_introduccion.tex` por si citan alguna de
  las 19 referencias obsoletas. Los `\bibitem` nuevos que hagan falta se piden aquí en
  `TASKS.md` (título + DOI) para que claude-2 los agregue a
  `secciones/99_bibliografia.tex` (append-only, no lo edita claude-1 directamente). Ver
  motivo detallado en C-003. [claude-1]

(Task-010, Task-011, Task-012, Task-013, Task-014 y Task-015 completadas y mergeadas;
ver `Estado de secciones` arriba y las notas de cierre abajo. Se sacan de la cola para
que no se vuelvan a tomar por error.)

### Cerradas por claude-1 en esta sesión

- **Task-015** — Aplicado el fix propuesto en **C-001**: se agregó, en
  `secciones/03_objetivos.tex` (objetivo específico 4), la cláusula "...ortogonales al
  enfoque frecuentista ya empleado en el eslabón de consenso estadístico (objetivo
  específico anterior)... que sumados a ese enfoque completan la triangulación de cinco
  paradigmas declarada en el resumen y la introducción"; y en `secciones/02_estado_arte.tex`
  (fila (d) de `tab:familias`) la coletilla "...adicionales al probabilístico de (c), que
  juntos completan los cinco". No se tocó `00_resumen.tex` ni `01_introduccion.tex`: ya
  eran correctos y C-001 pide expresamente no alterar su "cinco". Compila en 18 páginas.
  Pendiente de revisión de claude-2 (ver PR abajo); C-001 sigue `ABIERTA` hasta que se
  confirme y mergee.

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
- **Task-011** — Cotejadas las cifras de `secciones/04_arquitectura.tex` y
  `secciones/05_formalizacion.tex` contra `Prototipo_Preliminar.ipynb`: coinciden
  exactamente. Encoder Conv2D 3×3 stride 2 (dos capas, 32→embed_dim), `d_model=128`,
  `num_heads=4`, `num_layers=2`, decoder ConvTranspose2D×2 + interpolación bicúbica
  (`F.interpolate(..., mode="bicubic")`), rollout con fracción residual
  ($\gamma I + (1-\gamma)\bar A$, `residual_fraction` en el código) — todo verificado
  línea por línea. La discontinuidad del operador Top-K que describe
  `05_formalizacion.tex` es exactamente el bug que el notebook documenta y corrige
  ("FIX BUG: Top-K fijo en vez de umbral mean+std"). Sin acción requerida sobre estos
  dos archivos. Se encontró, en cambio, una contradicción de alcance en archivos ajenos
  (00/01/07): ver **C-002** y **Task-016**.

## PR abiertos

Un PR aquí es una petición de revisión dirigida al **otro** agente. Nadie mergea lo
propio. Ninguno de los dos recibe notificaciones, así que este listado es el único aviso
que existe: si no se anota, el PR queda esperando para siempre.

- PR #7 | rama: claude-2/review | autor: claude-2 | revisa: claude-1
  Toca: .gitignore, elimina `.claude/settings.local.json`
  Task: — (PR #4 se mergeó con ese archivo adentro por una carrera de tiempos; #7 lo
  saca de `main`)

- PR #9 | rama: claude-2/review | autor: claude-2 | revisa: claude-1
  Toca: CLAUDE.md, tesis/TASKS.md
  Task: — (fija Referencias_seminario.xlsx como corpus vigente; ver C-003 y Task-017)
  Nota: urgente de revisar/mergear pronto — mientras esté sin mergear, la sesión de
  claude-1 sigue viendo las reglas viejas y puede seguir citando del corpus obsoleto.

(PR #4 revisado y mergeado por claude-1: cifras cotejadas contra
`Prototipo_Preliminar.ipynb` línea por línea, ver comentario en el PR.
PR #6 —.gitignore del autor humano + Task-015 de claude-1 agregado como commits
adicionales porque GitHub no dejó abrir un segundo PR desde la misma rama— revisado
y mergeado por claude-2 a `main`.)

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
  Estado: RESUELTA — claude-2 revisó el fix de claude-1 (Task-015) línea por línea:
  en `03_objetivos.tex` la cláusula ancla el "cuatro" al "enfoque frecuentista ya
  empleado en el eslabón de consenso estadístico (objetivo específico anterior)" y
  cierra con "que sumados a ese enfoque completan la triangulación de cinco
  paradigmas"; en `02_estado_arte.tex` (fila (d) de `tab:familias`) queda "cuatro
  paradigmas adicionales al probabilístico de (c), que juntos completan los cinco" —
  ancla directo a la fila (c), que es precisamente el modelo nulo/binomial. Ningún
  número se borró. Mergeado a `main` en PR #6.

- [C-002] Secciones implicadas: 00_resumen.tex, 01_introduccion.tex, 07_conclusiones.tex
  Descripción: las tres afirman, en bloque, que la cadena de evidencia completa (los
  cinco eslabones y la triangulación de paradigmas) está "diseñada y especificada" pero
  "aún no ejecutada", y que su ejecución es "el programa de trabajo de la tesis" — como
  si nada se hubiera corrido todavía. `Prototipo_Preliminar.ipynb` contradice ese
  blanqueo total: contiene dos experimentos ejecutados (10 y 50 semillas,
  `stability_experiment_50seeds_v2.py`) que corresponden al primer eslabón (estabilidad
  continua de $A$) con resultados cuantitativos reales: Jaccard promedio 0.195-0.204,
  arista dominante presente en 76-78% de los modelos, "núcleo estable pero periferia
  varía". El propio notebook concluye sobre esta corrida: "Esto valida el supuesto
  central de la tesis" (celda de 10 semillas). El mismo script de 50 semillas también
  computa análisis espectral (autovalores del Laplaciano, gap espectral) y estructural
  de grafo (PageRank, HITS, comunidades Louvain, roles nodales hub/source/bridge,
  reciprocidad, modularidad) — un prototipo parcial del paradigma geométrico-topológico
  de la triangulación, no solo del primer eslabón.
  Lo que SÍ sigue sin ejecutarse, confirmado por ausencia total en el notebook (cero
  ocurrencias de los términos): el modelo nulo binomial con corrección de
  Bonferroni/Benjamini-Hochberg (eslabón 3), la ablación de fidelidad con
  renormalización (eslabón 4), y los tres baselines deterministas restantes de la
  triangulación (Pearson, Granger, Erdős-Rényi por Z-score) junto con los paradigmas
  intervencional e informacional. Sobre esa parte, "diseñado, no ejecutado" es preciso.
  Auditoría: no corresponde borrar "no ejecutado" en general, sino dejar de aplicarlo de
  forma pareja a los cinco eslabones. El eslabón 1 (y parcialmente el paradigma
  geométrico-topológico) tiene evidencia preliminar real y citable; los eslabones 3 y 4
  y el resto de los baselines no. Tratar ambos como igualmente "sin ejecutar" subestima
  el avance real del prototipo y desaprovecha evidencia que sostiene justamente la
  premisa de inestabilidad entre semillas que 01_introduccion.tex plantea en "El
  problema".
  Corrección propuesta (para quien la aplique, ver Task-016): matizar la frase de
  cierre de 01_introduccion.tex ("...aún no ejecutado") para acotarla a los eslabones
  3-5 y a la triangulación de baselines, y citar el resultado preliminar del eslabón 1
  (Jaccard ~0.20, núcleo estable en ~76-78% de los modelos sobre 50 semillas) como
  evidencia ya obtenida en el prototipo. Aplicar la misma precisión en 00_resumen.tex y
  07_conclusiones.tex. No eliminar la afirmación de que el sistema multicriterio en su
  conjunto sigue pendiente de ejecución: eso sigue siendo cierto para eslabones 3-5.
  Detectada por: claude-2, auditoría de Task-011 sobre `Prototipo_Preliminar.ipynb`,
  2026-09-07.
  Estado: ABIERTA — bloquea el merge de 00_resumen.tex, 01_introduccion.tex y
  07_conclusiones.tex hasta que claude-1 aplique Task-016. No bloquea el resto del
  documento, que ya está mergeado a `main`.

- [C-003] Secciones implicadas: 02_estado_arte.tex (principal), 99_bibliografia.tex,
  A_matriz_literatura.tex, y por revisar 00_resumen.tex/01_introduccion.tex
  Descripción: `secciones/99_bibliografia.tex` trae 22 `\bibitem` heredados del
  documento monolítico previo al flujo multiagente. Cotejados por DOI contra
  `Protocolo PRISMA/Referencias_seminario.xlsx` (el corpus vigente, 25 referencias,
  todas `Leído`, con PDF completo en `PDF seleccionados para el seminario/`):
  **solo 3 de los 22 coinciden** — `reichstein2019deep` (fila 24),
  `liu2024itransformer` (fila 3), `zhao2026causalguided` (fila 16). Los 19 restantes
  (`lapuschkin2019unmasking`, `abnar2020quantifying`, `slack2020fooling`,
  `chefer2021transformer`, `jakubowski2022performance`, `cai2024msgnet`,
  `vrahatis2024graph`, `petrosian2024solar`, `tew2024kans`, `zhu2025attention`,
  `chatterjee2025multicriteria`, `klotz2025xai`, `han2026tscad`, `qin2026multisensor`,
  `ewuzie2026robust`, `zhang2026ehtgnn`, `kapoor2026fustt`, `zhou2026causal`,
  `khayitov2026stgeonet`) no están en `Referencias_seminario.xlsx`: pertenecen al
  corpus obsoleto (`Analisis 137 referencias.xlsx` / `PDF primera iteración/`) que
  `CLAUDE.md` ahora excluye explícitamente. En sentido inverso, 22 de las 25 filas del
  excel vigente no se citan en ninguna parte del documento — todas menos las filas 3
  (`liu2024itransformer`), 16 (`zhao2026causalguided`) y 24 (`reichstein2019deep`).
  Filas sin citar, con su PDF correspondiente en `PDF seleccionados para el seminario/`
  bajo el mismo número: 1 Quantifying Attention Flow in Transformers; 2 Transformer
  Interpretability Beyond Attention Visualization; 4 Mechanistic Interpretability for
  Transformer-based Time Series Classification; 5 Evaluating the Faithfulness of
  Causality in Saliency-Based Explanations of Deep Learning Models for Temporal Colour
  Constancy; 6 Instability and interpretability discrepancies between CNNs and vision
  transformers in keratoconus detection; 7 Explainability and Evaluation of Vision
  Transformers: An In-Depth Experimental Study; 8 Explanation Variability in Text
  Classification: Humans vs. LLMs; 9 Fooling LIME and SHAP: Adversarial Attacks on Post
  hoc Explanation Methods; 10 Unmasking Clever Hans Predictors and Assessing What
  Machines Really Learn; 11 Understanding Transformer-Based Classifications of Medical
  Text Using a Large Language Model for the Attribution of Feature Importance:
  Proof-of-Concept Algorithm Development and Validation Study; 12 A comprehensive
  analysis of perturbation methods in explainable AI feature attribution validation
  for neural time series classifiers; 13 Explaining time series classifiers through
  meaningful perturbation and optimisation; 14 Dynamic Causal Graph Network for
  Reliable Pipeline Leak Detection; 15 MSGNet: Learning Multi-Scale Inter-Series
  Correlations for Multivariate Time Series Forecasting; 17 A global model-agnostic
  rule-based XAI method based on Parameterized Event Primitives for time series
  classifiers; 18 SENTINEL: Multi-Patch Transformer with Temporal and Channel
  Attention for Time Series Forecasting; 19 Unlocking the Power of Patch: Patch-Based
  MLP for Long-Term Time Series Forecasting; 20 ConvLSTM-GCN-Transformer:
  Spatiotemporal graph-attention model for vegetation index map forecasting; 21
  Disentangling Regional Impacts of Joint Teleconnections Using Causal Representation
  Learning; 22 An end-to-end explainability framework for spatio-temporal predictive
  modeling; 23 Explainable AI in Rotorcraft Aerodynamics: Autonomous Discovery and
  Dynamic Tracking of Vortex Ring State Mechanisms via Vision Transformers; 25 Remote
  Sensing for Precision Agriculture: Sentinel-2 Improved Features and Applications.
  Los 19 `\bibitem` obsoletos con su DOI actual, para localizarlos en
  `99_bibliografia.tex`: lapuschkin2019unmasking, abnar2020quantifying,
  slack2020fooling, chefer2021transformer, jakubowski2022performance, cai2024msgnet,
  vrahatis2024graph, petrosian2024solar, tew2024kans, zhu2025attention,
  chatterjee2025multicriteria, klotz2025xai, han2026tscad, qin2026multisensor,
  ewuzie2026robust, zhang2026ehtgnn, kapoor2026fustt, zhou2026causal,
  khayitov2026stgeonet.
  Auditoría: esto no es un error puntual, es que la selección de corpus completa que
  sostiene `secciones/02_estado_arte.tex` (las seis "familias metodológicas" y
  `tab:familias`) se construyó sobre el corpus viejo antes de que
  `Referencias_seminario.xlsx` existiera como fuente vigente. La matriz de revisión
  sistemática en `secciones/A_matriz_literatura.tex` (`tab:matriz`, propiedad de
  claude-2) también está construida fila por fila sobre esas mismas 22 referencias
  obsoletas, y habrá que reconstruirla una vez que claude-1 reemplace el corpus citado
  en la prosa.
  Corrección propuesta (para quien la aplique, ver Task-017): claude-1 revisa las 25
  filas de `Referencias_seminario.xlsx` (con ayuda de la hoja `Aporte por sección`, que
  marca a qué sección aporta cada una) y reescribe `02_estado_arte.tex` (y lo que
  corresponda de 00/01) citando ese corpus en vez del heredado. Pide en `TASKS.md` los
  `\bibitem` que falten (título + DOI) para que claude-2 los agregue a
  `99_bibliografia.tex`. Los 3 `\bibitem` ya vigentes (reichstein2019deep,
  liu2024itransformer, zhao2026causalguided) se conservan; los 19 obsoletos se retiran
  de la prosa cuando ya no los cite nadie, y claude-2 los quita de
  `99_bibliografia.tex` recién en ese momento (append-only no impide borrar un
  `\bibitem` que quedó sin ningún `\cite`, solo impide reordenar los que quedan).
  Detectada por: claude-2, a pedido explícito del autor humano, 2026-09-07.
  Estado: ABIERTA — bloquea el merge de cambios futuros a 02_estado_arte.tex y
  A_matriz_literatura.tex hasta que claude-1 aplique Task-017 y claude-2 reconstruya
  `tab:matriz` sobre el corpus vigente. No revierte lo ya mergeado a `main`.

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
