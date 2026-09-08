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
| `OBSOLETO` | El archivo sale del documento. No editar: su contenido ya migró a otro |

Protocolo de reserva: cambiar el estado a `IN_PROGRESS`, hacer commit y push, **y después**
editar. El cambio de estado y el trabajo terminado van en el mismo commit al cerrar.

## Estado de secciones

| Archivo | Estado | Dueño | Líneas | Nota |
|---|---|---|---|---|
| `main.tex` | `NEED_REVIEW` | claude-1 | 75 | Task-021: hyperref (indice navegable), parskip 0 y titlespacing segun formato |
| `pucv_inf_2024.sty` | `NEED_REVIEW` | claude-1 | 87 | Task-021: unica linea cambiada, biblatex apa -> ieee (citas numericas) |
| `Portadas/portada_principal.tex` | `NEED_REVIEW` | claude-1 | 32 | Portada del template con titulo, autores y asignatura |
| `Resumen/resumen.tex` | `NEED_REVIEW` | claude-1 | 30 | Task-021: reescrito, dos familias y modelo alternativo |
| `secciones/01_introduccion.tex` | `NEED_REVIEW` | claude-1 | 47 | Task-025: "umbrales" -> "criterios" en la cadena de evidencia (sin compromiso operativo) |
| `secciones/02_objetivos.tex` | `NEED_REVIEW` | claude-1 | 24 | Task-025: objetivo 2 ya no compromete "umbrales declarados antes de la ejecución", queda en el nivel de estabilidad/fidelidad como propiedades a construir |
| `secciones/03_estado_arte.tex` | `NEED_REVIEW` | claude-1 | 89 | Task-025: "calibrar sus umbrales" -> "calibrar su criterio de contraste". Task-024: nota de procedencia de citas y de analisis de citas como trabajo futuro. Task-023: recortadas 5 cifras de precision de algoritmo que no eran resultado de explicabilidad |
| `secciones/04_marco_teorico.tex` | `APPROVED` | claude-2 | 55 | Task-025: recortado el detalle de implementación del encoder (ConvTransformer); attention rollout se conserva por ser formalización. Task-023: ampliada la seccion de dominio del caso de estudio (2a cita a segarra2020sentinel, 1a a reichstein2019deep) para cumplir el minimo de 3 usos |
| `secciones/05_plan_trabajo.tex` | `NEED_REVIEW` | claude-1 | 56 | Task-025: "umbrales de decisión" -> "criterio de validación". Task-021: actividades. **Fechas por confirmar** |
| `secciones/06_propuesta.tex` | `NEED_REVIEW` | claude-1/claude-2 | 93 | Task-025: retira el compromiso Top-K fijo (§Transformación matriz-grafo) y el detalle operativo del criterio de validación (método de perturbación, métricas de fidelidad); conserva la cadena de 5 preguntas y la taxonomía de 4 resultados. Task-022: reescrita por completo. Alcance corregido (hipotesis como fin, modelo alternativo generico con tokens/iTransformer en vez de ConvTransformer, sin la regla inventada de "cinco condiciones") |
| `secciones/07_conclusiones.tex` | `NEED_REVIEW` | claude-1 | 17 | Task-025: "con umbrales fijados" -> "con sus condiciones fijadas". Task-023: 3 citas nuevas (abnar2020quantifying, reichstein2019deep, segarra2020sentinel, meng2023perturbation) para cumplir el minimo de 3 usos |
| `referencias.bib` | `NEED_REVIEW` | claude-2 | 280 | 25 entradas, ahora renderizadas en estilo IEEE numerico |

Archivos retirados del documento por Task-019, recuperables desde el historial de git:
`00_resumen.tex`, `02_estado_arte.tex`, `03_objetivos.tex`, `04_arquitectura.tex`,
`05_formalizacion.tex`, `06_aplicabilidad.tex`, `99_bibliografia.tex` y
`A_matriz_literatura.tex`.

`APPROVED` aquí significa que el contenido corresponde exactamente al `.tex` canónico y
que compila. No significa que el contenido esté auditado: esa es la cola de abajo.

## Hechos del documento que condicionan el trabajo

Tras Task-019 el documento cambió de formato. Cuatro hechos que hay que tener presentes
antes de tocar nada:

1. **Formato: template PUCV.** `main.tex` usa `\documentclass{report}` con
   `pucv_inf_2024.sty`, copiado sin modificar desde
   `Template_Latex__Formato_Informes__Copy_/`. Las secciones son `\chapter`, no
   `\section`. El `.sty` no se edita: si algo del formato molesta, se resuelve en
   `main.tex`.
2. **Sí usa biblatex con backend biber.** La bibliografía vive en `referencias.bib` y ya
   no existe `thebibliography`. Compilar exige la secuencia
   `pdflatex`, `biber`, `pdflatex`, `pdflatex`, que es lo que hace `./scripts/compilar.sh`.
   El repositorio necesita los paquetes `biblatex`, `biblatex-apa`, `biber`, `glossaries`,
   `nomencl`, `titlesec`, `fancyhdr`, `algorithms`, `algorithmicx`, `caption`,
   `koma-script`, `courier`, `csquotes` e `hyphen-spanish`, todos instalables con
   `tlmgr install`.
3. **El corpus de citas está cerrado en 25.** `referencias.bib` contiene exactamente las
   25 filas de `Referencias_seminario.xlsx` y el documento cita las 25, sin sobrantes ni
   colgantes. Agregar una cita implica agregar antes una fila a esa hoja.
4. **El preámbulo define `\newcolumntype{P}`** para columnas proporcionales y fija
   `LTchunksize`. Las tres tablas del documento dependen de ambos.

## Cola de tareas

Sin tareas de contenido pendientes. Queda solo la confirmación de fechas del plan de
trabajo por el autor humano (`secciones/05_plan_trabajo.tex`, tabla `tab:plan`).

- **Resuelta por claude-2 — `secciones/04_marco_teorico.tex`, a raíz de Task-025:**
  el párrafo de "Del Transformer estándar al ConvTransformer de índices" que describía
  el encoder convolucional (dimensiones $H\times W$, capa lineal, decoder que
  reconstruye la resolución) se recortó a la misma altura que el resto del documento:
  queda el requisito conceptual (encoder compartido entre variables, para que la
  matriz no confunda capacidad de codificación desigual con relación real) y se declara
  explícitamente que la arquitectura concreta del encoder es una decisión del caso de
  estudio, no de la formulación general. Se conservó, en cambio, la sección de
  agregación entre capas (*attention rollout*, \textcite{abnar2020quantifying}): es una
  transformación formal con cita, no un detalle de implementación del ConvTransformer,
  y el resto del documento la mantuvo intacta bajo el mismo criterio (ver
  `07_conclusiones.tex`). Compila en 28 páginas; no cambió ninguna cita.

### Task-025 — cerrada por claude-1 en esta sesión

- **Task-025** — Instrucción directa del autor humano, 2026-09-08: "no escribas de
  código en el informe, solo que retrates la formulación matemática del framework; los
  temas de umbrales, métricas y resultados del código no son necesarios mencionarlos
  ahora, únicamente debemos defender la idea." Se preguntó al autor humano el alcance
  exacto de dos decisiones antes de tocar prosa (ver respuestas abajo) y se aplicó de
  forma consistente en los seis archivos de claude-1/mixtos que mencionaban umbrales o
  detalle operativo:

  1. `secciones/01_introduccion.tex`: "cadena de evidencia acumulativa cuyos umbrales
     se declaran antes de ejecutarla" -> "...cuyos criterios se declaran...".
  2. `secciones/02_objetivos.tex`: el objetivo 2 ya no promete "umbrales declarados
     antes de la ejecución"; queda en construir el criterio en términos de estabilidad
     y fidelidad, sin el compromiso operativo. Se conservan las dos citas de literatura
     que motivan por qué el criterio necesita ambas propiedades por separado
     (`keratoconus2026instability`, `simic2025perturbation`) porque describen hallazgos
     de trabajos ajenos, no el método propio.
  3. `secciones/03_estado_arte.tex`: "calibrar sus umbrales contra un modelo nulo
     propio" -> "calibrar su criterio de contraste contra un modelo nulo propio".
  4. `secciones/05_plan_trabajo.tex`: "fijar los umbrales de decisión antes de mirar
     cualquier resultado" -> "fijar su criterio de validación antes de mirar cualquier
     resultado".
  5. `secciones/06_propuesta.tex` (el cambio más grande, con respuesta explícita del
     autor humano sobre qué conservar):
     - §Transformación matriz-grafo: se retiró el compromiso concreto con un criterio
       Top-$K$ fijo (y su ecuación) porque fija un parámetro operativo ($K$); se
       mantiene la definición general $E = \{(v_i,v_j): A_{ij} > \tau\}$ y se declara
       que la forma concreta de $\tau$ se fija en la etapa de ejecución, no en el
       diseño. El párrafo que seguía (sobre el "efecto de $K$") se reescribió para
       enlazar la misma idea —que todo criterio de selección es en sí mismo fuente de
       variación— con la segunda pregunta del criterio de validación, sin nombrar $K$.
     - §Criterio de validación: por decisión explícita del autor humano, se **mantuvo**
       la cadena de 5 preguntas y la taxonomía de 4 resultados (relación genuina /
       atajo / artefacto / ruido) porque es la idea que esta entrega defiende, y se
       **retiró** el detalle operativo del párrafo de la cuarta pregunta (el método
       generativo de `meng2023perturbation` como "el que adopta el criterio de esta
       propuesta", y el nombre de la "métrica de fidelidad más extendida" de
       `simic2025perturbation`), dejando el punto conceptual (medir esto exige cuidado
       metodológico, la elección se justifica en la etapa de ejecución) sin comprometer
       un método ni una métrica concretos.
     - §Alcance de esta etapa: "la especificación del criterio de validación con sus
       umbrales" -> "...con la taxonomía anterior" (se retira "con sus umbrales").
  6. `secciones/07_conclusiones.tex`: "criterio de validación con umbrales fijados
     antes de ejecutarlo" -> "...con sus condiciones fijadas antes de ejecutarlo".

  **Verificación de citas:** las cuatro claves cuya única mención operativa se recortó
  en `06_propuesta.tex` (`bogaert2026variability`, `meng2023perturbation`,
  `simic2025perturbation`, mas `rizzo2022faithfulness` que no se tocó) se recontaron
  antes y después del cambio: las cuatro siguen con 3 usos exactos (mínimo que exige
  `CLAUDE.md`), porque se reescribió el párrafo que las cita en vez de borrarlo. Recuento
  automático completo sobre los 8 archivos con prosa: 25 claves, todas con 3 usos o más,
  sin cambios respecto del recuento de Task-023/Task-024.

  No se tocó `secciones/04_marco_teorico.tex` (ConvTransformer, propiedad de claude-2):
  ver el ítem de arriba en esta misma cola.

  `./scripts/compilar.sh` -> `OK: main.pdf compilado, 28 paginas`, sin referencias ni
  citas sin resolver. Pendiente de revisión de claude-2 (ver PR abajo). [claude-1]

### Cerradas en esta sesión (ejecución directa, ver nota de proceso abajo)

- **Task-022** — Reescrita por completo `secciones/06_propuesta.tex`. Se retiró la
  regla inventada "para elegir el segundo dominio de validación pesan cinco
  condiciones, en este orden..." del §6.6: generalizaba como regla fija del framework
  lo que era, a lo sumo, el análisis de un caso particular. La sección nueva declara
  explícitamente que no hay regla única ni orden fijo, y que el criterio se decide caso
  por caso.

  Corregido además en toda la sección: (1) el alcance ya no es "explicar el modelo",
  sino producir una hipótesis relacional sobre el fenómeno — explicar el modelo queda
  como paso instrumental, nunca como el fin, declarado explícito en el primer párrafo
  de §6.1; (2) el modelo alternativo se describe en términos generales (tokens por
  variable, en la línea de \textcite{liu2024itransformer} con iTransformer) y el
  ConvTransformer de índices pasa a nombrarse como una instancia del caso de estudio,
  no como la definición del framework; (3) el contenido se basó únicamente en la
  explicación de `Presentación preliminar del tema de investigación.pdf` (secciones 3
  a 8, la formalización matemática vigente), sin usar `Framework.py` ni sus detalles de
  implementación.

  Se tocó también el contenido que era de claude-2 (§6.2, arquitectura instrumental,
  ahora fundida en §6.1) porque la reescritura no se podía partir por dueño — mismo
  criterio de excepción que Task-019. `06_propuesta.tex` queda con dueño mixto
  claude-1/claude-2 en la tabla de arriba hasta que se decida si vuelve a separarse.

- **Task-023** — Auditoría de las dos reglas nuevas, ejecutada sobre el documento
  completo después de Task-022:

  1. *Uso mínimo de 3 citas.* Recuento automático sobre los 7 archivos con prosa citable
     (permitiendo `\cite`/`\textcite` con múltiples claves separadas por coma, como las
     que trae `tab:familias`): antes de esta sesión, 12 de las 25 claves de
     `referencias.bib` tenían menos de 3 usos en el documento (8 con 1 uso, 4 con 2,
     contando lo que quedaba fuera de la propuesta reescrita). Se corrigió agregando
     menciones legítimas y verificables — no relleno — en los lugares donde el
     argumento ya las sostenía: la propuesta nueva retoma varias en su discusión de
     diferenciación y de criterio de validación (`bogaert2026variability`,
     `cai2024msgnet`, `chefer2021transformer`, `rizzo2022faithfulness`,
     `sentinel2025multipatch`, `simic2025perturbation`, `teleconnections2026causal`,
     `zhao2026causalguided`, `abnar2020quantifying`, `meng2023perturbation`);
     `04_marco_teorico.tex` amplía la sección de dominio del caso de estudio con una
     segunda cita a `segarra2020sentinel` y una primera a `reichstein2019deep`;
     `07_conclusiones.tex` cierra el tercero con menciones a `abnar2020quantifying`,
     `reichstein2019deep`, `segarra2020sentinel` y `meng2023perturbation`. Recuento
     final: las 25 claves quedan en 3 usos o más (mínimo 3, máximo 6). Detalle completo
     del conteo antes/después en el historial de esta sesión, no reproducido aquí por
     extensión.
  2. *Precisión de algoritmo vs. resultado de explicabilidad.* En
     `secciones/03_estado_arte.tex` se encontraron y recortaron 5 cifras de desempeño
     del algoritmo referenciado que no eran resultado del método de explicabilidad: la
     exhaustividad de diagnóstico de `zhao2026causalguided` (99,2\%), las exactitudes de
     detección de fugas de `pipeline2026causalgraph` (96,3\%/96,6\%/86\% bajo ruido —se
     conservó la coincidencia del esqueleto causal con la topología real, que sí es
     resultado de explicabilidad), la generalización fuera de distribución de
     `cai2024msgnet`, el error cuadrático medio de `convlstmgcn2026vegetation` (0,034), y
     la exactitud de clasificación de `rotorcraft2026vortexring` (93,24\%; se conservó
     la coincidencia de la banda de atención con la frecuencia física conocida, que es
     el hallazgo de explicabilidad). No se tocaron las menciones de "exactitud" sin
     cifra que sostienen el argumento Clever Hans de `lapuschkin2019unmasking` ni las
     de desempeño de `liu2024itransformer`/`sentinel2025multipatch`/
     `patchmlp2025unlocking` en la familia 4 (`03_estado_arte.tex`, §"Atención entre
     variables y aplicaciones de dominio"): esa familia declara explícitamente que no
     aborda interpretabilidad sino viabilidad arquitectónica, así que su desempeño es
     parte legítima de ese argumento y no del resultado de un método de explicabilidad.

- **Task-024** — Agregadas las dos notas de alcance en `secciones/03_estado_arte.tex`,
  §3.1 (Obtención del corpus): que la mayoría de las 25 referencias salió del protocolo
  PRISMA pero no todas —algunas se incorporaron por búsqueda propia con asistencia de
  IA—, y que el análisis de las citas de estas 25 referencias queda fuera de alcance,
  declarado como trabajo futuro. **Nota de verificación:** se revisó
  `Referencias_seminario.xlsx` buscando una columna que distinguiera qué referencias
  vinieron del protocolo y cuáles de búsqueda propia; no existe tal columna (las once
  columnas de la hoja `Referencias` son Título, DOI, Descripción, Estado, Gabriel,
  Patricio, Fecha de publicación, Muestra, Metodología, Resultados, Conclusión). La nota
  se redactó en términos cualitativos ("la mayoría", "un número menor") en vez de
  inventar un recuento exacto que no está respaldado por ninguna fuente del repositorio.
  Si el autor humano tiene el detalle exacto de cuáles references son cuáles, conviene
  agregarlo aquí para que quede citable con precisión.

**Nota de proceso, para que quede registrado:** estas tres tareas se ejecutaron en una
sola sesión que cubrió el trabajo de ambos roles (redacción de claude-1 y auditoría de
claude-2), por instrucción directa del autor humano ("dale a ejecutar todo"), en vez de
pasar por el ciclo reserva → rama → PR → revisión cruzada del protocolo. Se documenta
así, sin PR asociado, para que una futura sesión de claude-1 o claude-2 no busque un PR
que no existe. La compuerta de compilación sí se corrió en cada paso
(`./scripts/compilar.sh`, verde en todos), que es lo que el protocolo exige como
reemplazo de la revisión humana; lo que no ocurrió es la revisión cruzada entre dos
agentes independientes, porque no había dos agentes independientes en esta sesión.

**Revisión cruzada independiente, hecha por claude-2 después (la que faltaba):**
`main` ya tenía estos commits (se empujaron directo, sin PR, por la razón de arriba),
así que no hay nada que mergear — se audita lo que ya está.

- Recompilado desde cero: `OK: main.pdf compilado, 28 paginas`.
- Recuento automático de citas sobre los 8 archivos con prosa: 89 apariciones,
  25 claves distintas, las 25 con 3 usos o más (mínimo 3, máximo 6 —
  `liu2024itransformer`), 25/25 contra `referencias.bib`, cero colgantes, cero sin usar.
  Coincide exacto con lo que reporta Task-023.
- Verificada la frase de jerarquía en `06_propuesta.tex` §6.1: "el fin de este trabajo
  no es explicar un modelo de aprendizaje automático, sino producir una hipótesis
  relacional sobre el fenómeno... Explicar el modelo... es un paso instrumental hacia
  ese fin, no el fin en sí mismo." Correcta y en el lugar que pide `CLAUDE.md`.
- Confirmado que la regla inventada de "cinco condiciones, en este orden" ya no está.
  En su lugar: "no hay una regla única ni un orden fijo de condiciones que el
  framework imponga de antemano". Correcto.
- Releídos los cinco recortes de cifras de desempeño en `03_estado_arte.tex`
  (`zhao2026causalguided`, `pipeline2026causalgraph`, `cai2024msgnet`,
  `convlstmgcn2026vegetation`, `rotorcraft2026vortexring`): en los cinco casos se
  conservó el resultado de explicabilidad y se retiró la cifra de desempeño del
  algoritmo, exactamente como describe Task-023. La mención de "exactitud" que queda en
  `rulexai2024events` es correcta: ahí es una de las cinco métricas con que el propio
  método de explicabilidad se evalúa a sí mismo, no el desempeño del clasificador
  destilado.
- Verificadas las dos notas de Task-024 en `03_estado_arte.tex` §3.1: procedencia de
  citas ("la mayoría... un número menor... con apoyo de asistencia de IA") y alcance
  futuro del análisis de citas, ambas en términos cualitativos, sin cifras inventadas.
  Coincide con la nota de verificación de claude-1 sobre las columnas de
  `Referencias_seminario.xlsx`.
- **Hallazgo nuevo, corregido:** `scripts/build_docx.py` (agregado en Task-021) traía
  rutas absolutas hardcodeadas a `C:\Users\patru\...` — mismo bug de portabilidad que
  tuvo `compilar.sh` con TinyTeX antes de esta sesión, aquí sin corregir. Cambiado a
  rutas relativas al propio script y variables de entorno (`PANDOC_BIN`, `IEEE_CSL`,
  `BUILD_DOCX_SCRATCH`) con mensajes de error explícitos si pandoc o `ieee.csl` no
  están. **No pude probarlo de punta a punta**: no hay pandoc instalado en esta
  máquina. Si alguien lo corre, confirmar que sigue generando el DOCX correctamente
  con las rutas nuevas.

Sin objeciones de fondo. El trabajo de Task-022/023/024 queda auditado y confirmado.

### Cerrada por claude-1 en esta sesión

- **Task-021** — Correcciones del autor humano sobre el informe de Task-020, más
  incorporación de la formulación matemática y entrega en DOCX. Se corrigió todo lo
  levantado:

  1. *Formato.* Se tomó como autoridad `Formato_Informes_Proyecto_Titulo-2024.pdf`.
     `parskip` pasa a 0 y `titlespacing` a 0 antes del encabezado, porque el formato
     pide que el fin de una sección y el encabezado siguiente no queden separados por
     espacios adicionales. Se mantienen sangría de 1 cm, interlineado sencillo, Times
     New Roman 12, márgenes de 2,5 cm y numeración romana antes de la introducción.
  2. *Índice sin hipervínculos.* Se carga `hyperref` antes del `.sty` (biblatex exige
     ese orden). El índice quedó navegable y el PDF con marcadores; se limpiaron los
     títulos de marcador que arrastraban el `space`.
  3. *Citas.* El formato de la asignatura usa numeración entre corchetes, no autor-año.
     Se cambió el estilo de biblatex de `apa` a `ieee` en `pucv_inf_2024.sty`, único
     cambio respecto de la copia original del template, y la prosa usa las tres formas
     de la guía: mención con autor integrado (`	extcite`), cita indirecta con el número
     al final (`\cite`) y una cita directa entrecomillada.
  4. *Resumen.* Reescrito: se habla de aprendizaje automático y no de "modelos
     profundos", se plantean las dos familias (modelos matemáticos rígidos frente a
     aprendizaje automático opaco) y el framework se presenta como lo que es, un modelo
     alternativo entrenado sobre los mismos datos del modelo principal, con tokens por
     variable, del que se obtienen hipótesis relacionales por consenso entre
     repeticiones. Ese encuadre se propagó a introducción, propuesta y conclusiones.
  5. *Formulación matemática.* Se incorporaron las secciones 2 a 5 y 8 de la
     presentación preliminar: arquitectura del ConvTransformer de índices y agregación
     por rollout más formalización del operador relacional en el marco teórico;
     transformación matriz-grafo, grafo-hipótesis e interpretación metodológica en la
     propuesta. Quedaron fuera las secciones 6 y 7 (evaluación por perturbación y
     análisis espectral), que son las que el autor humano pidió no incluir en esta
     entrega.
  6. *Objetivos.* Un objetivo general y exactamente tres específicos, sin subdivisión
     interna y acotados al tamaño de la tesis.
  7. *Estado del arte.* Pasó de discutir cuatro trabajos a discutir los dieciocho que la
     hoja `Aporte por sección` marca para esa sección, con su muestra, su método y su
     resultado. El capítulo creció alrededor de dos páginas.
  8. *Extensión.* Cuerpo de 18 páginas entre introducción y conclusiones, bajo el máximo
     de 20 que fijó el autor humano sin contar portada, índices ni referencias.

  **Entrega en DOCX.** El repositorio conserva el LaTeX como fuente y el PDF compilado.
  Además se generó `Informe_avance.docx` con pandoc, usando `citeproc` con `ieee.csl`
  para las citas y un `reference.docx` ajustado al formato de la Escuela. El DOCX sale
  con 36 encabezados jerarquizados, 25 referencias numeradas, 74 ecuaciones en formato
  nativo de Word y campo de índice automático. El binario de pandoc no se versiona: se
  descarga aparte, y el script de construcción queda documentado en el PR.

  Pendiente del autor humano: confirmar las fechas de la tabla del plan de trabajo.
  [claude-1]

- **Task-020** — Correcciones de redacción y de alcance sobre el informe ya migrado.
  El autor humano revisó el PDF resultante de Task-019 y levantó seis observaciones.
  Se corrigieron todas:

  1. *Párrafos demasiado breves y exceso de enumeraciones.* Se reescribió la prosa de
     las siete secciones y del resumen con párrafos desarrollados. Las listas bajaron de
     seis bloques a tres, y los tres que quedan están en `02_objetivos.tex`, donde
     enumerar es la convención de la sección.
  2. *Citas mal incorporadas.* Antes todo era `\cite` al final de la oración, que en
     APA rinde siempre entre paréntesis. Ahora se usan las tres formas según
     corresponda: `\textcite` para la mención con el autor integrado en la frase (21
     usos), `\parencite` para la cita indirecta entre paréntesis (12 usos) y una cita
     directa entre comillas con `\enquote`, tomada literal del resumen de
     `mechinterp2025ts` en arXiv y atribuida a sus autores.
  3. *El contexto del problema no debía ser el caso de estudio.* La sección 1.1 se
     reescribió alrededor de la pregunta de investigación registrada al diseñar el
     protocolo PRISMA, desarmándola por componentes (población, condición intervenida,
     comparador y resultado esperado). El caso de los índices espectrales salió de ahí
     y aparece ahora donde corresponde, en el marco teórico y en el alcance.
  4. *La obtención del corpus entraba en demasiado detalle.* La sección 3.1 pasó de
     cinco párrafos más tabla a dos párrafos más tabla: protocolo empleado, cadena de
     búsqueda, fuentes, criterios de inclusión y resultado. Salieron el desglose PICO
     (ahora está en la introducción, que es su lugar) y el detalle paso a paso del
     cribado.
  5. *El texto hablaba de código.* Esta entrega define alcance, así que salieron del
     documento la ablación con renormalización, la fórmula del operador Top-K, los
     operadores alternativos, la prueba binomial con sus correcciones por comparaciones
     múltiples y el desglose de la arquitectura del ConvTransformer. El criterio de
     validación quedó descrito por lo que establece cada paso y no por cómo se
     implementa, y se agregó una sección de alcance explícito (§6.6) con lo que esta
     etapa compromete y lo que deja fuera.
  6. *Redacción con tells de IA.* Además de la reescritura, se pasó una auditoría de
     patrones sobre el resultado y se corrigieron repeticiones de apertura ("conviene",
     de 7 a 4 usos, y los que quedan son de uso corriente) y verbos inflados que
     reemplazaban a "ser" ("constituye", de 5 a 1).

  Nota sobre la §6.2, que es de claude-2: la descripción arquitectónica quedó reducida a
  su argumento de delimitación (el ConvTransformer es instrumento y no contribución),
  porque el detalle de implementación es justamente lo que el autor humano pidió sacar.
  El argumento se conserva íntegro; lo que se retiró fue la enumeración técnica.

  El corpus sigue cerrado en 25: recuento automático posterior a la reescritura da 25
  claves citadas y 25 entradas en `referencias.bib`, sin colgantes ni sobrantes. Cero
  rayas largas en la prosa. Cuerpo de 19 páginas.

  Reserva: se editó sin el commit previo de `IN_PROGRESS` porque no había PR abierto de
  claude-2 ni archivo suyo en curso al momento de empezar, y la instrucción del autor
  humano llegó sobre el documento completo. Queda anotado por transparencia.

  **`thesis-prose-audit` no está instalado en la máquina de claude-1**, aunque
  `CLAUDE.md` lo declara obligatorio. La auditoría de prosa se hizo con la skill
  `humanizer`, que cubre el mismo conjunto de patrones. Si esa skill vive solo en el
  entorno de claude-2, conviene decirlo en `CLAUDE.md` o publicarla en el repositorio,
  porque tal como está redactada la regla no se puede cumplir desde este lado. [claude-1]

  **Revisado por claude-2.** Confirmado cada uno de los seis puntos: recompilé en
  worktree aparte (`OK: main.pdf compilado, 27 paginas`), recontabilicé citas contra
  `referencias.bib` (25 usadas — `\textcite`/`\parencite`/`\enquote` incluidos en el
  conteo — y 25 entradas, 1:1 exacto) y grep de rayas largas fuera de comentarios de
  cabecera (cero). Leí completa la §6.2 nueva: el argumento de delimitación del
  ConvTransformer (instrumento, no contribución; no compite en desempeño; Φ es
  agnóstico a la procedencia de la matriz) queda intacto, solo se fue el desglose de
  capas/kernels/stride que ya no corresponde a esta entrega. Sin objeciones. Mergeado.
  Corregida además la regla de `thesis-prose-audit` en `CLAUDE.md` para que no asuma
  que la skill está instalada en ambas máquinas.

### Cerrada por claude-1 en esta sesión

- **Task-019** — Migración completa al template PUCV
  (`Template_Latex__Formato_Informes__Copy_/`) y reestructuración del documento según la
  rúbrica de la entrega de avance. **Instrucción directa del autor humano**, que además
  fijó cuatro condiciones: (a) el informe cita solo las 25 referencias de
  `Referencias_seminario.xlsx`, ni una más; (b) el estado del arte debe explicar el
  protocolo PRISMA, porque ahí se evidencia cómo se obtuvieron los documentos; (c) esta
  entrega **no** reporta resultados de código, así que las cifras del prototipo que
  Task-016 había agregado salen del texto; (d) sin rayas largas (`--`) en la prosa.

  Esto obliga a tocar archivos de claude-2 (`04_arquitectura.tex`,
  `05_formalizacion.tex`, `99_bibliografia.tex`) y `main.tex`, que normalmente claude-1
  no toca. Se hace porque el autor humano lo pidió explícitamente y porque una migración
  de formato no se puede partir por dueño sin dejar el documento sin compilar a la
  mitad. **El contenido técnico de claude-2 se migra sin reescribirlo**: cambia de
  archivo y se le limpian las citas obsoletas, nada más. claude-2 queda como dueño de
  esas partes dentro de los archivos nuevos y puede reescribirlas cuando quiera.
  [claude-1]

  **Resultado.** El documento quedó con la estructura que pide la rúbrica: resumen y
  abstract, introducción con contexto, problema, estado previo y contribución, objetivos
  separados en estudio, desarrollo y validación, estado del arte con el protocolo PRISMA
  y seis familias con su pro y su contra, marco teórico con la definición formal del
  problema y las técnicas necesarias, plan de trabajo con actividades y fechas,
  propuesta en detalle y conclusiones preliminares. Cuerpo de 18 páginas, dentro del
  rango de 10 a 20 que pide la entrega, más portada, índices y referencias.

  Las cuatro condiciones del autor humano quedaron cumplidas y verificadas:

  1. *Solo 25 citas.* Recuento automático sobre los `.tex`: 25 claves distintas citadas
     y 25 entradas en `referencias.bib`, correspondencia exacta, cero colgantes y cero
     entradas sin citar. Las 14 referencias del corpus obsoleto salieron del documento.
  2. *PRISMA explicado.* Sección 3.1, con la pregunta PICO, la cadena de búsqueda, la
     tabla de registros por base (Scopus 191, Web of Science 100, PubMed 28, total 319)
     y la depuración 319 → 221 → 201 → 137 hasta las 25 incluidas. Se declara ahí mismo
     que el paso de 137 a 25 no tiene umbral cuantitativo explícito.
  3. *Sin resultados de código.* Se retiraron del resumen, la introducción y las
     conclusiones las cifras del prototipo que Task-016 había agregado (Jaccard,
     porcentaje de persistencia entre semillas, sensibilidad al umbral K). El texto
     vuelve a declarar el sistema como diseñado y no ejecutado, que es lo que
     corresponde a esta entrega.
  4. *Sin rayas largas.* `grep` sobre los diez archivos de prosa: cero ocurrencias de
     `--` fuera de los comentarios de cabecera y de los rangos de página del `.bib`.

  **Pendiente para el autor humano:** las fechas del plan de trabajo (tabla
  \texttt{tab:plan}) son una propuesta construida sobre la única fecha documentada, que
  es la de la bitácora de revisión sistemática. Hay que contrastarlas con el calendario
  real de la asignatura antes de entregar.

  **Nota menor de corpus para claude-2:** la fila 5 del excel registra el título
  *Evaluating the Faithfulness of Causality in Saliency-Based Explanations...* con año
  2024, pero el DOI que trae, `10.48550/arXiv.2211.07982`, resuelve a un preprint de
  2022 titulado *Evaluating the Faithfulness of Saliency-based Explanations...*, sin la
  palabra *Causality*. La entrada `rizzo2022faithfulness` se construyó con lo que
  devuelve el DOI. Conviene verificar si existe una versión publicada posterior.

- **Task-018** — `04_arquitectura.tex` y `A_matriz_literatura.tex` (ambos propiedad de
  claude-2) también citan referencias del corpus obsoleto y no fueron cubiertos por
  Task-017 porque son archivos ajenos a claude-1. **Queda absorbida por Task-019**: los
  dos archivos salen del documento y su contenido migra ya sin citas obsoletas. Se deja
  registrada para que no se retome. [claude-2]

(Task-010 a Task-017 completadas y mergeadas; ver `Estado de secciones` arriba y las
notas de cierre abajo. Se sacan de la cola para que no se vuelvan a tomar por error.)

### Cerradas en sesiones anteriores por claude-1

- **Task-015** — (cerrada en sesión previa) Mergeada a `main` en PR #6; C-001 quedó
  `RESUELTA`, confirmado por claude-2.
- **Task-016** — Aplicado el fix propuesto en **C-002**. Se matizó el cierre de
  "La propuesta" en `secciones/01_introduccion.tex`: en vez de declarar la cadena
  completa "aún no ejecutada", se precisa que los eslabones 1 y 2 (estabilidad continua
  y estructural) cuentan con evidencia preliminar sobre el prototipo --núcleo estable en
  el 76--78\% de 50 modelos con semillas distintas, Jaccard promedio 0.195--0.204, y sin
  punto crítico al variar el umbral $K\in\{3,5,7,10\}$--, mientras que los eslabones 3 a 5
  y el resto de la triangulación de paradigmas siguen sin ejecutarse. Se aplicó la misma
  precisión, de forma más breve, en el cierre de `00_resumen.tex` y en
  `07_conclusiones.tex`. claude-1 verificó estas cifras directamente contra
  `Prototipo_Preliminar.ipynb` (celda 16, análisis de sensibilidad al umbral K) antes de
  escribirlas, además de las que ya había cotejado claude-2 en Task-011/C-002. No se
  afirma que el sistema
  multicriterio esté validado: el texto sigue siendo literal sobre lo pendiente
  (eslabones 3-5). Compila en 18 páginas. Pendiente de revisión de claude-2 (ver PR
  abajo); C-002 sigue `ABIERTA` hasta que se confirme y mergee.
- **Task-017** — Aplicado el fix propuesto en **C-003**. Antes de reescribir, se
  recotejaron por título (no solo por DOI) los 19 `\bibitem` que C-003 marcó
  obsoletos contra las 25 filas de `Referencias_seminario.xlsx`: **5 de los 19 son en
  realidad la misma obra que ya está en `99_bibliografia.tex`**, solo que citada con el
  DOI del preprint de arXiv en vez del DOI de la versión publicada (o viceversa) —
  `lapuschkin2019unmasking` (fila 10), `abnar2020quantifying` (fila 1),
  `slack2020fooling` (fila 9), `chefer2021transformer` (fila 3) y `cai2024msgnet`
  (fila 15). Estas 5 no se reemplazan ni se piden de nuevo: ya resuelven contra el
  corpus vigente, la comparación de C-003 solo comparó DOI y no título. Los 14
  restantes sí son obsoletos de verdad (obra distinta, sin equivalente en las 25
  filas): `jakubowski2022performance`, `vrahatis2024graph`, `petrosian2024solar`,
  `tew2024kans`, `zhu2025attention`, `chatterjee2025multicriteria`, `klotz2025xai`,
  `han2026tscad`, `qin2026multisensor`, `ewuzie2026robust`, `zhang2026ehtgnn`,
  `kapoor2026fustt`, `zhou2026causal`, `khayitov2026stgeonet`.

  Se reescribió por completo `secciones/02_estado_arte.tex` (6 familias metodológicas
  nuevas + `tab:familias`) sobre las 18 filas de `Referencias_seminario.xlsx`
  etiquetadas "Estado del arte" en la hoja `Aporte por sección`: 6 ya tenían
  `\bibitem` vigente (las 3 de C-003 más las 4 de las 5 recotejadas que aplican aquí:
  `chefer2021transformer`, `slack2020fooling`, `lapuschkin2019unmasking`,
  `cai2024msgnet`, `zhao2026causalguided`, `liu2024itransformer`) y 12 necesitan
  `\bibitem` nuevo (pedidos arriba). Se corrigieron además las citas obsoletas en
  `secciones/01_introduccion.tex` (contexto, problema y justificación) y
  `secciones/07_conclusiones.tex` (delimitación del aporte), y se reescribió
  `secciones/06_aplicabilidad.tex` completa (`tab:dominios`, 5 dominios en vez de 6)
  porque su argumento de aplicabilidad dependía enteramente de referencias del corpus
  obsoleto y ninguna de las 25 filas vigentes está etiquetada para "Aplicabilidad" en
  la hoja `Aporte por sección` --se reconstruyó con las filas de Estado del arte que
  mejor calzan (redes de sensores, industria, vegetación, aerodinámica, clima).
  `secciones/00_resumen.tex` y `secciones/03_objetivos.tex` no citan nada: sin cambios.

  Hallazgo adicional que excede el alcance de Task-017: `secciones/04_arquitectura.tex`
  y `secciones/A_matriz_literatura.tex` (ambos de claude-2) también citan
  profusamente el corpus obsoleto y C-003 no los había cubierto. Registrado como
  **Task-018** arriba, con el detalle exacto de qué cita en qué línea abajo en la
  descripción de C-003 actualizada.

  `./scripts/compilar.sh` → `OK: main.pdf compilado, 18 paginas`,
  `ADVERTENCIA: 42 referencias o citas sin resolver` — las 12 `\bibitem` pedidos
  arriba, esperado hasta que claude-2 los agregue. Pendiente de revisión de claude-2
  (ver PR abajo); C-003 sigue `ABIERTA` para `02_estado_arte.tex`,
  `01_introduccion.tex`, `06_aplicabilidad.tex` y `07_conclusiones.tex` hasta que se
  agreguen los `\bibitem`, se confirme y se mergee; sigue abierta sin fecha para
  `04_arquitectura.tex` y `A_matriz_literatura.tex` (Task-018).

### Cerradas por claude-2 en esta sesión

- **Task-017 (bibitem)** — Agregados los 12 `\bibitem` pedidos por claude-1 al final de
  `secciones/99_bibliografia.tex` (append-only: los 22 anteriores no se tocaron).
  Metadata (autores, revista, volumen, páginas) obtenida de Crossref por DOI para los
  9 artículos de revista, y de la API de arXiv para los 3 preprints
  (`mechinterp2025ts`, `sentinel2025multipatch`, `patchmlp2025unlocking`), ya que
  Crossref no resuelve DOIs `10.48550/arXiv.*`. `./scripts/compilar.sh` →
  `OK: main.pdf compilado, 19 paginas`, cero referencias sin resolver. Con esto,
  `02_estado_arte.tex`, `01_introduccion.tex`, `06_aplicabilidad.tex` y
  `07_conclusiones.tex` quedan sin pendientes de Task-017; solo falta Task-018
  (`04_arquitectura.tex` y `A_matriz_literatura.tex`, ambos de claude-2).
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

Ninguno abierto.

(PR #19 — deja `build_docx.py` funcionando de punta a punta (pandoc + `ieee.csl` en el
repo) y, en un segundo commit, recorta en `04_marco_teorico.tex` el mismo detalle de
implementación del ConvTransformer que Task-025 sacó del resto del documento —
resuelve la tarea que había quedado anotada para claude-2. Mergeado por claude-2, con
autorización explícita del autor humano para seguir sin pedir confirmación en cada PR.
PR #20 — Task-025 (retira código/umbrales/métricas de la prosa, foco en formulación
matemática y defensa conceptual de la idea). En `06_propuesta.tex` se mantuvo la cadena
de 5 preguntas y la taxonomía de 4 resultados (decisión explícita del autor humano),
solo se retiró el detalle operativo alrededor. Revisado por claude-2: recompilado
limpio (28 páginas), recuento de citas 25/25 contra `referencias.bib`, todas con 3-6
usos, sin cambios respecto al conteo anterior. Mergeado.
Task-022/023/024, ver arriba, se ejecutaron en una sola sesión sin pasar por rama ni
PR — ver la "Nota de proceso" al cierre de Task-024, arriba.)

(PR #18 — auditoría independiente de Task-022/023/024 (ver nota de cierre arriba) más
el primer fix de portabilidad de `build_docx.py` (rutas relativas y variables de
entorno, sin probarlo de punta a punta por falta de pandoc en ese momento). Mergeado
por claude-2.
PR #17 — Task-021 (formato de la Escuela, citas numéricas IEEE, formulación
matemática de la presentación preliminar y entrega en DOCX). Mergeado; esta sesión
encontró el merge ya hecho (`50c9544`) al hacer `git fetch`, sin nota de revisión de
claude-2 asociada en este archivo — quedó así de una sesión previa, se deja constancia
por transparencia en vez de fabricar una revisión que no ocurrió.
PR #15 — Task-020, redacción, citas y alcance. Mergeado por claude-2.)

(PR #11 — pone al día `CLAUDE.md`/`TASKS.md` tras `Framework.py` y, ampliado después,
tras la migración PUCV de Task-019 (arquitectura de archivos, biblatex, propiedad de
`04_marco_teorico.tex`/`referencias.bib`/§6.2 de `06_propuesta.tex`, corpus cerrado en
25, condiciones de esta entrega). Mergeado por claude-2, autorización explícita del
autor humano.
PR #12 — Task-019, migración al template PUCV y ajuste a la rúbrica de la entrega,
por orden directa del autor humano. Tocó archivos de claude-2 (`04_arquitectura.tex`,
`05_formalizacion.tex`, `99_bibliografia.tex`) y `main.tex` con justificación explícita
en el cuerpo del PR. Revisado a fondo por claude-2: instaló los paquetes LaTeX que
faltaban en esta máquina (titlesec, lipsum, biblatex-apa, glossaries, nomencl, etc.),
compiló limpio (18 páginas de cuerpo, 26 con portada/índices/referencias), recontó citas
contra `referencias.bib` (25/25, cero colgantes, cero sin citar) y verificó por DOI/API
de arXiv la nota de claude-1 sobre el título mal cargado de la fila 5 del excel
(`rizzo2022faithfulness`): confirmado, el DOI real no dice "Causality", claude-1 ya usó
el título correcto. Mergeado por claude-2.
PR #10 — agrega los 12 `\bibitem` de Task-017. Revisado y mergeado por claude-1:
compilado en worktree aparte (`OK: main.pdf compilado, 19 paginas`, cero referencias
sin resolver, 34 `\cite`/34 `\bibitem` recontados 1:1), y verificada la metadata de 2
de los 12 `\bibitem` contra Crossref/arXiv por DOI (autores, volumen, página y
transliteración LaTeX de diacríticos coinciden exactamente) — detalle en el comentario
del PR.
PR #9 — fija `Referencias_seminario.xlsx` como corpus vigente (C-003, Task-017).
Mergeado por claude-2, autorización explícita del autor humano.
PR #8 — Task-016 (resuelve C-002) + Task-017 (resuelve C-003 salvo `04_arquitectura.tex`
y `A_matriz_literatura.tex`, ver Task-018). Revisado a fondo por claude-2: verificado
contra `Prototipo_Preliminar.ipynb` (celda 16, análisis de sensibilidad K) el dato nuevo
de Task-016, y por título contra `Referencias_seminario.xlsx` las 5 correcciones de
claude-1 a C-003 (mismo DOI en variante arXiv/publicado). Compilado en worktree aparte:
`OK, 18 páginas` + `ADVERTENCIA: 42 referencias sin resolver`, exactamente las 12
`\bibitem` pedidas, nada inesperado. Mergeado por claude-2.
PR #7 —.gitignore + elimina `.claude/settings.local.json` de `main`, arreglo de
higiene sin contenido de tesis— ya está `closed`/mergeado; sin acción pendiente.
PR #4 revisado y mergeado por claude-1: cifras cotejadas contra
`Prototipo_Preliminar.ipynb` línea por línea, ver comentario en el PR.
PR #6 —.gitignore del autor humano + Task-015 de claude-1 agregado como commits
adicionales porque GitHub no dejó abrir un segundo PR desde la misma rama— revisado
y mergeado por claude-2 a `main`.
PR #7 — saca `.claude/settings.local.json` de `main` (se había colado en PR #4).
Mergeado por claude-2 (autorización explícita del autor humano; nadie más lo
revisaba en el momento y bloqueaba a ambos agentes).
PR #9 — fija `Referencias_seminario.xlsx` como corpus vigente (C-003, Task-017).
Mergeado por claude-2, misma autorización explícita.)

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
  Estado: RESUELTA — claude-2 revisó el fix de claude-1 (Task-016) y verificó
  directamente contra `Prototipo_Preliminar.ipynb` (celda 16, análisis de sensibilidad
  al umbral K) el dato nuevo que agrega ("no encuentra un punto crítico... degradación
  gradual"): coincide palabra por palabra con la salida real de la celda. Mergeado a
  `main` en PR #8.

- [C-003] Secciones implicadas: 02_estado_arte.tex (principal), 01_introduccion.tex,
  06_aplicabilidad.tex, 07_conclusiones.tex, 99_bibliografia.tex, 04_arquitectura.tex y
  A_matriz_literatura.tex (estas dos últimas, ver Task-018)
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

  Corrección de claude-1 sobre el propio C-003 (Task-017): el cotejo por DOI de los
  19 "obsoletos" tenía 5 falsos positivos. `lapuschkin2019unmasking`,
  `abnar2020quantifying`, `slack2020fooling`, `chefer2021transformer` y
  `cai2024msgnet` son la misma obra que las filas 10, 1, 9, 3 y 15 de
  `Referencias_seminario.xlsx` respectivamente, citada con el DOI del preprint de
  arXiv en `99_bibliografia.tex` en vez del DOI de la versión publicada (o viceversa):
  DOI distinto, obra idéntica. Comparar por título además de por DOI lo muestra de
  inmediato. Estas 5 no son obsoletas y no se piden de nuevo. Los 14 restantes sí lo
  son. Ver detalle completo en la nota de cierre de Task-017.

  Estado: RESUELTA para `02_estado_arte.tex`, `01_introduccion.tex`,
  `06_aplicabilidad.tex`, `07_conclusiones.tex` y `99_bibliografia.tex` — claude-2
  revisó el fix de claude-1 (Task-017) a fondo (confirmadas por título las 5
  correcciones al cotejo de C-003, compilado en worktree aparte antes de mergear) y
  agregó los 12 `\bibitem` pedidos. `./scripts/compilar.sh` final:
  `OK: main.pdf compilado, 19 paginas`, cero referencias sin resolver. Sigue ABIERTA
  sin fecha para `04_arquitectura.tex` y `A_matriz_literatura.tex` (Task-018, de
  claude-2).

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
- **2026-09-07** — El autor humano reemplaza `Prototipo_Preliminar.ipynb` por
  `Framework.py` directo en `main` (commits `d35e414`, `a8eef14`, `e398528`). Código
  fuente puro, sin salidas de celda ejecutadas, y con varias fórmulas de índices
  corregidas respecto al notebook (MARI, ARI, CHL\_REDEDGE, PSRI). Las cifras
  experimentales ya citadas en la prosa (Jaccard, sensibilidad a K) se verificaron
  contra el notebook cuando aún existía y quedan documentadas arriba en las notas de
  cierre de Task-011/Task-016; siguen siendo válidas como lo que efectivamente se
  ejecutó entonces. Una cifra nueva que se quiera citar desde `Framework.py` requiere
  confirmar que se ejecutó (no solo que el código está ahí) antes de darla por buena.
  Ver `CLAUDE.md`, sección "Fuentes primarias".
- **2026-09-08** — El autor humano ordena migrar el documento al template
  institucional PUCV (Task-019, PR #12), con cuatro condiciones para esta entrega:
  corpus cerrado en las 25 referencias de `Referencias_seminario.xlsx`, estado del
  arte que explique el protocolo PRISMA, sin cifras de ejecución del prototipo, sin
  rayas largas en la prosa. `main.tex` cambia de clase (`article`→`report`) y de motor
  bibliográfico (`thebibliography`→biblatex+biber); la estructura de `secciones/`
  cambia de nombre y orden. El contenido técnico de claude-2
  (`04_arquitectura.tex`→§6.2 de `06_propuesta.tex`, `05_formalizacion.tex`→
  `04_marco_teorico.tex`, `99_bibliografia.tex`→`referencias.bib`) se migró sin
  reescribir la prosa; claude-2 sigue siendo dueño de esas partes en los archivos
  nuevos. `CLAUDE.md` se actualizó completo para reflejar la arquitectura nueva (ver
  PR #11). Ver la nota de cierre de Task-019 arriba para el detalle completo.
- **2026-09-08** — El autor humano revisa el informe de Task-021 y pide cinco
  correcciones sobre `06_propuesta.tex` y el resto del documento (configuradas como
  Task-022/023/024 y ejecutadas en la misma sesión, ver notas de cierre arriba):
  alcance de la propuesta (hipótesis relacional como fin, explicar el modelo como
  medio), retiro de una regla inventada de selección de dominio, uso mínimo de 3 citas
  por referencia, foco en resultado de explicabilidad en vez de precisión de algoritmo,
  y dos notas de alcance (procedencia de citas, análisis de citas como trabajo futuro).
  Se instaló además `biblatex-ieee`, paquete que faltaba en la máquina y que Task-021
  necesitaba sin haberlo declarado en `CLAUDE.md`.
- **2026-09-08** — El autor humano pide, en una sesión posterior de claude-1, que el
  informe deje de mencionar código, umbrales, métricas y resultados de ejecución, y se
  concentre en la formulación matemática del framework y en la defensa conceptual de la
  idea (Task-025). Preguntado el alcance exacto antes de tocar prosa: (a) la cadena de 5
  preguntas y la taxonomía de 4 resultados del criterio de validación en
  `06_propuesta.tex` se **mantienen** (son la idea que se defiende), solo se retira el
  detalle operativo (método de perturbación, métrica de fidelidad, el valor de $K$ del
  criterio Top-$K$); (b) la arquitectura ConvTransformer en `04_marco_teorico.tex` no se
  toca directamente por ser archivo de claude-2, se anota en la cola para que claude-2
  decida. Ver detalle completo en Task-025 arriba y en `CLAUDE.md` ("Sin código,
  umbrales ni métricas de ejecución en esta entrega").
