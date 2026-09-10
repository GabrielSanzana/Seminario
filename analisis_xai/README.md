# Análisis XAI del ConvTransformer espectral: bitácora cronológica

Registro de lo que se hizo entre el 8 de septiembre de 2026 (16:00) y el 9 de
septiembre de 2026 (21:20) sobre los 50 modelos ya entrenados del caso de
estudio: un ConvTransformer entrenado en reconstrucción enmascarada de 12
índices espectrales Sentinel-2 sobre un viñedo en Pencahue, Maule.

El documento está ordenado por lo que se fue descubriendo, no por la estructura
final del código. Incluye los caminos que se abandonaron y por qué, porque en
varios casos el motivo del abandono es el resultado.

## Índice

1. [Punto de partida](#1-punto-de-partida)
2. [Estructura del grafo y reproducibilidad](#2-estructura-del-grafo-y-reproducibilidad)
3. [Verdad externa: el cultivo](#3-verdad-externa-el-cultivo)
4. [La descomposición por aridad](#4-la-descomposición-por-aridad)
5. [Intervenir en vez de leer](#5-intervenir-en-vez-de-leer)
6. [Cuatro intentos de arreglar la lectura](#6-cuatro-intentos-de-arreglar-la-lectura)
7. [Invariancia causal con la fenología como entorno](#7-invariancia-causal-con-la-fenología-como-entorno)
8. [Separar los regímenes](#8-separar-los-regímenes)
9. [Disociación por fase y el control de exposición](#9-disociación-por-fase-y-el-control-de-exposición)
10. [Conclusión](#10-conclusión)
11. [Errores cometidos y cómo se detectaron](#11-errores-cometidos-y-cómo-se-detectaron)
12. [Mapa de archivos](#12-mapa-de-archivos)

---

## 1. Punto de partida

El pipeline entrena 50 ConvTransformers con semillas distintas y guarda, por
cada uno, la matriz de atención (rollout, por capa, por cabeza) y una matriz de
dependencia obtenida por ablación de la entrada. La pregunta original de la
tesis era si de la matriz de atención se pueden extraer hipótesis relacionales
válidas entre los 12 índices.

`certificar_todo.py` es el módulo base del que dependen todos los demás:
carga las fuentes, define los 66 pares, la correlación parcial `|pc|` como
verdad de referencia, la correlación marginal `|r|` como listón a superar, las
familias espectrales y las bandas Sentinel-2 de cada índice.

Dos correcciones tempranas al material de partida:

- Las fórmulas del PDF de referencia del framework estaban desactualizadas.
  Se verificaron una a una contra `compute_12_indices()` del pipeline y contra
  la documentación de sentinel-hub. **NDMI usa B8A, no B08.** Eso cambia la
  clasificación de resolución de ese índice y debilitó un contraste que se daba
  por bueno: el test exacto de Fisher sobre pares que comparten bandas pasó de
  `odds` 3.84, `p` = 0.0287 a `odds` 3.03, `p` = 0.0627. Dejó de ser
  significativo al 5%.

- La modularidad de Newman-Girvan se estaba interpretando mal: el sentido lo
  decide el signo de Q contra cero, no la comparación contra la media del nulo.
  El nulo sirve para el `p`. Corregido, la modularidad contra familias
  espectrales resultó **negativa** en 6 de 8 fuentes, o sea que el grafo conecta
  entre familias y no dentro.

---

## 2. Estructura del grafo y reproducibilidad

Módulos: `analisis_grafo.py`, `analisis_espectral.py`, `analisis_profundo.py`

Primera línea de ataque: caracterizar el grafo que produce cada fuente y medir
si las 50 semillas coinciden.

**Lo que salió bien.** La reproducibilidad entre semillas es alta en las 8
fuentes (ARI de +0.38 a +0.91 sobre comunidades detectadas de forma
independiente). El mejor resultado del bloque: las comunidades de la ablación
en k = 3 dan ARI +0.910 entre semillas, con la partición

```
{NDVI, CHL_REDEDGE, PSRI, KNDVI}   {EVI, EVI2, NDMI, NDII, SAVI}   {MARI, ARI, NDWI}
```

y sólo 0.158 de ARI contra las familias espectrales, así que no es la partición
que la máscara de entrenamiento le inyecta al modelo.

**Tres métricas degeneradas que hubo que arreglar.** Las tres daban el mismo
número en las 8 fuentes, lo que era la pista:

- El primer número de Betti salía 55 siempre. La fórmula de Euler
  `b1 = E − V + C` sobre un grafo denso de 12 nodos y 66 aristas da 55 por
  aritmética. Se corrigió calculando `b1` sobre el complejo de clíques a lo
  largo de la filtración, descontando triángulos rellenos.
- La entropía de von Neumann daba 2.398 = log(11) en todas. El laplaciano
  normalizado de un grafo denso tiene casi todos los autovalores en 1. Se
  cambió al laplaciano combinatorio.
- Lo mismo para la escala de difusión del heat kernel.

**Métodos nuevos que se añadieron aquí.** Cadenas de Markov (estacionaria,
corriente neta, tiempos de primera pasada, irreversibilidad), homología
persistente, curvatura de Ollivier-Ricci por transporte óptimo, resolvente e
índice de Estrada, y autovalores fuera del bulk contra un nulo calibrado
permutando las entradas de la propia matriz.

**Primera señal fuerte.** La curvatura de Ollivier-Ricci separa las fuentes:
la atención tiene curvatura casi constante (+0.518 a +0.532, sin ninguna arista
negativa), la ablación tiene 4 aristas de curvatura negativa, y la más negativa
es `CHL_REDEDGE-KNDVI` (−0.22). Esa misma arista aparece con el mejor SNR entre
semillas y en el análisis de difusión. Triangulación de tres métodos
independientes.

**Corrección importante de esta etapa.** La asimetría cruda `A[i,j] − A[j,i]`
estaba dominada en un 85% de su varianza por cuán sumidero es cada nodo, que es
consecuencia matemática de normalizar filas a suma 1. Se implementó
`_parte_de_par(S)`, que descompone la matriz antisimétrica en efecto aditivo de
nodo más parte de par limpia. Con esa corrección **se invirtió una conclusión
previa**: la ablación tiene dirección genuina (`rho_par` +0.186, `p` de signos
1e-07 sobre 43/50 semillas) y la atención no (`rho_par` −0.064, `p` = 0.64).
Sin corregir, el resultado decía lo contrario.

---

## 3. Verdad externa: el cultivo

Módulos: `analisis_fenologico.py`, `reinferir_por_fase.py`

Las conclusiones de la tesis afirmaban que para índices Sentinel-2 no existe
referencia externa contra la cual comparar. Resultó falso.

El catastro CIREN 2024 de viveros (`viveros_2024_h19.shp`, 3185 polígonos)
sitúa un vivero de **Vid Vinífera de 41.05 ha en Pencahue, a 0.00 km del centro
del recorte**. Las coordenadas del stack convertidas a UTM 19S dan 243117,
6073342, dentro de ese polígono. El shapefile se parseó a mano con `struct`
porque no había pyshp ni geopandas en el entorno.

Saber que es vid obliga a reinterpretar dos resultados anteriores:

- Se había atribuido a nubosidad invernal que 7 de las 10 escenas de mayor
  residuo cayeran entre mayo y septiembre. En vid, mayo a agosto es
  **dormancia**: la planta no tiene hojas y los índices de vegetación colapsan
  por biología. Las dos causas se confunden en el mismo periodo.
- El 39.8% de píxeles con NDVI > 0.3 y el 57.3% de suelo desnudo dejan de ser
  un reparto arbitrario: es la firma geométrica de un viñedo en hileras.

**Medición.** La estructura de dependencia sí cambia con la fenología: la
correlación de Spearman entre el `|pc|` de dormancia y el de crecimiento es
0.510, no 1.

**Experimento y resultado negativo.** `reinferir_por_fase.py` carga los 50
checkpoints y los re-infiere restringiendo la entrada a cada fase, sin
reentrenar. Si el modelo aprendió los regímenes, evaluado en una fase debería
producir la estructura de esa fase. Resultado: **2 de 5 fases aciertan la
diagonal**, cuando el azar da 1 de 5. El modelo colapsó los regímenes. La
explicación más simple es `SEQ_LENGTH = 1` sin codificación temporal: el modelo
nunca ve la fecha.

Dos bugs que costaron tiempo aquí y que conviene dejar escritos:
`img_size` tiene que ser `(TOKEN_PARCHE, TOKEN_PARCHE) = (13,13)`, no el tamaño
del frame, porque el modelo se entrenó sobre parches; y el scaler tiene que ser
el global ajustado sobre todo el tramo de train, no sobre la fase, o las
matrices entre fases quedan en escalas incomparables.

---

## 4. La descomposición por aridad

Módulo: `analisis_atencion.py`

Aquí cambia el enfoque. Hasta este punto todas las pruebas le preguntaban a la
atención lo mismo: qué par de índices se relaciona con cuál. Y en todas fallaba.
La lista de fracasos admitía una lectura más barata que "la atención es ruido":
que lleve señal reproducible pero **de aridad uno**.

Sobre las entradas fuera de la diagonal se hace la descomposición aditiva de dos
vías, que es la del Social Relations Model de Kenny y La Voie (1984) y el modelo
p1 de Holland y Leinhardt (1981), aplicada a matrices de atención:

```
A[i,j] = mu + r_i + c_j + R[i,j]
         |    |     |     |
         |    |     |     relacion de PAR      (aridad 2)
         |    |     cuanto RECIBE la columna   (aridad 1, efecto sumidero)
         |    cuanto EMITE la fila             (aridad 1)
         nivel global                          (aridad 0)
```

Los efectos salen por centrado iterativo por filas y columnas, que con la
diagonal ausente converge a mínimos cuadrados. Validado con controles
sintéticos: recomposición exacta a 6.7e-16, matriz puramente unaria da
`frac_par` = 0.000000, matriz puramente de par da 1.000000.

**Resultado principal.**

```
fuente                        fila  columna     PAR
atencion rollout              1.0%    74.7%   24.3%
atencion capa A1              0.8%    95.4%    3.8%
atencion A2*A1                0.1%    98.6%    1.3%
dependencia por ablacion      0.1%    15.3%   84.6%
```

Imagen especular. La atención es 75 a 99% efecto de nodo; la ablación es 85%
relación de par.

**Prueba constructiva.** Se fabrica el sustituto unario
`A_sur = mu + r_i + c_j`, con parte de par exactamente cero por construcción, y
se le pasan los mismos descriptores:

```
fuente                    rho rank  Jaccard  b1 real  b1 sur  curv real  curv sur
atencion rollout             0.773    0.669        0       0     +0.532    +0.544
dependencia por ablacion     0.292    0.221        2       0     +0.236    +0.521
```

Una matriz sin ninguna información de par reproduce el ranking de la atención,
su grafo Top-P, su `b1` = 0 y su curvatura. En la ablación falla en todo, así
que el control discrimina y la prueba vale.

Consecuencia: **cinco resultados anteriores quedan retirados** — `b1` = 0 como
hallazgo topológico, la curvatura constante, el grafo Top-P, y (verificado
aparte en `nmi_sustituto.py`) los acoplamientos NMI = 1.0 dentro de G1, donde el
sustituto produce 76 frente a los 50 reales con el 100% compartiendo destino.

**Qué es la parte unaria.** El vector de saliencia `c` correlaciona `rho` =
+0.853 (`p` = 0.0004) con la correlación marginal media del índice, +0.763 con
el tamaño de su familia y +0.741 con su redundancia. Ranking: SAVI, NDVI, EVI2,
EVI, KNDVI arriba; NDWI, MARI, PSRI, ARI abajo. **La atención asigna peso en
proporción a lo redundante que es un índice.** Con `familia_balanceada`, cuando
se oculta una familia entera, apoyarse en el superviviente más genérico es la
política óptima para MSE. El objetivo de entrenamiento y el de interpretabilidad
apuntan a lados opuestos, y ahora está medido.

**Hallazgo lateral que cambió una comparación previa.** Quitar el efecto de nodo
lleva el acuerdo entre atención y ablación de `rho` = +0.176 (`p` de permutación
0.145, no significativo) a **+0.471** (`p` = 0.0025), con 49 de 50 semillas de
acuerdo en signo. El efecto de nodo estaba tapando activamente la coincidencia.

**Bug corregido en el camino.** El primer nulo del ICC barajaba semillas dentro
de cada componente, lo que no cambia ni la media ni la varianza de columna, así
que el ICC nulo era idéntico al observado por construcción. El nulo correcto
baraja entre componentes dentro de cada semilla.

**Detalle algebraico.** En las fuentes normalizadas por fila, `corr(r,c)` =
+1.000 exacta con razón de desviaciones 0.0909 = 1/11. Es derivable: con suma de
fila constante y diagonal excluida, la fila `i` no ve `c_i`, y como los efectos
de columna suman cero, la media de fila queda en `+c_i/11`. El efecto emisor no
es una cantidad propia en esas matrices.

---

## 5. Intervenir en vez de leer

Módulos: `operador_bilineal.py`, `ablacion_atencion.py`

Dos vías en paralelo para ver si la atención se puede rescatar sin reentrenar.

### 5.1 El operador bilineal como parámetro

El logit de la primera capa se abre en nueve términos exactos con
`src = contenido + posición` y `M = W_Q^T W_K`. Los cuatro términos con sesgo
son unarios **por álgebra**. Control de exactitud: la suma de los nueve
reconstruye el logit del modelo con error máximo 1.82e-06.

Resultado, y es negativo:

```
objeto                                     fila  columna    PAR
logit de la capa 1 (antes de normalizar)  33.9%    65.9%    0.2%
rollout, ya normalizado                    1.0%    74.7%   24.3%
```

**El logit ya es 0.2% de par antes de tocar nada.** El sumidero no lo fabrica la
normalización: está en el logit. Esto corrige una hipótesis intermedia que
atribuía el problema a la renormalización por fila.

El término posición-posición `p_i^T M p_j`, que era el mejor candidato a
operador relacional puro, tiene ICC 0.044 entre semillas y Spearman medio
+0.023: no se reproduce. Y el espectro de `M` es indistinguible de un modelo sin
entrenar (rango efectivo 85.0 frente a 85.6, antisimetría 50.2% frente a 49.8%).

Un control de esta sección salió mal diseñado y se deja escrito: comparar los
pesos entrenados contra una inicialización fresca da 1.42 = raíz de 2 en todos
los bloques, que es lo que dan dos sorteos aleatorios independientes. No
distingue nada porque no se guardó el estado inicial de cada entrenamiento.

### 5.2 Ablación de aristas de atención

La idea que sí funciona. La atención es un objeto **observacional**: se mira lo
que el modelo hizo. La ablación de entrada es **intervencional**. Ese cambio de
estatus epistémico, y no una diferencia de calidad, explica la asimetría de
resultados. Así que se pone la atención en el mismo estatus:

```
Datt[v,j] = MSE(reconstruir v | A[v,j] = 0) - MSE(reconstruir v)
```

`forward_enmascarado` arma `B*N` secuencias donde la fila `b*N+v` es la variante
`v`, y de esa variante sólo se decodifica el canal `v`. Anular `A[v,j]`
únicamente en las filas cuya variante es `v` aísla el efecto por completo, y
permite cortar las 12 aristas de una columna en una sola pasada: 13 pasadas en
vez de 132, el mismo coste que la ablación de entrada.

El corte no renormaliza la fila, porque con `sigmoid` las filas no tienen por
qué sumar 1 y quitar una entrada es exactamente quitar ese canal.

```
fraccion de aridad 2                   81.4%    (rollout 24.3%, ablacion entrada 84.6%)
ICC entre 15 semillas                  0.650
aristas con efecto > 3 veces el ruido  19 de 132
rho_par con la ablacion de entrada    +0.893
```

**Cortar aristas de atención reproduce la matriz de ablación de entrada con
`rho` = +0.893.** Dos intervenciones sobre objetos distintos que coinciden.

El peso predice su efecto sólo con `rho` = +0.369 (`p` = 1.4e-05, 15/15
semillas). Reproducible, pero débil: discretizar con Top-P sobre el peso
selecciona aristas por una cantidad que apenas predice el efecto de quitarlas.

Aristas cuyo corte más duele: `ARI <- KNDVI` (12.9× el ruido),
`KNDVI <- CHL_REDEDGE` (12.8×), `NDVI <- CHL_REDEDGE` (12.1×). Las dos últimas
son las mismas que la persistencia con test binomial y BH dio en 50/50 semillas
con `q` = 4.2e-40 y que la curvatura marcó como puente. Cuarta confirmación
independiente.

---

## 6. Cuatro intentos de arreglar la lectura

Módulos: `atencion_cruda.py`, `entrenar_mascara_aleatoria.py`

Se probaron cuatro intervenciones sobre la lectura y el entrenamiento. Tres
fallaron, y los tres fallos son informativos.

### 6.1 Leer sin renormalizar

Descubrimiento: los 50 modelos **se entrenaron con
`ATENCION_NORMALIZACION = "sigmoid"`**, que existe precisamente para que una
fila pueda sumar menos de 1. Pero la lectura lo deshace en dos sitios
(`attention_rollout` divide por la suma de fila; la lectura por variante anula
la diagonal y vuelve a dividir), y las matrices guardadas tienen suma de fila
1.0000 con desviación 5e-08. El arreglo se aplicó al entrenar y se tiró al medir.

Al leer sin dividir aparece una observable que se estaba perdiendo:

```
NDWI 1.913   PSRI 1.727   NDII 1.104   ...   MARI 0.916   ARI 0.892
```

NDWI y PSRI miran el doble que el resto, y son de los que menos valor aportan.
Para NDWI encaja con que la máscara de cobertura diera 0% de píxeles de agua en
esta escena.

Corrección a una afirmación intermedia: renormalizar **no** destruye la señal de
par. Pone `frac_fila` en 0 por construcción, y esa varianza redistribuida infla
la cuota de PAR de 54% a 73%. Es denominador, no señal.

### 6.2 Ponderar por el valor — refutada

Predicción falsable: si lo que importa es `A[v,j]·v_j` y no `A[v,j]`, entonces
la norma de la contribución de Kobayashi et al. (2020),
`||suma_h A_h[i,j] O_h W_V^h x_j||`, debería predecir `Datt` mejor que +0.369.

```
A cruda              rho Datt +0.372
||c_ij||             rho Datt +0.374
A x norma de valor   rho Datt +0.364
```

Idénticos. El motivo está en los datos: las normas de valor van de 4.570 a
4.895, un 7% de rango, así que ponderar por ellas multiplica por algo casi
constante. **Hipótesis descartada.**

Lo que sobrevive es más simple: ninguna lectura estática del bloque de atención
—peso crudo, peso renormalizado, logit, norma de valor, contribución de
Kobayashi— recupera lo que sí recupera la intervención.

### 6.3 La condición de máscara importa 6 veces

Hallazgo lateral que resultó el mayor de esa sección. El mismo modelo, las
mismas semillas, leído bajo dos condiciones de máscara:

```
condicion de lectura                    fila    col    PAR   rho Datt
solo v oculto  (lo que lee el pipeline)  48%    43%     9%     +0.203
familia de v oculta                      28%    19%    54%     +0.372
```

Seis veces más estructura de par sin tocar el modelo. Explica por qué las
matrices guardadas del pipeline daban 3.8% y 21.8% de aridad 2 por capa: se
leyeron con `MASCARA_K_LECTURA = 1`, la condición mala.

### 6.4 Reentrenar con máscara aleatoria — refutada

Hipótesis: con `familia_balanceada` el patrón es casi determinista dado el
objetivo, así que el modelo puede aprender una política fija por objetivo, que
es exactamente una fila de atención que depende sólo de `v`. Con `k` variable el
conjunto visible cambia por muestra y la atención tendría que condicionarse.

Se reentrenaron 15 semillas con `k ~ Uniforme{1..10}` (validado: siempre ≥ 2
visibles, disponibilidad marginal plana entre 0.457 y 0.465). Mismas semillas en
los dos brazos para que la diferencia sea atribuible a la máscara.

```
                    fila    col    PAR     ICC   rho con Datt
base/identidad       28%    68%     4%   0.745      +0.064
alea/identidad        2%    93%     5%   0.452      +0.015
base/familia         15%    29%    56%   0.856      +0.308
alea/familia          4%    22%    74%   0.714      +0.290
```

Bajo la condición del pipeline la aridad 2 pasa de 4% a 5%, y el sumidero
**empeora** de 68% a 93%. Bajo la de familia el PAR sube, pero el efecto de fila
se derrumba de 15% a 4%, así que es denominador otra vez; y las dos medidas que
no dependen del reparto empeoran las dos (ICC 0.856 a 0.714, `rho` con `Datt`
+0.308 a +0.290).

Aleatorizar la máscara hizo la atención más genérica. Una política robusta
frente a máscaras de cualquier tamaño es una política que no depende de la
máscara.

**Nota de infraestructura.** Este entrenamiento se lanzó primero en el nodo de
login (`torii`, sin GPU) a 20 minutos por semilla. Corregido a
`sbatch --partition=student --qos=student --gres=gpu:1`, cada semilla tarda
0.2 a 0.3 minutos en una RTX 4070, con 26 a 50 épocas por early stopping.

---

## 7. Invariancia causal con la fenología como entorno

Módulo: `analisis_icp.py`

Peters, Bühlmann y Meinshausen (2016) parten de que si un conjunto de
predictores es el conjunto causal de una variable, el mecanismo no cambia
cuando cambia el entorno. Aquí los entornos vienen dados por el cultivo: las
cinco fases fenológicas de la vid particionan las fechas en regímenes
biológicamente distintos, y la partición la impone la planta.

Adyacencia dirigida por escena: `beta[i,j] = -Omega[i,j]/Omega[j,j]`, el
coeficiente de `i` al regresar `j` sobre los otros once, estimado en **cada
fecha**. La réplica es la escena y no el píxel: con 2704 píxeles por escena
cualquier diferencia saldría significativa por autocorrelación espacial.

**Dos bugs de diseño estadístico corregidos antes de correr sobre datos
reales.** El nulo por arista (desplazamiento circular de las etiquetas de fase)
tiene resolución mínima `1/n_fechas`, así que con BH sobre 132 aristas el `q`
mínimo alcanzable quedaba en 0.87 y el test no podía rechazar nada. Verificado
con un control sintético: un coeficiente que iba de 0.0 a 1.2 entre fases salía
con `q` = 0.995. Solución: agrupar el nulo entre aristas, legítimo porque
Kruskal-Wallis es libre de distribución y los tamaños de grupo son idénticos.
Tras el arreglo, el control sintético da `q` = 0.046 para el de régimen y 0.980
para el estructural, y ruido puro da 4.5% de `p` por debajo de 0.05.

Inflación del nulo sobre la chi-cuadrado asintótica: **2.27**. Usar el `p`
asintótico habría sido anticonservador por ese factor.

**El control positivo falla, y el motivo importa.** `KNDVI = tanh(NDVI²)` y
`MARI = ARI·B7` son identidades exactas y salen rechazadas. Ninguna es lineal
con pendiente constante: la pendiente de la tanh depende del nivel de NDVI y la
de MARI depende de B7, y ambos niveles son fenológicos. Este ICP prueba
invariancia del **coeficiente lineal**, no del mecanismo. Es la limitación
principal del módulo y hay que declararla. Lo que sí valida es que el test
detecta cambio de régimen donde se sabe que lo hay: `EVI2 -> KNDVI` va +3.86,
+5.82, +3.97, +1.13, +1.18 por fase, con pico en brotación cuando la tanh está
en su tramo empinado.

**Resultado: 68 aristas estructurales, 64 de régimen.**

```
celda                  n   regimen   frac       p
agua -> agua           6         6   100%   0.013
verdor -> verdor      20        12    60%   0.210
pigmento -> pigmento  12         3    25%   0.975
```

Las seis aristas entre índices de humedad son las seis estacionales. En un
viñedo con verano mediterráneo seco y riego, el estado hídrico oscila con la
estación, así que las relaciones mutuas entre NDWI, NDMI y NDII no son propiedad
del cultivo sino del momento del año.

**Contra este ground truth las fuentes fallan.** El `auc` de ordenar invariantes
por encima de estacionales va de 0.29 a 0.56, y varias capas de atención salen
**significativamente por debajo de 0.5**: rankean los artefactos estacionales
por encima de las relaciones estructurales, con `p` hasta 1e-5. Triangula con lo
de la aridad: la atención pesa redundancia, los pares redundantes son los de
dentro del bloque de verdor, y ésos son justo los de régimen.

---

## 8. Separar los regímenes

Módulos: `atencion_por_grupos.py`, `signo_por_fase.py`, `grafo_datt.py`

Último cambio de enfoque, y el que produce el resultado positivo.

**El error que arrastraba todo el proyecto.** Todas las evaluaciones se hicieron
sobre los 66 pares a la vez, mezclando pares intra-grupo colineal con
inter-grupo. La motivación para separarlos vino del teorema de imposibilidad
bajo colinealidad (arXiv 2605.21492), que predice que lo intra-grupo es
irrankeable y lo inter-grupo no.

**La predicción sale al revés, y ésa es la noticia.** Entre grupos las
correlaciones son reproducibles pero pequeñas (`|rho|` ≤ 0.36). Dentro de los
grupos colineales está todo:

```
dentro de familia espectral (14 pares), contra |pc|
fuente                       rho    semillas         q
atencion capa A1           -0.459    50-/50   8.9e-15
atencion capa A2           -0.407    50-/50   8.9e-15
atencion A1+A2             -0.433    50-/50   8.9e-15
atencion max sobre cabezas -0.354    50-/50   8.9e-15
atencion A2*A1             -0.327    47-/50   8.2e-11
Datt cortar arista         -0.398    15-/15   5.4e-05
dependencia por ablacion   +0.613    50+/50   8.9e-15
dependencia cruda          +0.508    50+/50   8.9e-15
```

Cinco lecturas de la atención coinciden en signo negativo y las dos ablaciones
de entrada en positivo, cada una en todas sus semillas. Promediar los dos
regímenes suma señal negativa intra-grupo con ruido inter-grupo y da cero: por
eso ninguna evaluación anterior vio nada.

**El teorema no explica esto.** Su mecanismo es la inestabilidad entre modelos,
y la estabilidad por mitades sale entre 0.92 y 0.99 en todos los casos, dentro y
fuera de grupo. El ranking es estable. La cita sirve para situar el problema de
la colinealidad, no para explicar el resultado.

**Interpretación del signo negativo.** Dentro de una familia los índices son
casi duplicados. La ablación, con la familia oculta como referencia, mide cuánto
ayuda devolver a un hermano: cuanto más dependen entre sí, más ayuda. La
atención hace lo contrario: cuanto más sustituible es `j` por otro hermano, más
reparte el peso. El peso de atención mide **no-sustituibilidad al margen**, que
dentro de un conjunto redundante es lo inverso de la dependencia condicional.

**Un confound que apareció y se resolvió.** Contra el `|beta|` del ICP los
signos se invierten. Causa medida: en los 14 pares intra-familia, `|pc|` y
`|beta|` correlacionan sólo +0.209 (`p` = 0.47), y 11 de esos 14 son "de
régimen", así que el `|pc|` agrupado promedia cinco regímenes incompatibles y no
es blanco estacionario. Sólo 3 de los 14 son invariantes.

`signo_por_fase.py` cierra el punto re-estimando `|pc|` **dentro de cada fase**,
donde no hay mezcla por construcción:

```
dentro de familia espectral   dormancia brotacion crecimien maduracio  acuerdo
atencion capa A1                -0.39     -0.26     -0.42     -0.40      4/4
atencion capa A2                -0.35     -0.26     -0.35     -0.32      4/4
atencion A2*A1                  -0.35     -0.27     -0.27     -0.23      4/4
atencion A1+A2                  -0.36     -0.26     -0.36     -0.35      4/4
atencion max sobre cabezas      -0.29     -0.20     -0.29     -0.27      4/4
Datt cortar arista              -0.41     -0.41     -0.36     -0.29      4/4
dependencia por ablacion        +0.54     +0.42     +0.48     +0.61      4/4
dependencia cruda               +0.53     +0.48     +0.32     +0.34      4/4
```

Todas conservan su signo en las cuatro fases con datos suficientes (postcosecha
cae por tener 7 fechas). La contaminación del blanco queda descartada: el signo
es de la fuente. La excepción es el rollout, que da 0/4 y por tanto no tiene
señal intra-familia que reportar, coherente con ser la lectura más contaminada
por su `0.25·I` y su renormalización.

Queda abierto por qué `|beta|` da el signo contrario. La réplica por fase
establece que no viene de promediar regímenes; la diferencia queda entre dos
estimadores de dependencia y no se resolvió.

**El grafo.** `grafo_datt.py` dibuja las 19 aristas dirigidas cuyo corte sube el
error por encima de tres veces el ruido, con el nivel de ruido estimado por la
cola negativa de `Datt` (cortar una arista no puede mejorar la reconstrucción
salvo por error de estimación). La flecha va del índice que informa al que lo
necesita; el trazo continuo marca las invariantes según el ICP y el punteado las
de régimen.

```
fuentes (cuantas veces informa)  CHL_REDEDGE 6  NDMI 3  NDII 3  KNDVI 3  NDVI 2  NDWI 2
7 invariantes, 12 de regimen
aristas reciprocas: 0 de 19
```

Cero aristas recíprocas de 19: cada relación fuerte medida va en un solo
sentido. CHL_REDEDGE es la fuente dominante y el bloque de verdor es casi todo
receptor, lo que invierte la expectativa de tratar NDVI como índice maestro.

---

## 9. Disociación por fase y el control de exposición

Módulos: `datt_por_fase.py`, `entrenar_balanceado.py`

Última pregunta del hilo: si el ICP clasifica las aristas en estructurales y de
régimen usando sólo el dato, ¿el modelo respalda esa clasificación? Se corre
$D_{att}$ restringiendo la entrada a cada fase y se mira cuántas fases aguanta
cada arista por encima de tres veces el ruido.

**Control que el diseño ingenuo no tiene.** El suelo de ruido depende del
tamaño de muestra, y las fases van de 8 a 90 fechas. Sin igualar, una arista se
"apaga" por falta de datos. Se igualaron fechas (8) y parches (128) en las
cinco fases, y el suelo se recalcula dentro de cada una con su propia cola
negativa.

**La predicción no se cumple.**

```
                              invariantes ICP (7)   de régimen ICP (12)
fases por encima del umbral          3.00                  2.83
Mann-Whitney unilateral                        p = 0.4134
```

La etiqueta del ICP no predice la persistencia de $D_{att}$. Las dos medidas no
se validan entre sí, y la clasificación del grafo queda apoyada sólo en el lado
del dato.

**Lo que sí apareció, y el confound que casi se me pasa.**

```
fase           aristas de las 19 sobre 3x    MSE base    suelo de ruido
dormancia                6                    0.2918        8.96e-03
brotacion               17                    0.1696        1.73e-03
crecimiento             15                    0.1159        1.20e-03
maduracion              13                    0.1094        1.28e-03
postcosecha              4                    0.1747        5.75e-03
```

La lectura inmediata fue biológica: sin hoja la reconstrucción se desestabiliza.
Pero al medir el MSE base por fase apareció otra explicación, y las fechas de
entrenamiento por fase son 8, 17, 40, 76 y 7. El modelo reconstruye peor donde
menos entrenó. Igualar la evaluación no toca ese desbalance.

**El control decisivo.** `entrenar_balanceado.py` reentrena 15 semillas
sobremuestreando cada fase hasta que todas pesen igual en la pérdida: dormancia
×10.00, postcosecha ×11.25, brotación ×4.09, crecimiento ×1.58, maduración
×1.00. Razones contra maduración:

```
fase           MSE base  ruido  |  MSE bal  ruido bal
dormancia         2.67    7.02  |    2.87     6.75
postcosecha       1.60    4.51  |    1.71     4.03
crecimiento       1.06    0.94  |    0.96     0.64
maduracion        1.00    1.00  |    1.00     1.00
```

Igualar el peso **no cierra la brecha**. La explicación por exposición queda
descartada. Lo que no se separa, y estaba declarado antes de correr: biología
frente a que 8 escenas distintas no alcancen. Repetirlas diez veces iguala el
gradiente sin añadir información; distinguirlo pide más fechas de invierno.

El brazo balanceado es peor modelo en las cinco fases, entre 2.3 y 2.6 veces de
MSE, porque la duplicación reduce la diversidad efectiva por paso de gradiente.
Sirve como control, no como estimación.

**Dos hallazgos del brazo base que este control tumba.**

La arista `ARI <- KNDVI` daba $D_{att}$ = −6.0 veces el ruido en brotación, o
sea que cortarla mejoraba la reconstrucción, y se propuso como ruta mal
aprendida. En el brazo balanceado da **+5.6**. Cambia de signo, así que no es un
defecto estable del modelo. Se retira.

La prueba contra el ICP empeora en el brazo balanceado: invariantes 1.57 fases
contra 2.17 de las de régimen, p = 0.8364, con la diferencia en sentido
contrario al esperado. En ninguno de los dos brazos hay apoyo.

**Lo que queda en pie de esta sección.** La dependencia interna del modelo
varía con la fase de forma sistemática y robusta al balanceo, con la estructura
concentrada en la estación de crecimiento. Eso no equivale a que el modelo
capture la biología: `reinferir_por_fase.py` ya había medido que evaluado en una
fase el modelo no expone la estructura de esa fase (2 aciertos de 5, azar 1 de
5). Lo compatible es más chico: la calidad de reconstrucción depende de la fase,
la estructura relacional no identifica la fase.

---

## 10. Conclusión

El trabajo demuestra que no es necesario sacrificar la capacidad no lineal y
espacial de los Transformers para obtener interpretabilidad. Mediante ablación
interventiva ($D_{att}$), convertimos la atención en un estimador causal interno
que ofrece el mismo rigor epistémico y transparencia que un modelo gráfico
clásico, pero sobre la variedad compleja de los datos espectrales.

### Alcance de lo demostrado

Lo que la evidencia sostiene, con los números que la respaldan:

| Propiedad | Modelo gráfico clásico | Transformer + $D_{att}$ |
|---|---|---|
| No linealidad | no | sí |
| Contexto espacial (parches 13×13) | no | sí |
| Dirección sin supuesto generativo | no | sí, por intervención |
| Grafo identificable | sí | sí |
| Reproducible entre ajustes | sí, analítico | sí, ICC 0.650 |
| Suelo de ruido y multiplicidad | sí, teoría | sí, cola negativa y BH |
| Coincide con `\|pc\|` | por definición | parcialmente, `rho` hasta 0.63 intra-familia |
| Validado contra verdad externa | no lo intenta | no, `auc` 0.29 a 0.56 |

### Lo que queda pendiente

La equivalencia epistémica está establecida; la superioridad en validez no. Los
dos operadores dan respuestas distintas y el ICP con fenología, que fue el
intento de arbitrar, no favorece a ninguno. El experimento que decidiría es una
verdad conocida por construcción y a la vez no lineal y espacial: simulación de
transferencia radiativa tipo PROSAIL con doseles sintéticos de parámetros
conocidos, calculando los 12 índices y midiendo qué método recupera la
estructura verdadera.

### Correcciones que hay que llevar al documento de tesis

1. Retirar como hallazgos el `b1` = 0, la curvatura constante, el grafo Top-P y
   el acoplamiento NMI = 1.0 de la atención: el sustituto unario los reproduce.
2. El test exacto de Fisher sobre bandas compartidas baja a `odds` 3.03,
   `p` = 0.0627 con NDMI corregido a B8A.
3. La afirmación de que no existe referencia externa es falsa: hay dos, el
   catastro CIREN y la fenología de la vid.
4. Las secciones que presentan la atención como fuente relacional cambian de
   signo y de régimen: la relación existe, es intra-familia, y es negativa.

---

## 11. Errores cometidos y cómo se detectaron

Se dejan escritos porque varios cambiaron conclusiones ya reportadas, y porque
el mecanismo de detección es reutilizable.

| Error | Cómo apareció | Efecto |
|---|---|---|
| Modularidad decidida contra la media del nulo en vez de contra cero | Etiquetas contradictorias en la misma tabla | Se invirtió el sentido en 6 de 8 fuentes |
| `b1` = 55 en las 8 fuentes | El mismo número en todas: señal de constante matemática | Se pasó al complejo de clíques |
| Entropía de von Neumann = 2.398 en todas | log(11), otra constante | Se pasó al laplaciano combinatorio |
| Asimetría dominada por efecto de nodo | Se midió el reparto: 85% de la varianza | Invirtió la conclusión sobre qué fuente tiene dirección |
| Nulo del ICC barajando dentro de columna | El p salía pegado a 1 pase lo que pasase | Permutar dentro de columna no cambia media ni varianza |
| Nulo del ICP por arista sin agrupar | Control sintético con efecto enorme daba q = 0.995 | Resolución mínima 1/n; se agrupó entre aristas |
| Control de distancia a la inicialización | Todos los bloques en 1.42 = raíz de 2 | Comparaba dos sorteos independientes; no distingue nada |
| "Igualé el tamaño de muestra" | Se midió el MSE base por fase | Sólo se había igualado la evaluación, no el entrenamiento |

El patrón que los une: **cuando un estadístico da el mismo valor en fuentes que
deberían diferir, o un p pegado a un extremo, casi siempre es la métrica y no el
dato.** Y el remedio que más veces funcionó fue construir un control sintético
con respuesta conocida antes de correr sobre datos reales.

---

## 12. Mapa de archivos

### Módulos (`modulos/`)

| Archivo | Qué hace |
|---|---|
| `certificar_todo.py` | Base del que dependen todos: fuentes, `\|pc\|`, `\|r\|`, familias, bandas, dirección heterocedástica, dependencia no lineal |
| `certificar.py`, `consolidar_resultados.py` | Certificación y consolidación previas |
| `analisis_grafo.py` | Reproducibilidad entre semillas, modularidad bilateral, comunidades con ARI, Top-P, PageRank |
| `analisis_espectral.py` | Markov, corriente neta, perfil espectral, ángulos principales, barrido de comunidades |
| `analisis_profundo.py` | Homología persistente, heat kernel, curvatura de Ollivier-Ricci, resolvente, matrices aleatorias |
| `analisis_fenologico.py` | Fases de la vid, catastro CIREN, estructura por fase |
| `reinferir_por_fase.py` | Re-inferencia de los 50 checkpoints restringiendo la entrada a cada fase |
| `analisis_sustitucion.py` | NMI entre aristas, reciprocidad, overlap por nodo, persistencia con binomial y BH |
| `analisis_atencion.py` | Descomposición por aridad, sustituto unario, ICC por aridad, covariables del ranking |
| `nmi_sustituto.py` | Comprueba si el acoplamiento NMI = 1 lo produce el sustituto unario |
| `analisis_icp.py` | Invariant Causal Prediction con las fases como entornos |
| `operador_bilineal.py` | Los nueve términos del logit, `M = W_Q^T W_K`, espectro contra un modelo sin entrenar |
| `ablacion_atencion.py` | $D_{att}$: cortar una arista y medir el cambio de MSE |
| `atencion_cruda.py` | Lectura sin renormalizar y ponderada por la contribución del valor |
| `entrenar_mascara_aleatoria.py` | Reentrenamiento con `k ~ Uniforme{1..10}` |
| `atencion_por_grupos.py` | Partición intra/inter grupo, estabilidad por mitades, resolución de grupo |
| `signo_por_fase.py` | El signo intra-familia contra `\|pc\|` estimado dentro de cada fase |
| `grafo_datt.py` | Figura del grafo de las 19 aristas |
| `datt_por_fase.py` | $D_{att}$ restringido a cada fase, con fechas y parches igualados y MSE base |
| `entrenar_balanceado.py` | Reentrenamiento con sobremuestreo por fase, control del confound de exposición |

### Pruebas (`pruebas/`)

`test_aridad.py` valida la descomposición con controles sintéticos:
recomposición exacta, ortogonalidad, matriz unaria pura, matriz de par pura, e
ICC con señal contra ruido.

### Resultados (`resultados/`)

Salidas `.txt` legibles y `.json` con los números completos de cada módulo. Los
sufijos `_base` y `_alea` distinguen los dos brazos de entrenamiento;
`_identidad` y `_familia` las dos condiciones de lectura.

### Figuras (`figuras/`)

`grafo_datt.pdf` (vectorial, para LaTeX) y `grafo_datt.png`.

---

## Reproducir

Los módulos importan `hello.py` del pipeline y esperan los checkpoints en
`resultados/sensitivity_K_experiment/`. En un clúster con SLURM:

```bash
srun --partition=student --qos=student --gres=gpu:1 --mem=16G \
     .venv/bin/python -u modulos/analisis_atencion.py --permutaciones 2000
```

Orden sugerido: `certificar_todo.py`, luego `analisis_atencion.py`,
`ablacion_atencion.py`, `analisis_icp.py`, `atencion_por_grupos.py`,
`signo_por_fase.py` y por último `grafo_datt.py`, que consume los JSON de
`ablacion_atencion.py` y `analisis_icp.py`.
