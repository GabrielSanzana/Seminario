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
13. [El experimento que faltaba: verdad conocida por construcción](#13-el-experimento-que-faltaba-verdad-conocida-por-construcción)
14. [El viñedo con retardos, y el hallazgo sobre el propio estimador](#14-el-viñedo-con-retardos-y-el-hallazgo-sobre-el-propio-estimador)

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

**El primer control estaba sesgado, y hubo que arreglarlo.** La partición
train/val original es cronológica, y el bloque de validación tiene 17 fechas de
crecimiento y 14 de maduración contra 1 de dormancia y 1 de postcosecha. El
early stopping y la selección de checkpoint optimizan ese reparto, así que un
brazo que gasta un quinto de su gradiente en dormancia queda penalizado por un
criterio donde esa fase casi no aparece. El control medía en contra de sí mismo.

El arreglo: partición tomando el último 20% **dentro de cada fase**, y
sobremuestreo también del val para que el criterio de parada pese igual las
cinco. Más un brazo de control con el mismo split y el mismo criterio pero sin
sobremuestrear el entrenamiento, de modo que la única diferencia sea ésa.

### 9.1 El estimador colgaba de qué fechas se eligieran

Las cuatro cifras que había en este lugar salían de `datt_por_fase.py`, que
igualaba el número de fechas entre fases tomando una submuestra fija con
`np.linspace`. Ese detalle resultó no ser un detalle.

Igualar hacía falta para el **suelo de ruido**, que se estima de la cola
negativa de $D_{att}$ y se encoge cuando hay más fechas. Para el **MSE** no
hacía ninguna falta: una media es una media con 8 fechas o con 90, sólo cambia
su varianza. Y ahí la submuestra fija hizo daño.

```
maduracion, brazo base
  8 fechas (linspace)   MSE 0.1094    razon dormancia/maduracion  2.67
  9 fechas (linspace)   MSE 0.1487    razon                       1.87
  90 fechas (todas)     MSE 0.1077    razon                       2.58
```

Una sola escena de maduración con MSE 0.4111, contra una mediana de 0.10, entra
en el muestreo de 9 y no en el de 8. La cifra publicada se movía un 30% por un
día. Sorteando la submuestra 2000 veces, la elección fija cae en el **percentil
1 a 4 en los cuatro brazos**: sesgaba a la baja de forma sistemática, porque
`linspace` siempre incluye la primera y la última fecha de la fase, que son las
de transición y las de mayor error.

Arreglado: el MSE usa todas las fechas y lleva intervalo por bootstrap; el
suelo sigue igualando fechas pero promediando 200 sorteos al azar. Las dos
cantidades se calculan del tensor de error por fecha de `errores_por_fecha.py`,
y `test_errores.py` comprueba que ese tensor reproduce la pasada por GPU al
0.00%.

### 9.2 Las cifras, ya con intervalo

```
brazo                  razon MSE dormancia/maduracion   IC95        razon del suelo
cronologico  base                 2.58              [2.04, 3.16]        9.13
cronologico  bal                  2.27              [1.69, 2.90]        6.12
estratificado ctrl                1.54              [1.13, 2.00]        2.24
estratificado bal                 1.89              [1.42, 2.43]        3.96
```

La comparación limpia es el par estratificado, que sólo difiere en el
sobremuestreo del train: dormancia pasa de 1.54 a 1.89 y su suelo de 2.24 a
3.96. Balancear la exposición sigue sin cerrar la brecha; la ensancha un poco.

El efecto es **bastante menor de lo reportado en cualquier versión anterior**.
En el brazo limpio dormancia es 1.5 veces peor que maduración, no 2.1 ni 2.7, y
su suelo de ruido 2.2 veces mayor, no 3.5 ni 7. El límite inferior del
intervalo sigue por encima de 1 en los cuatro brazos, así que el efecto existe;
su tamaño era el inflado.

### 9.3 Cuatro controles que el efecto podía no sobrevivir

`control_fases.py` somete la razón a cuatro pruebas que hasta aquí no se habían
corrido. Ninguna de las cuatro lo tumba, y una lo refuerza.

```
                            base      bal     ctrl     bal2
placebo de etiquetas      0.0005   0.0005   0.0005   0.0005
calendario rotado         0.0833   0.0833   0.0833   0.0833
suelo con n igualado      0.0000   0.0000   0.0000   0.0000
razon ya normalizada        2.47     2.38     1.89     2.19
```

**Placebo de etiquetas.** Se reparten las fechas al azar en grupos de los
mismos tamaños, 2000 veces. La razón nula tiene mediana 0.99 e IC95 de 0.65 a
1.48; la observada queda fuera en los cuatro brazos. El salto no es el tamaño
del grupo.

**Calendario rotado.** Se gira el ciclo fenológico de 1 a 11 meses, con lo que
los grupos conservan tamaño, recurrencia anual y contigüidad y sólo dejan de
coincidir con el ciclo de la vid. El calendario real da la razón más alta de
las doce en los cuatro brazos (base: 2.58 contra un máximo rotado de 2.05). El
p de 0.0833 es el mínimo alcanzable con doce rotaciones, o sea la evidencia
máxima que este control puede dar.

**Suelo de ruido con el mismo número de fechas.** El control decisivo del
suelo: se estima el de maduración con las mismas 9 fechas que tiene dormancia,
2000 veces. Nunca alcanza el de dormancia, en ningún brazo. La diferencia de
suelos no es un artefacto de tamaño de muestra, que era la explicación más
económica y la que quedaba viva.

**Dificultad de la escena.** Un MSE alto podía significar "el modelo falla ahí"
o "esa escena es más heterogénea y cualquiera fallaría" — plausible en un viñedo
invernal con suelo desnudo y sombras largas. La referencia es el predictor
trivial que devuelve la media espacial del canal tapado, cuyo error es la
varianza espacial. Resultado: las escenas de dormancia son **menos** heterogéneas
que las de maduración (0.690 contra 0.970), así que normalizar no baja la razón
sino que la sube. Descartado.

### 9.4 Qué predice el error de una fecha

Con 186 fechas en vez de una tabla de cinco filas, las explicaciones que
compiten se pueden meter en el mismo modelo. Coeficientes estandarizados, brazo
base:

```
exposicion (log fechas de train de su fase)   -0.378
NDVI de la fecha                              +0.436
dificultad (log varianza espacial)            -0.089     R2 0.280
```

La correlación entre exposición y NDVI es +0.291, baja, así que los dos
coeficientes son separables aquí.

El resultado incómodo está dentro de cada fase, donde la exposición es
constante por construcción: la correlación entre error y NDVI es **positiva en
las cinco fases** (+0.23 a +0.62, significativa en tres). Más verde, más error.
Entre fases el signo se invierte, porque dormancia tiene poco NDVI y mucho
error. Los dos niveles dicen cosas opuestas, y la lectura "sin hoja los índices
degeneran y el modelo se pierde" no sobrevive a la versión intra-fase.

Lo que queda, dicho sin adornos: el error depende de la fase de forma
sistemática y robusta, y **no se sabe por qué**. Exposición, biología y
dificultad de la escena quedan las tres descartadas o insuficientes.

### 9.5 Dos hallazgos del brazo base que este control tumba

La arista `ARI <- KNDVI` daba $D_{att}$ = −6.0 veces el ruido en brotación, o
sea que cortarla mejoraba la reconstrucción, y se propuso como ruta mal
aprendida. En los otros tres brazos da **+5.6, +9.0 y +4.4**. Cambia de signo y sólo
aparece con la partición cronológica original, así que no es un defecto del
modelo. Se retira.

La prueba contra el ICP no encuentra apoyo en **ninguno de los cuatro brazos**,
y en tres de ellos las invariantes aguantan menos fases que las de régimen. Con
19 aristas y 7 invariantes hay $\binom{19}{7} = 50388$ repartos posibles, así
que el p se enumera exacto en vez de aproximarse:

```
brazo   p exacto   p minimo alcanzable
base     0.4146          0.0001
bal      0.8577          0.0050
ctrl     0.7477          0.0017
bal2     0.8723          0.0014
```

La segunda columna es la que faltaba. Sin ella, un p alto no distingue "no hay
señal" de "el test no podía detectarla". Aquí el test **podía** llegar a 0.0001
y llegó a 0.41: es ausencia de señal, no falta de potencia. El desacople entre
la medida del dato y la del modelo es un resultado, no un empate.

**Lo que queda en pie de esta sección.** La dependencia interna del modelo
varía con la fase de forma sistemática, con la estructura concentrada en la
estación de crecimiento. El efecto sobrevive a cuatro controles —placebo de
etiquetas, calendario rotado, suelo con `n` igualado, dificultad de escena— en
los cuatro brazos de entrenamiento, y su tamaño es de 1.5 a 2.6 veces según el
brazo, con el límite inferior del IC95 siempre por encima de 1.

Eso no equivale a que el modelo capture la biología: `reinferir_por_fase.py` ya
había medido que evaluado en una fase el modelo no expone la estructura de esa
fase (2 aciertos de 5, azar 1 de 5), y la correlación intra-fase entre error y
NDVI sale positiva, al revés de lo que pediría la lectura biológica. Lo
compatible es más chico: la calidad de reconstrucción depende de la fase, la
estructura relacional no identifica la fase, y la causa de la dependencia queda
sin identificar tras descartar exposición, biología y dificultad de escena.

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
| Recupera estructura conocida (sintética) | sí, `auc` 0.98 lineal | no, `auc` 0.68 (sección 13) |
| Orienta contra verdad conocida | hasta la clase de Markov | sí, 0.645 no lineal y 0.974 temporal |
| Funciona con mecanismos retardados | no, `auc` 0.478 | sí, `auc` 0.722, empata con LOCO con retardos |

### Lo que queda pendiente

La equivalencia epistémica está establecida; la superioridad general en
validez está **refutada**, pero hay una excepción medida. La sección 13 corrió
el experimento que faltaba —verdad conocida por construcción— sobre cuatro
regímenes. $D_{att}$ pierde en el lineal, el no lineal y el espacial, donde la
recomendación es LOCO con árboles potenciados. **Empata en el temporal**, que
es el único donde ningún método clásico estándar pasa del azar, y ahí su
orientación llega a 0.974 sobre 76 aristas verdaderas.

La afirmación defendible no es "recupera mejor" sino "recupera donde lo
clásico no llega, sin que se le diga dónde mirar". Con una condición previa:
la configuración del caso de estudio (`SEQ_LENGTH = 1`, tarea enmascarada sin
ventanas temporales) impide captar tiempo, así que hay que reentrenar con los
retardos como tokens antes de sostener nada temporal sobre el viñedo.

El experimento que quedaba pendiente y ya no lo está era una
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
| El control de exposición usaba un val cronológico | 31 de 38 fechas de val eran crecimiento y maduración | El criterio de parada penalizaba justo lo que el control quería medir |
| El arreglo del split podía meter la misma fecha en train y en val | Se escribió el test antes de confiar en la función | Sólo se dispara con fases de 1 o 2 fechas; no llegó a afectar a ninguna cifra |
| El MSE por fase colgaba de una submuestra fija de fechas | Cambiar 8 fechas por 9 movía la razón de 2.67 a 1.87 | Una escena con MSE 0.41 entre 90; la elección fija caía en el percentil 1 a 4 |
| Igualar fechas se aplicaba también donde no hacía falta | Una media no necesita igualarse, sólo su varianza cambia | El efecto real es 1.5x en el brazo limpio, no 2.1x ni 2.7x |
| Se reportaban razones entre fases sin ningún intervalo | Cinco puntos sin incertidumbre admiten casi cualquier relato | Con IC95 el efecto sigue vivo pero es la mitad de grande |
| Nunca se había probado un placebo de fases | Se corrieron dos, etiquetas al azar y calendario rotado | Los pasa los dos; era el control que faltaba para llamarlo resultado |
| El enmascarado por familia dejaba 24% de las aristas fuera del alcance del modelo | Se contaron: 23 de 95 son intra-familia y esa familia va siempre oculta | Techo sobre la puntuación, no "ruido de configuración" como se había escrito |
| Se comparó cortar una arista de atención contra quitar una variable | Son preguntas distintas: residual y convolución enrutan alrededor de la arista | La orientación pasa de 0.526 a 0.645 al medirla como corresponde |
| Se concluyó que la direccionalidad no existe con el experimento mal montado | Al arreglar los dos anteriores sale 0.645 y 0.974 con p bajo | Se retira la retirada: la afirmación central del framework se sostiene |
| El modelo del caso de estudio no puede ver tiempo | `SEQ_LENGTH = 1` y la tarea enmascarada descarta ventanas por diseño | Toda afirmación temporal del documento queda sin respaldo hasta reentrenar |
| El test del parche 26×26 no separaba lo que decía separar | Cuadruplicar el parche divide por cuatro las muestras; el `val` sube a 0.944 | Test contaminado: sirve si sale bien, no si sale mal |
| Se umbralizaron los tres bloques de retardo con el suelo de la matriz 36×36 completa | Los suelos por bloque van de 4.11e-5 a 2.02e-4, un factor 5 | Inflaba el conteo de aristas del presente |
| El grafo de 19 aristas se presentaba como "el grafo del modelo" | Con `k=1` el `rho` contra él es 0.05 y el solapamiento no supera el azar | Es el grafo bajo `familia_balanceada`; la condición es parte del resultado |
| El suelo de ruido por cola negativa se daba por universal | Bajo `k=1` hay 0 de 132 valores negativos y el suelo sale `nan` | El estimador exige un enmascarado que produzca cortes con efecto negativo |

El patrón que los une: **cuando un estadístico da el mismo valor en fuentes que
deberían diferir, o un p pegado a un extremo, casi siempre es la métrica y no el
dato.** Y el remedio que más veces funcionó fue construir un control sintético
con respuesta conocida antes de correr sobre datos reales.

Los seis últimos añaden un patrón propio: **un número sin intervalo, sin
placebo y sin test de su propio estimador no es un resultado todavía.** Cuatro
de las seis filas no se detectaron mirando salidas raras sino escribiendo el
test que la función debía pasar. Las pruebas están en `pruebas/`:
`test_particion.py` con 22 casos sobre el reparto por fase y el sobremuestreo,
y `test_errores.py` con 8 que comprueban que el camino de cálculo nuevo
reproduce el viejo hasta 1e-9.

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
| `datt_por_fase.py` | $D_{att}$ por fase: MSE con todas las fechas e intervalo, suelo de ruido igualando fechas por sorteo |
| `entrenar_balanceado.py` | Reentrenamiento con sobremuestreo por fase, partición estratificada, control del confound de exposición |
| `errores_por_fecha.py` | Tensor `E[semilla, fecha, canal, corte]` en una pasada por GPU; de él salen todas las agrupaciones sin volver a evaluar |
| `verdad_sintetica.py` | Genera los tres regímenes sintéticos con el DAG conocido por construcción |
| `entrenar_sintetico.py` | Entrena el ConvTransformer sobre cada régimen y calcula $D_{att}$ |
| `benchmark_recuperacion.py` | Siete estimadores de estructura puntuados contra la misma verdad |
| `resumen_replicas.py` | Agrega las réplicas; la orientación se suma, no se promedia |
| `entrenar_temporal_tokens.py` | Los retardos como tokens (12 índices × 3 instantes = 36) para darle tiempo al transformer sin tocar `forward_enmascarado` |
| `grafo_temporal.py` | Reparto de la dependencia por instante y comparación con el grafo publicado |
| `control_fases.py` | Los cuatro controles del efecto por fase: bootstrap, placebo de etiquetas, calendario rotado, suelo con `n` igualado, dificultad de escena, regresión por fecha, ICP con permutación exacta |

### Pruebas (`pruebas/`)

`test_aridad.py` valida la descomposición con controles sintéticos:
recomposición exacta, ortogonalidad, matriz unaria pura, matriz de par pura, e
ICC con señal contra ruido.

`test_particion.py`, 22 casos sobre el reparto por fase y el sobremuestreo. Los
que importan: que `FASES` sea una partición de los doce meses (un mes fuera
haría desaparecer sus frames del entrenamiento sin ningún aviso), que train y
val nunca compartan una fecha, que cada fase llegue al val, que el
sobremuestreo no pierda ni invente índices, y que una desalineación entre
frames y fechas —la que produciría `SEQ_LENGTH > 1`— levante error en vez de
etiquetar mal cada frame. Corre sin GPU y sin datos, sobre un calendario
sintético **ordenado en el tiempo**: con las fechas barajadas los tests pasan
sin probar nada, porque el sesgo que se está midiendo nace del orden.

`test_errores.py`, 8 casos que atan el camino de cálculo nuevo al viejo:
`errores_por_frame` promediado da `error_por_canal` hasta 4e-9 con y sin arista
cortada, el parche `k` viene de la fecha `k // 16`, y el MSE por fase reproduce
el `DATT_POR_FASE.json` publicado con **0.00%** de desviación. Sin este último
los controles de `control_fases.py` estarían midiendo otra cosa que las cifras
del texto.

### Resultados (`resultados/`)

Salidas `.txt` legibles y `.json` con los números completos de cada módulo. Los
sufijos `_base` y `_alea` distinguen los dos brazos de entrenamiento;
`_identidad` y `_familia` las dos condiciones de lectura.

### Figuras (`figuras/`)

`grafo_datt.pdf` (vectorial, para LaTeX) y `grafo_datt.png`.

---

## 13. El experimento que faltaba: verdad conocida por construcción

La sección 10 dejó escrito que la superioridad en validez no estaba demostrada
y nombró el experimento que la decidiría: una verdad conocida por
construcción, a la vez no lineal y espacial. Está hecho, y el resultado va en
contra del framework.

### 13.1 El diseño

Un DAG disperso de 19 aristas sobre 12 nodos, la misma densidad que el grafo
real, genera campos de 186×52×52×12. Tres regímenes:

```
lineal      mecanismos lineales, ruido gaussiano. Los supuestos del metodo
            clasico se cumplen EXACTAMENTE. Es una trampa deliberada: si el
            framework gana aqui, el montaje esta mal.
no lineal   interacciones entre padres y no linealidad simetrica (x^2). La
            segunda anula la correlacion de Pearson dejando la dependencia
            intacta.
espacial    el mecanismo pasa por el LAPLACIANO y la media local del padre,
            no por su valor en el pixel. Es el unico regimen donde el
            contexto espacial aporta, y por tanto el unico donde la
            arquitectura convolucional puede justificar su coste.
```

Los campos son aleatorios suavizados por filtrado en frecuencia, no píxeles
independientes. Sin eso la parte convolucional no tendría nada que hacer y la
comparación estaría amañada a favor del método clásico.

**El blanco justo no es el DAG.** El predictor óptimo de $X_i$ dado el resto
usa su manto de Markov —padres, hijos y cónyuges— así que ningún método basado
en dependencia condicional puede recuperar el DAG: ni $D_{att}$, ni la
correlación parcial, ni LOCO. Se puntúa contra el esqueleto, contra el grafo
moralizado, y por separado la orientación de las aristas verdaderas.

**Un solo DAG no alcanza.** Con 19 aristas el error estándar de la orientación
bajo la nula es 0.115, así que 0.63 y 0.37 caben los dos dentro del azar. Se
replicó con cuatro DAG independientes: 76 ensayos, error estándar 0.057.

### 13.2 El montaje es correcto

```
REGIMEN LINEAL, 4 replicas    auc esq     sd  auc moral  prec@k  orientacion
correlacion parcial             0.983  0.012      0.963    0.88   simetrico
LOCO con arboles                0.977  0.012      0.931    0.88   0.671  p 0.0019
LOCO con vecindario             0.975  0.015      0.936    0.86   0.671  p 0.0019
PC (Fisher z)                   0.918  0.051      0.704    0.86   simetrico
correlacion                     0.865  0.048      0.634    0.64   simetrico
```

Gana la correlación parcial, que es lo que tenía que pasar donde sus supuestos
se cumplen exactamente. Si el framework hubiera ganado aquí, habría que
sospechar del experimento antes que celebrarlo.

### 13.3 Lo no lineal sí rompe a lo clásico, pero no gana el transformer

```
REGIMEN NO LINEAL             auc esq     sd  auc moral  prec@k  orientacion
LOCO con arboles                0.912  0.089      0.854    0.76   0.763  p 0.0000
informacion mutua               0.900  0.036      0.740    0.67   simetrico
LOCO con vecindario             0.893  0.104      0.844    0.74   0.697  p 0.0004
distancia de corr.              0.842  0.062      0.715    0.67   simetrico
correlacion parcial             0.710  0.090      0.703    0.57   simetrico
PC (Fisher z)                   0.631  0.093      0.629    0.38   simetrico
Datt identidad                  0.613  0.114      0.580    0.46   0.526  p 0.3655
Datt familia                    0.601  0.139      0.587    0.43   0.618  p 0.0252
```

La predicción teórica se cumple: PC cae de 0.918 a **0.631**, por debajo de la
correlación cruda, con precisión en `k` de 0.38. Con interacciones y no
linealidad simétrica, los métodos de correlación se desploman.

**Quien recoge los pedazos es LOCO con árboles potenciados**: 0.912 de AUC y
**0.763 de orientación con p < 0.0001**. Sin red neuronal, sin atención, sin
contexto espacial. Es la misma lógica interventiva de $D_{att}$ aplicada a las
variables en vez de a las aristas de atención.

### 13.4 Ni siquiera en su propio terreno

```
REGIMEN ESPACIAL              auc esq     sd  auc moral  prec@k  orientacion
LOCO con vecindario             0.952  0.016      0.823    0.83   0.684  p 0.0009
correlacion parcial             0.901  0.039      0.742    0.75   simetrico
LOCO con arboles                0.883  0.038      0.724    0.76   0.539  p 0.2833
informacion mutua               0.830  0.047      0.659    0.62   simetrico
Datt familia                    0.619  0.077      0.593    0.48   0.553  p 0.2111
Datt identidad                  0.618  0.063      0.592    0.48   0.513  p 0.4544
```

El régimen espacial se diseñó para que un método por píxel esté ciego por
construcción. Para no amañarlo al revés, LOCO recibe además la media 5×5 de
cada canal: la misma información de vecindario que ve el modelo. Con eso llega
a 0.952 contra 0.618 de $D_{att}$.

Dar acceso al vecindario a unos árboles potenciados basta. La arquitectura
convolucional no aporta sobre eso.

### 13.5 El único hallazgo positivo, y es sobre el estimador

La primera tanda entrenó 50 épocas y el log mostraba el régimen lineal
mejorando todavía en la época 48. Infraentrenado. Concluir "$D_{att}$ falla"
con ese log habría sido atribuir al fenómeno lo que era del protocolo. Se
repitió con 600 épocas y paciencia 60, con la predicción escrita antes de
mirar el resultado:

```
regimen      val 50 ep   val 600 ep   varianza expl.   auc de Datt
lineal          0.643       0.634       36% a 37%      0.844 a 0.814
no_lineal       0.811       0.801       19% a 20%      0.665 a 0.663
espacial        0.958       0.848        4% a 15%      0.513 a 0.695
```

Los dos regímenes ya convergidos no se movieron. El espacial, que era el único
sin converger, pasó de **azar exacto (0.513) a 0.695**.

$D_{att}$ recupera exactamente en la medida en que el modelo aprendió. No es
un fallo del estimador: es fidelidad. **$D_{att}$ audita al modelo, y el
modelo era el eslabón débil.** Ésa es la propiedad que se le pide a un
auditor, y es el resultado positivo del experimento.

### 13.6 Lo que hay que retirar de la tesis

1. Que $D_{att}$ recupere estructura mejor que un método clásico. Queda último
   en los tres regímenes, con AUC de 0.60 a 0.62 contra 0.88 a 0.98.
2. ~~Que la asimetría de $D_{att}$ indique dirección causal.~~
   **Esta conclusión se retira en 13.13.** Venía de comparar el corte de una
   arista de atención contra la ablación de una variable, que no son la misma
   pregunta, y de un enmascarado que dejaba un cuarto del grafo fuera del
   alcance del modelo. Corregidos los dos, la orientación es 0.645 con
   p = 0.0077 en el régimen no lineal y 0.974 en el temporal.
3. Que la arquitectura espacial aporte. Árboles con rasgos de vecindario la
   superan, 0.952 contra 0.618.

### 13.7 Por qué falla, que es lo publicable

Cortar una arista de atención **no elimina la variable**. El ConvTransformer
tiene convolución y conexiones residuales, así que la información se enruta
por otro camino. $D_{att}$ mide cuánto aporta *esa arista*, no cuánto aporta
*esa variable*.

Eso reconcilia las dos observaciones que parecían contradictorias: el `rho`
+0.893 con LOCO sobre el dato real —coinciden en magnitud, porque las dos
miden dependencia— y el fracaso en orientación, porque la asimetría de una
arista de atención no hereda la asimetría causal que sí tiene la ablación de
una variable.

**La ablación de atención no es un sustituto de la ablación de variable.**
Buena parte de la literatura de XAI sobre atención lo asume sin comprobarlo.
Aquí está comprobado que no, con verdad conocida y cuatro réplicas.

### 13.8 Lo que queda en pie

$D_{att}$ sigue siendo válido para lo que nunca dejó de ser: **auditar un
modelo desplegado**, donde el modelo *es* el objeto de estudio y la pregunta
"de qué depende esta decisión" no tiene método clásico, porque el modelo no es
un proceso natural del que tomar muestras sino un artefacto que se puede
intervenir directamente. Ahí no compite con nadie.

Para descubrir estructura en el mundo, la recomendación que sale de este
experimento es LOCO con árboles potenciados: más barato, más simple y mejor en
los tres regímenes.

### 13.9 Límites de este experimento

Cuatro DAG, cinco semillas por DAG, una arquitectura, 12 nodos. No es un
barrido. Lo que sí está establecido con esa potencia es lo negativo: con 76
aristas verdaderas y error estándar 0.057, una orientación de 0.51 a 0.62 no
sostiene la afirmación de direccionalidad, y una AUC de 0.61 contra 0.91 de
LOCO no sostiene la de recuperación.

Lo que no está establecido es el límite superior: la relación monótona entre
ajuste del modelo y AUC de $D_{att}$ deja abierto que un modelo mucho mejor
entrenado lo acerque a LOCO. Es contrastable y barato, y es lo siguiente que
haría falta correr.

### 13.10 Tres handicaps de configuración, y qué pasa al quitarlos

La primera versión de este experimento concluyó que $D_{att}$ pierde en todos
los regímenes y que su direccionalidad no se sostiene. Al revisar el código
aparecieron tres defectos, dos de diseño del experimento y uno de la
configuración de la tesis. Ninguno se veía en las cifras de salida.

**Primero: el enmascarado por familia hacía inaprendible un cuarto del grafo.**
`MASCARA_MODO="familia_balanceada"` oculta *siempre* la familia entera del
índice que se reconstruye. Sobre el viñedo eso es deliberado y correcto: mata
el atajo colineal entre índices que comparten bandas. Sobre datos sintéticos
las familias son una partición arbitraria de los doce canales, y una arista
verdadera entre dos canales de la misma familia queda fuera del alcance del
modelo por construcción, porque nunca ve uno con el otro disponible.

```
datos_sinteticos    4 de 19 aristas dentro de familia
datos_rep11         4 de 19
datos_rep12         4 de 19
datos_rep13         6 de 19
datos_rep14         5 de 19
TOTAL              23 de 95 = 24% inaprendibles
```

La versión anterior de este README llamó a eso "ruido de configuración, ni
favorece ni perjudica". Es falso: es un techo sobre la puntuación. Corregido
con `MASCARA_K = 1` —ocultar sólo el canal objetivo—, el modelo aprende
$E[X_i \mid \text{los otros once}]$, cuya estructura de dependencia es
exactamente el manto de Markov que se quiere recuperar. El `val` baja de 0.801
a 0.690 y el AUC sube de 0.613 a 0.682.

**Segundo: se comparaba cortar una arista contra quitar una variable.**
$D_{att}$ corta una arista de la matriz de atención; LOCO con árboles quita la
variable entera. No son la misma pregunta, y presentarlas como tal fue un
error de diseño. Con conexiones residuales y una rama convolucional, cortar
una arista de atención deja abiertos otros caminos por los que la misma
información vuelve a entrar.

El competidor justo del LOCO con árboles no es $D_{att}$ sino la ablación de
**entrada** hecha con el mismo transformer, que el pipeline ya tenía en
`matriz_dependencia_ablacion`:

$$D[i,j] = \text{MSE}(\text{reconstruir } i \mid i,j \text{ ocultos}) - \text{MSE}(\text{reconstruir } i \mid i)$$

**Tercero, y es el que más importa: el modelo nunca tuvo acceso al tiempo.**
En `process_indices_data`:

```
if TAREA == "enmascarado":
    # No hay ventanas temporales: cada frame es independiente
```

y `forward_enmascarado` codifica cada token con `reshape(B*N, 1, H, W)`: una
imagen, un instante. `ConvEncoder` sí admite `in_channels = seq_length`, o sea
que la arquitectura soporta ventanas, pero el camino de la tarea enmascarada
no las usa. La tesis corre con `SEQ_LENGTH = 1`.

Consecuencia inmediata: **toda afirmación del documento sobre relaciones
temporales estaba sin respaldo**, no porque el transformer no pueda captarlas
sino porque no se le estaban dando.

### 13.11 Un cuarto régimen: el tiempo

Se añadió el régimen `temporal`, donde $X_i(t)$ depende de $X_p(t-1)$ y
$X_p(t-2)$ y no de $X_p(t)$. En el mismo instante la dependencia es débil, así
que cualquier método que trate cada fecha como una observación independiente
está ciego por construcción.

Para que la comparación siga siendo simétrica, LOCO recibe los mismos
retardos, igual que en el régimen espacial recibió la media 5×5. Y al
transformer se le dan como tokens adicionales: 12 índices × 3 instantes = 36
tokens, con la atención viviendo en una matriz 36×36 donde una arista puede
cruzar instantes. Para puntuar contra el DAG de 12 nodos se suman las columnas
de los tres instantes de cada variable, que es lo que hace LOCO con retardos
al ablacionar todos los retardos de $j$ a la vez.

No se reescribió `forward_enmascarado`: es la ruta optimizada del pipeline y
romperla afectaría a todo lo demás.

```
REGIMEN TEMPORAL, 4 DAG        auc esq     sd  auc moral  prec@k  orientacion   n       p
LOCO con retardos                0.730  0.035     0.669    0.57     1.000      76  0.0000
Datt entrada tokens              0.722  0.041     0.604    0.58     0.961      76  0.0000
Datt atencion tokens             0.706  0.014     0.687    0.55     0.974      76  0.0000
Datt entrada neutro (12 tokens)  0.436  0.034     0.557    0.26     0.632      76  0.0143
LOCO con arboles                 0.461  0.128     0.486    0.22     0.526      76  0.3655
LOCO con vecindario              0.467  0.072     0.444    0.27     0.500      76  0.5456
correlacion parcial              0.478  0.094     0.530    0.33   simetrico
PC (Fisher z)                    0.442  0.043     0.541    0.28   simetrico
informacion mutua                0.463  0.111     0.542    0.29   simetrico
```

Dándole el pasado, el transformer pasa de 0.436 a **0.722** y su orientación de
0.632 a **0.974**: 74 de 76 aristas verdaderas bien orientadas. Empata con LOCO
con retardos dentro de la desviación entre réplicas, y contra el grafo
moralizado lo supera, 0.687 frente a 0.669.

Ningún método clásico estándar funciona en este régimen. El único que compite
es LOCO con retardos, que ya es una construcción *ad hoc* con el pasado
inyectado a mano: alguien tuvo que decidir que los retardos relevantes eran 1 y
2. El transformer llega al mismo sitio sin que se le diga qué mirar.

Y aquí la ablación de atención funciona tan bien como la de entrada, 0.706 y
0.974 frente a 0.722 y 0.961. Tiene una explicación: con la información
temporal repartida en tokens separados, cortar la arista de atención sí aísla
la variable, porque ya no hay una ruta residual que traiga lo mismo desde otro
instante.

### 13.12 Lo espacial sigue sin explicarse, y el test que hice no vale

```
REGIMEN ESPACIAL          auc esq  orientacion       p
LOCO con vecindario         0.952     0.684      0.0009
correlacion parcial         0.901   simetrico
Datt atencion neutro        0.634     0.408      0.9577
Datt atencion p26           0.406     0.474      0.7167
```

La hipótesis era que un laplaciano no sobrevive a dos convoluciones de stride 2
sobre un parche de 13×13, que llegan a la atención con 4×4 posiciones. Se probó
con `TOKEN_PARCHE = 26` y salió peor.

**Ese test no concluye nada**, y conviene decirlo antes que apoyarse en él:
cuadruplicar el área del parche divide por cuatro las muestras de
entrenamiento, de 2368 a 592, y el `val` sube a 0.944. No distingue "el campo
receptivo no era el problema" de "se quedó sin datos". Separarlo pediría
parches grandes con solape, o más fechas.

Lo que sí queda establecido es que en el régimen espacial el transformer
pierde y su orientación cae por debajo del azar (0.408, 0.395), mientras unos
árboles con la media 5×5 alcanzan 0.952. Sin explicación por ahora.

### 13.13 El cuadro final

```
regimen      mejor clasico              mejor transformer        veredicto
lineal       0.983 correlacion parcial   -                       clasico, como debe
no lineal    0.912 LOCO con arboles      0.682 Datt neutro       clasico
espacial     0.952 LOCO con vecindario   0.634 Datt neutro       clasico
temporal     0.730 LOCO con retardos     0.722 Datt tokens       empate
```

Y la orientación, que es la afirmación propia del framework:

```
regimen      Datt          p         LOCO mejor
no lineal    0.645     0.0077          0.763
espacial     0.408     0.9577          0.684
temporal     0.974     0.0000          1.000
```

**Lo que se sostiene.** La direccionalidad de $D_{att}$ es real: 0.645 en el
régimen no lineal y 0.974 en el temporal, las dos con $p$ pequeño sobre 76
aristas verdaderas. La sección 13.6 de la versión anterior la daba por
refutada, y esa conclusión se retira: venía de una comparación mal montada
—arista contra variable— y de un enmascarado que tapaba un cuarto del grafo.

**Lo que sigue sin sostenerse.** Que $D_{att}$ recupere el esqueleto mejor que
un método clásico. Pierde en tres de cuatro regímenes y sólo empata en el
cuarto.

**Lo que el experimento sí concede al transformer**, y es lo que la tesis puede
defender: en el régimen temporal ningún método clásico estándar pasa del azar.
Correlación parcial 0.478, PC 0.442, información mutua 0.463. El único
competidor es una construcción a la que hubo que decirle a mano qué retardos
mirar. Ésa es una ventaja de aplicabilidad real, medida contra verdad conocida,
y es la que hay que escribir: no "recupera mejor", sino **"recupera donde lo
clásico no llega, sin que se le diga dónde mirar"**.

**Lo que hay que corregir en el documento antes que nada.** `SEQ_LENGTH = 1`
y la tarea enmascarada sin ventanas temporales. Con esa configuración el
modelo del caso de estudio no puede captar ninguna relación temporal, así que
cualquier frase del documento que lo afirme está sin respaldo hasta reentrenar
con los retardos como tokens.

---

## 14. El viñedo con retardos, y el hallazgo sobre el propio estimador

La sección 13 mostró que el transformer sólo compite con lo clásico cuando el
mecanismo pasa por el pasado, y que con `SEQ_LENGTH = 1` no puede verlo. Se
reentrenó el viñedo real metiendo los retardos como tokens: 12 índices × 3
instantes = 36 tokens, 15 semillas, para preguntar si el dato tiene estructura
temporal cruzada que el grafo publicado se estuviera perdiendo.

La respuesta a esa pregunta es no. Pero por el camino apareció algo sobre el
estimador que importa más.

### 14.1 El dato del viñedo casi no tiene estructura temporal cruzada

```
reparto de la dependencia positiva por instante del origen
                        atencion   entrada
  retardo t-0             99.1%     99.3%
  retardo t-1              0.5%      0.4%
  retardo t-2              0.4%      0.3%

  quitando la autocorrelacion del mismo indice
  retardo t-0             99.4%     99.5%

  conteo de aristas sobre 3x, con suelo POR BLOQUE
  t-0                        80        82
  t-1                         3         7
  t-2                         4         7
```

Por masa y por conteo, el presente se lo lleva todo. El grafo de 19 aristas no
se queda corto por no haber mirado el tiempo.

La única arista temporal que aparece en las dos ablaciones y en los dos
retardos es `PSRI <- MARI`, con 11.1x y 12.4x en t-1. Antocianina precediendo
a senescencia en una vid caducifolia es plausible, pero es una arista de 132 y
queda como candidata, no como hallazgo.

**Un fallo propio, corregido:** la primera versión umbralizaba los tres bloques
con el suelo de la matriz 36×36 completa. Los suelos por bloque son
2.02e-4 (t-0), 5.43e-5 (t-1) y 4.11e-5 (t-2): el del presente es tres veces el
global. Usar el global inflaba el conteo del presente. Las cifras de arriba ya
usan suelo por bloque.

**Y una limitación que no se puede quitar:** las fechas de Sentinel-2 no son
equiespaciadas. Mediana de huecos 5 días, percentil 90 igual a 23, máximo 200.
Para las escenas detrás de un hueco grande, "t-1" es otra fase fenológica. Con
retardo mediano de 5 días, además, `X_j(t-1)` aporta poquísimo sobre `X_j(t)`
que ya está visible. **Este diseño puede estar condenado a dar cero por
construcción**, y eso limita lo que la ausencia de estructura temporal
significa. Un diseño que sí lo probaría: ocultar el presente entero y forzar
la predicción desde el pasado.

### 14.2 El grafo publicado no sobrevive al cambio de enmascarado, y no es lo que parece

El modelo con retardos usa `MASCARA_MODO="aleatoria"` con `MASCARA_K = 1`
—ocultar sólo el token objetivo—, mientras el grafo publicado se calculó con
`familia_balanceada` y máscara base de familia. Al comparar salió `rho = 0.063`
contra el grafo publicado, y 7 de 19 aristas sobreviviendo cuando el azar
predice 8.6 (`p` hipergeométrico 0.857). O sea, **ningún parecido**.

Eso admitía dos lecturas: los retardos reordenan el presente, o el
enmascarado. Se corrió el control que las separa: 12 tokens, sin retardos,
mismo enmascarado neutro, mismo entrenamiento.

```
rho(control 12 tokens, publicado)   +0.050   IC95 [-0.122, +0.219]  p 0.5660
rho(36 tokens,         publicado)   +0.063   IC95 [-0.109, +0.231]  p 0.4749
rho(control 12 tokens, 36 tokens)   +0.661   IC95 [+0.552, +0.747]  p 0.0000
```

Concluyente. Los dos modelos de máscara neutra se parecen entre sí (0.661, al
nivel de la reproducibilidad entre semillas que ya se había medido, ICC 0.650)
y ninguno se parece al publicado. **Los retardos no cambian nada; el
enmascarado lo cambia todo.**

### 14.3 Por qué, y es lo publicable de esta sección

La distribución de valores lo explica:

```
                      negativos    mediana       maximo
publicado (familia)     60/132     +6.83e-4    +4.54e-2
36 tokens (k=1)         24/132     +3.96e-4    +4.63e-1
control 12 tok (k=1)     0/132     +2.81e-3    +4.95e-1
```

Bajo `familia_balanceada`, casi la mitad de los cortes **mejoran** la
reconstrucción: la familia del objetivo ya está oculta, así que cortar una
arista más a menudo no quita nada y el ruido de estimación domina. Bajo
`k = 1`, con los once canales disponibles, cortar cualquier arista siempre
duele, y duele diez veces más.

No son la misma magnitud reescalada. Son dos cantidades distintas:

```
familia_balanceada    lo que j aporta MAS ALLA de la familia de v
                      dependencia condicional excluyendo la redundancia
k = 1                 lo que j aporta en total, redundancia incluida
```

No hay razón para que correlacionen, y no correlacionan. El grafo de 19
aristas es válido, pero **es un grafo de dependencia no redundante**, y esa
condición hay que escribirla junto al resultado: no es "el grafo del modelo",
es "el grafo bajo enmascarado por familia".

**Consecuencia para el método, que es lo que trasciende al caso:** el suelo de
ruido se estima de la cola negativa de $D_{att}$, y bajo `k = 1` **no hay cola
negativa** —cero de 132 valores negativos—, así que el suelo sale `nan` y el
criterio de 3x no se puede aplicar. El estimador de ruido del framework
**requiere un esquema de enmascarado que genere cortes con efecto negativo**.
Es un requisito no declarado en ninguna parte del trabajo hasta aquí, y quien
reutilice el método sin saberlo se encuentra con un `nan` o, peor, con un
umbral inventado.

### 14.4 Qué queda escrito

1. El viñedo no tiene estructura temporal cruzada detectable con este diseño,
   con la salvedad de 14.1 sobre huecos y retardo corto. `PSRI <- MARI (t-1)`
   queda como única candidata.
2. El grafo de 19 aristas es válido bajo `familia_balanceada` y no transfiere a
   otros esquemas de enmascarado. La condición pasa a formar parte del
   enunciado del resultado.
3. $D_{att}$ mide cosas distintas según el enmascarado: dependencia no
   redundante con familia, dependencia total con `k = 1`. Elegir el esquema es
   elegir la pregunta, no un detalle de implementación.
4. El suelo de ruido por cola negativa exige un enmascarado que la produzca.
   Bajo `k = 1` el método se queda sin criterio de umbral.

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
