# PARCIAL 1 - HIGH PERFORMANCE COMPUTING

Se realizó la implementación en ```CUDA c``` para la multiplicación de matrices  de diferentes dimensiones, las dimensiones con las que trabajamos fueron para la matriz  ```A(MxN) y B(NxY) ```, se realizó un algoritmo que realizaba la multiplicación de matrices en el host, es decir de forma secuencial y se implementó otro que realiza la multiplicación de matrices aprovechando el paralelismo que nos proporcionan las GPU usando memoria compartida haciendo uso del concepto de TILES, posteriormente se tomaron tiempos sobre diferentes dimensiones de las matrices y de los TILES sacando promedios para obtener una medida más exacta y así comprobar el rendimiento de ambos algoritmos junto con su factor de aceleración. A continuacion las tablas con su respectiva informacion


## Tiempos de la primera multiplicacion de matrices


### Dimensiones

|  MATRIZ A | MATRIZ B  | TILES | BLOCKSIZE |
|-----------|-----------|-------|-----------|
|m=20  n=20 | n=20 y=25 |   4   |     4     |  
### tiempo en segundos

|   CPU   |    GPU    |
|---------|-----------|
|0,000065 |	0,000067  |
|0,000065 |	0,000065  |
|0,000031 |	0,000066  |
|0,000065 |	0,000066  |
|0,000065 |	0,000067  |
|0,00003  |  0,000065 |
|0,000028 |	0,000066  |
|0,000028 |	0,000066  |
|0,000065 |	0,000068  |
|0,000028 |	0,000067  |


### promedio

|CPU     | GPU       |
|--------| ----------|
|0,000047|	0,0000663|

## Tiempos de la segunda multiplicacion de matrices

### Dimensiones

|  MATRIZ A   |   MATRIZ B  | TILES | BLOCKSIZE|
|-------------|-------------|-------|----------|
|m=100  n=100 | n=100 y=120 | 8     | 8        |

### tiempo en segundos

|   CPU   |   GPU   |
|---------|---------|
|0,0032   |	0,000124|
|0,003239 |	0,000131|
|0,004021 |	0,000122|
|0,003585 |	0,000127|
|0,003249 |	0,000126|
|0,003241 |	0,000125|
|0,007284 |	0,000124|
|0,007591 |	0,000124|
|0,00406  |	0,000124|
|0,007605 |	0,000125|

### promedio
|   CPU    |    GPU    |
|--------- |-----------|
|0,0047075 |	0,0001252|


## Tiempos de la tercera multiplicacion de matrices

### Dimensiones

|MATRIZ A     | MATRIZ B    | TILES | BLOCKSIZE |
|-------------|-------------|-------|-----------|
|m=300  n=300 | n=300 y=290 |   16  |    16     |

### tiempo en segundos

|   CPU   |    GPU  |
|---------|---------|
|0,074824 |	0,000583|
|0,076225 |	0,000586|
|0,073705 |	0,000585|
|0,073485 |	0,00058 |
|0,071331 |	0,000581|
|0,07158  |	0,000584|
|0,080298 |	0,000605|
|0,075655 |	0,000587|
|0,077322 |	0,000584|
|0,077331 |	0,000582|

### promedio

|   CPU    |     GPU   |
|----------|-----------|
|0,0751756 |	0,0005857|

## Tiempos de la cuarta multiplicacion de matrices

### Dimensiones

|  MATRIZ A   |   MATRIZ B  | TILES | BLOCKSIZE|
|-------------|-------------|-------|----------|
|m=800  n=800 | n=800 y=900 |   32  |    32    |

### tiempo en segundos

|  CPU    |    GPU  |
|---------|---------|
|1,67595  |	0,005672|
|2,55008  |	0,0054  |
|1,702742 |	0,005414|
|2,552726 |	0,005755|
|1,672767 |	0,005912|
|1,675532 |	0,003065|
|2,562902 |	0,005293|
|2,565965 |	0,005278|
|1,721134 |	0,006358|
|1,69067  |	0,005297|

### promedio
|   CPU    |    GPU    |
|----------|-----------|
|2,0370468 |	0,0053444|


## Tiempos de la quinta multiplicacion de matrices

### Dimensiones

|    MATRIZ A   |    MATRIZ B   | TILES | BLOCKSIZE|
|---------------|---------------|-------|----------|
|m=1600  n=1600 | n=1600 y=1200 |   32  |   32     |

### tiempo en segundos

|   CPU    |    GPU   |
|----------|----------|
|12,821206 |	0,022904|
|15,800835 |	0,023165|
|13,052345 |	0,022837|
|12,599202 |	0,023017|
|13,383396 |	0,023274|
|12,605863 |	0,023114|
|16,220326 |  0,023223|
|12,603338 |	0,023033|
|14,314566 |	0,022895|
|14,527891 |	0,023025|

### promedio
|    CPU    |    GPU     |
|-----------|------------|
|13,7928968 |	0,0230487  |

## Comparación de tiempos y calculo de aceleración.

|N° prueba	| tiempo  CPU |	 tiempo  GPU |	Aceleracion |
|-----------|-------------|--------------|--------------|
|     1     |	0,000047    |	0,0000663	   |0,708898944   |
|     2     |	0,0047075	  |0,0001252	   |37,59984026   |
|     3     |	0,0751756 	|0,0005857	   |128,3517159   |
|     4     |	2,0370468	  |0,0053444	   |381,1553776   |
|     5     |	13,7928968 	|0,0230487	   |598,4240673   |

## Grafica de tiempos

![alt tag](/img/tiempos.jpg)
### Grafica Aceleracion
![alt tag](/img/aceleracion.jpg)


## Conclusiones

* Mientras que las dimensiones de las matrices sean pequeñas es mejor realizar la multiplicacion de matrices en la CPU, ya que el costo de enviar datos a la GPU es alto, pero cuando aumenta el tamaño considerablemente es mucho mejor usar GPU
* Por lo anterior vemos entonces que La transferencia de datos a traves del PCI Express representa la mayor parte del consumo de tiempo en la implementacion paralela con GPU.
*
