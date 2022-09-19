# Bitacora: detección de sentimientos utilizando redes neuronales
### Investigadores:
* Luis Carlos Quesada
* Mario Viquez
* Gianfranco Bagnarello
* Isaac Herrera
* Daniel Ramirez
---
 
### Indice:

* [Replicación de la topología base](#replicación-de-la-topología-base-issa-et-al.)
* [Busqueda de los datos de entrada al modelo propuesto](#busqueda-de-los-datos-de-entrada-al-modelo-propuesto)
* [Pruebas del modelo base](#pruebas-del-modelo-base)
* [Reconstrucción del modelo utilizando Pytorch en lugar de Keras](#reconstrucción-del-modelo-utilizando-pytorch-en-lugar-de-keras)
* [Data augmentation (speed)](#data-augmentation-(speed))
* [Data augmentation](#data-augmentation)
* [Pruebas con el modelo replicado](#pruebas-con-el-modelo-replicado)


---

## Replicación de la topología base Issa et al.
### Luis Carlos Quesada - 5/10/2022
Repliqué el modelo de Issa et al. El modelo está descrito en el paper pero describe la capa de entrada como 193 neuronas, anteriormente habíamos hecho extracción de caracteristicas pero el resultado era de más de 228 variables a diferencia de este, por lo que no podemos comprobar de momento que la red neuronal se ajuste a las predicciones descritas en el paper.

### Resultados:
Se obtuvo un modelo con la topología descrita en el paper a replicar, sin embargo no se puede comparar los resultados ya que es necesario investigar la dimensionalidad de los datos de entrada pues no fueron descritos en el modelo original. 

---
<br>
<br>

## Busqueda de los datos de entrada al modelo propuesto
### Luis Carlos Quesada - 5/18/2022
Para poder hacer una clasificación de audio usando la topología de red propuesta en el paper es necesario saber las dimensiones de entrada de los datos, la red tiene 193 variables de entrada y sabemos que hay algunass caracteristicas con tamaños fijos.

Se sabe que el tonnetz es de shape (n, 6) entonces al aplicar el calculo de medias el shape final es de (6,).

Chromagram produce 12 bins default, pero el tamaño de bins puede ser cualquier numero > 0.

El MFCC usualmente tiene entre 20-40ms por cada bin, sabemos que tenemos 3 segundos por audio, entonces podemos mover eta variabe en el rango [75, 120].

Mel Spectrogram de la misma forma depende de la cantidad de mels, que de nuevo es [75, 120], el tamaño de MFCC y spectrogram deben ser iguales.

Contrast se calcula con 6 bandas default pero funciona con n_bands > 1 y retorna n_bands + 1

Así que al final, tenemos algo así

<!-- $$ 
6 + (bands + 1) + bins + 2mel = 193
$$ --> 

<div align="center"><img style="background: white;" src="svg\nBcoiMI68J.svg"></div> 

<!-- $$
mel ∈ [75, 120]
$$ --> 

<div align="center"><img style="background: white;" src="svg\qeqZGbzVgF.svg"></div>
<!-- $$
bands > 0
$$ --> 

<div align="center"><img style="background: white;" src="svg\6FeOJvlnpQ.svg"></div>
<!-- $$
bins > 0
$$ --> 

<div align="center"><img style="background: white;" src="svg\OPBItwIF41.svg"></div>

### Resultados:
Por ahora asumiendo los valores default para n_bands y bins podemos movernos por el numero de mels, esto nos deja con 84 mels que es un valor valido ya que 
<!-- $$ 
75 <= mel <= 120 
$$ --> 

<div align="center"><img style="background: white;" src="svg\zL6jhbHNkE.svg"></div>

---
<br>
<br>

## Pruebas del modelo base
### Luis Carlos Quesada - 5/19/2022
Probé el modelo base con los datos de prueba a ver que pasa, creo que no estan para nada cerca pero podría ser por la forma en que los datos están codificados, estoy casi seguro de eso!
El modelo original fue entrenado usando softmax como salida para 8 variables, esto hace que sea necesaria una codificación un toque diferente a la que tengo actualmente, más cuando el optimizador es RMSprop y una perdida de sparse_entropy_loss, así que casi de fijo es eso

### Resultados
Las predicciones son como del 20% en promedio, es terrible pero creo que se puede arreglas mañana porque haciendo un max de las diferentes salidas, si se activa la salida "correcta" digamos, ahí se revisa un toque en la tarde.

---
<br>
<br>

## Data augmentation (speed)
### Daniel Ramírez Umaña - 5/19/2022
Leyendo el paper noté que habíamos pasado por alto que para el segundo modelo se realizaron modificaciones en el dataset de entrenamoento, acelerando el audio a 1.23%, reducuiendo su velocidad 0.81%, además de manteniendo la velocidad normal, además de a todas estas instancias del audio tener una versión sin ruido y una versión a la que se le agregó un 25% de ruido.

Se trbajó en recorrer el código con los archivos de audio para que cada entrada tuviera una versión en velocidad 1.23% y 0.81%.

### Resultados
Se triplicó el dataset de entrenamiento.

---
<br>
<br>
## Reconstrucción del modelo utilizando Pytorch en lugar de Keras
### Luis Carlos Quesada, Mario Viquez, Daniel Ramirez - 5/19/2022
Ya despues de dormir hice algunas consultas y pareciera que podría ser la función de error, se me recomienda usar una biblioteca a la que esté más familiarizada y a la que se le pueda dar un contról más fino como PyTorch, podría sentarme a hacer un thinkering al error de la red anterior pero el coste de tiempo sería significativamente alto, por lo que mejor voy a crear la red nuevamente usando un modelo de torch, ya que la artquitetura está definida no debería tomar más de una hora.

Durante el día Gatto me dijo que es posible que no todos los datos sean del mismo tamaño, así que me interesa realizar un analisis preeliminar de los datos mas profundo que el anterior, ya que la vez pasada realmente no sabíamos mucho más que las distribuciones de datos y las composiciones de caracteristicas. Me interesa saber no solo los tamaños si no la posibilidad de una separación no supervisada con algun algoritmo tipo K-means simplemente para visualizar los puntos.

El día terminó siendo bastante productivo, hicimos algunas pruebas separando los datos con un PCA, K-means y graficando, salió terrible!
<div align="center"><img style="background: white;" src="Imgs\prueba-pca.png"></div>

Pero luego utilizando el modelo base hicimos un par de cambios: los datos ahora son pasados por un filtro que los escala y normaliza, luego las etiquetas fueron modificadas para hacer la predicción con one hot encoding en lugar de etiquetas ordinales.

### Resultados:
El modelo fue entrenado 700 epocas y alcanzó una precisión de 99%, lastimosamente esto es un obvio overfitting pues al revisar con el dataset de testing solo se alcanza un 46% de accuracy, aun así es mejor que el 20% de ayer.

En la tarde-malana del 21 esperamos aplicar finalmente el entrenamiento descrito en el paper, utilizando cross validation. 

---
<br>
<br>

## Data augmentation
### Luis Carlos Quesada - 5/21/2022
Luego de aplicar los metodos de data augmentation que se mencionaban en el paper llegamos a tener resultados de 99% en entrenamiento y 98% en testing, esto me pareció demasiado sospechoso y pasé casi 6 horas buscando que paso, el data augmentation que se describe en el paper realmente no cambia los datos en la dimensión espectral, unicamente en la dimensión de tiempo, esto deja varia preguntas porque si ninguno de los modelos descritos en el paper utilizan la dimensión de tiempo ¿cual es el sentido de hacer esto?

Al crear copias de datos que tienen una forma espectral tan similar, se dividen los datos en training y testing, los datos en testing probablemente tengan una copia muy similr en el training por la razon mencionada anteriormente, así que las altas predicciones de los datos de testing se dan por tener copias de estos en el training.

Ya quiero terminar la replicación, cada vez que veo el experimento original del paper le encuentro más y más fallas y no me parece que sea un metodo correcto o que si quiera los resultados reportados sean replicables.

### Resultados:
Descubrimos que si se usan caracteristicas espectrales del sonido, una tecnica de data augmentation que haga cambios sobre el sonido puede no ser la mejor idea, tal vez si esto se aplica unicamente a los datos de training y no a los de testing, pero de todas formas hay que considerar si la diferencia de los datos es lo suficientemente significativa.

### Nota: 
Llenen la bitacora cada vez que hacen algo para que no se repita lo de hoy.

---
<br>
<br>

## Data augmentation II
### Isaac Herrera y Daniel Ramírez Umaña - 5/21/2022
Investigando se logró implementar ruido a los audios, justo como en el segundo modelo. Aún no se aumenta el dataset ya que solo debe aplicarse al 25% de cada audio.

### Resultados:


### Nota: 
Llenen la bitacora cada vez que hacen algo para que no se repita lo de hoy.

---
<br>
<br>

## Data augmentation
### Isaac Herrera y Daniel Ramírez Umaña - 5/22/2022
Se implementó que el ruido solo se aplique al 25% de cada audio, y ahora sí se incrementó el dataset.

### Resultados:


### Nota: 
Llenen la bitacora cada vez que hacen algo para que no se repita lo de hoy.

---
<br>
<br>

## Pruebas con el modelo replicado
### Luis Carlos Quesada - 5/22/2022
Hoy nos quedamos sin GPU time, así que de ahora en adelante los experimentos van a ser considerablemente más lentos, tratamos con la GPU de Kaggle pero al parecer hay un error entre el entorno de kaggle y Colab que no nos permite usarlos igual, Kaggle nos dá un error en la dimensionalidad de los datos de entrada y no estamos seguros de por qué, ya que no tenemos control completo del entorno no nos queda más que usar lo que ya conocemos, el costo de tener control sobre el entorno es demasiado caro en tiempo de computación.

El modelo replicado está funcionando, el modelo es tal y como se describe en el paper al menos tan fiel a la descripción dada como se puede. 
Todavía desconocemos la distribución original de dimensiones para cada dato, lo que es bastante alarmante pero no encontramos ninguna fuente que nos confirme las dimensiones originales, simplemente las omitieron haciendo muy difici la repicación de experimento, tal vez si supieramos un poco más de audio antes habríamos previsto algo como esto pero bueno, esto al menos da bastante aprendizaje, por la misma razón no podemos saber que metodos utilizaron para modelar las salidas de datos o función de error.

A partir de lo que sabemos, podemos alcanzar 30%-40% de accuracy lo que no es mucho si lo comparamos con el paper original, pensamos que esto no se debe a la arquitectura del modelo pues ya vimos que algo funciona, pero hay un claro overfitting al realizar 700 epocas, casi desde la epoca 100 se empieza a llegar a una convergencia en el 100% de correctitud, 700 son demasiadas epocas, pero es la catidad de epocas descritas en el paper, la unica forma es que tuvieran muchos más datos de entrenamiento pero RAVDESS no es más grande ni se describe que hayan realizado algun procedimiento de data augmentation sobre RAVDESS, por eso pensamos que el Data Augmentation realizado sobre EMO-DB podría haber sido utilizado tambien en RAVDESS, pero ya vimos los resultados de eso.


### Resultados:
<div align="center"><img style="background: white;" src="Imgs\training_original.png"></div>
Los resultados siempre se quedan cerca del 40%, no parece que se le pueda sacar más a estos datos, no puedo asegurarlo pero he visto que hy algunos experiimentos donde han usado unicamente el MFCC con una SVM y obtienen mejores resultados, por lo que esto es medio shady.


---
<br>
<br>

## Sentan-I
### Luis Carlos Quesada - 5/9/2022
Ya que pasaron cosas graciosas con los modelos originales de Dias y no tenemos contacto, debemos realizar cambios para redirigir la investigación de una replicación a la creación de un modelo.
Curiosamente esto resulta más interesante y se siente menor presión a lograr una replicación exacta al modelo base propuesto por Dias.
El modelo base no deja de ser interesante claro, por lo que planeo trabajar sobre este, modificarlo de forma que creo podría producir resultados interesantes.

### Resultados:
Creación de la arquitectura Sentan-I, esta arquitectura aún no pasa por las pruebas metricas para ser evaluada pero al menos "funciona", será un buen base line para empezar a obtener resultados.


---
<br>
<br>

## Sentan - Cantidad de epocas
### Luis Carlos Quesada - 5/17/2022
Durante el desarrollo del modelo base, al realizar un entrenamiento y graficarlo se nota un comportamiento interesante.
<div align="center"><img style="background: white;" src="Imgs/Accuracy_700epochs_basemodel.png"></div>

De forma similar al entrenamiento anterior, podemos ver que aproximadamente por la epoca 100 ~ 200 se alcanza una "Perfección" en la clasificación de los datos de entrenamiento, además de esto se nota una convergencia para los valores de validación.

Hay una posibilidad de reducir la cantidad de epocas y no afectar demasiado el rendimiento, utilizando nuevamente K-folding y una reducción de epocas podremos ver que pasa.

### Resultados:
Como podemos ver en la siguiente imagen, al reducir las epocas a unicamente 200 se alcanza nuevamente la convergencia de entrenamiento y la validación nuevamente se acerca a 0.6, por lo que reducir la cantidad de epocas durante el entrenamiento podría ser en realidad una forma bastante practica de reducir el tiempo de entrenamiento necesario.
<div align="center"><img style="background: white;" src="Imgs/Accuracy_200epochs_basemodel.png"></div>


---
<br>
<br>


## Sentan Avance - Dropout
### Luis Carlos Quesada - 5/18/2022
Tomando algunas ideas de conocimiento previo como base, tenemos la idea de que el modelo base de Dias presenta un pequeño problema en la ubicación de las capas de dropout, usualmente estas capas no se encuentran en la sección convolucional si no en la sección de clasificación, pero en la arquitectura de Dias se colocan desde las capas convolucionales, por lo que se plantea eliminar estas capas de donde se encuentran, realizar una medición de accuracy y luego colocarlas en las capas de clasificación y nuevamente realizar mediciones.

### Resultados:
En la medición inicial que se realizó luego de eliminar las capas de dropout podemos notar que no hay un aumento significativo de accuracy, más bien al contrario se reduce el accuracy del modelo y además presenta un efecto etra interesante, podemos ver como el modelo comienza a converger al 100% en el entrenamiento mucho antes, es decir, las capaas ayudan a prevenir el overfitting a pesar de no estar colocadas en donde uno pensaría que serían más efectivas.
<div align="center"><img style="background: white;" src="Imgs/Accuracy_200epoch_no_dropout.png"></div>


Realiando pruebas con distintas configuraciónes para las capas de salida pareciera que sería posible añadir algunas capas de dropout si la clasificación no se deja en una unica capa, al dividir la clasificación en 2 capas se logró obtener la misma puntuación en accuracy que el modelo original, pero con 2 capas menos de dropout, actualmente el accuracy se encuentra aproximadamente en un 60%, siin embargo sigue existiendo el problema de convergencia en training muy rapida.


<div align="center"><img style="background: white;" src="Imgs/Accuracy_200epoch_class_dropout.png"></div>


---
<br>
<br>


## Sentan Avance Regularizador 
### Luis Carlos Quesada - 5/20/2022:

Para combatir el problema de overfitting, ya que es el problema más significativo que presenta el modelo, se decide optar por la introducción de un regularizador.
Para el modelo se utiliza un optimizador Adam y a este se le añade un regularizador L2 que debería ayudar a combatir el overitting

### Resultados:

Los resultados de aplicar una regularización no parecen mejorar las metricas del modelo, luego de probar diferentes combinaciones de hiperparametros se tiene la idea de que el modelo base es simplemente demasiado complejo para realizar predicciones, es posible que modificando la arquitectura a una forma más sencilla pueda reducir el overfitting de manera significativa.

<div align="center"><img style="background: white;" src="Prototypes/Sentan-I/Plots/Acc_700_sentan_regular.png"></div>


---

## Sentan Avance Simplificación de la arquitectura
### Luis Carlos Quesada - 5/22/2022:

Debido a que los anteriores intentos por obtener mejores resultados no parecen llegar a nada se opta por simplificar la arquitectura del modelo base completamente, se mantendrá la estructura de los datos por lo que solo será necesario modificar el modelo.

Considerando que ya se tiene experiencia con datos más complejos trabajados sobre redes más simples, se piensa utilizar un acercaiento a la reducción de la dimensionalidad de los datos de una manera mas sencilla reduciendo la cantidad de capas convolucionales necesarias.


### Resultados:

Inicialmente se creó un modelo pequeño, aunque no muy efectivo de una unica capa convolucional y una capa de clasificación
<div align="center"><img style="background: white;" src="Prototypes/Sentan-I/Plots/Minimizacion_1_layer.png"></div>


Al añadir una capa extra se obtuvieron mejores resultados llegando a casi un 70% de accuracy en promedio.
<div align="center"><img style="background: white;" src="Prototypes/Sentan-I/Plots/Minimización 2 capas.png"></div>


Finalmente se intentó añadir capas de dropout para contrarrestar aun más al overfitting, sn embargo esto no mejoró la eficiencia del modelo
<div align="center"><img style="background: white;" src="Prototypes/Sentan-I/Plots/Minimizacion2 + dropout 400e.png"></div>


---
## Elaboración del documento del Avance #2: 04 Julio 2022
### Luis Carlos Quesada
### Gianfranco Bagnarello
### Mario Viquez
### Isaac Herrera
### Daniel Ramirez
El equipo trabajó en conjunto para elaborar el documento del avance 2, se escribieron las bases y se organizó el documento. Se escribieron borradores de la mayoría de las secciones. Surgió una duda que se preparó para enviarla a los profesores (la duda es de las preguntas de investigación con respecto a la evolución del trabajo).

## Elaboración del documento del Avance #2: 05 Julio 2022
### Luis Carlos Quesada
### Gianfranco Bagnarello
### Mario Viquez
### Isaac Herrera
### Daniel Ramirez
El equipo trabajó en conjunto para continuar con la elaboración del documento del avance #2. Se refinaron los objetivos, los antecedentes, el diseño experimental, las hipótesis, las variables dependientes e independientes. Además se envió la duda al profesor que nos respondió, por lo que aplicamos los cambios sugeridos y se actualizó la investigación. Surgió otra duda que se preparó para enviarla mañana al profesor, que es sobre el enfoque de los edificios inteligentes. Se refinó bastante el documento en tema de redacción y ortografía.


