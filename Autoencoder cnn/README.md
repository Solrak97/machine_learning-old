# Auto Encoders: Reducción de dimensionalidad
## CI-0163 Análisis de grandes volúmenes de datos
## Universidad de Costa Rica

## Equipo Celtian

### Estudiantes

- Jose David Ureña Torres B88044

- Eduardo Ayales B90833

- Luis Quesada B65580


### Archivos del repositorio

- Autoencoder.pt: modelo entrenado del autoencoder para el dataset seleccionado.
- README.md: archivo README del proyecto.
- autoencoder_cat_dog_ipynb: notebook del autoencoder entrenado.
- dogs-cats-images: imágenes usadas para el entrenamiento del autoencoder.
- .gitignore: archivo de configuración para el repositorio.
- requirements.pp: archivo de requerimientos para el entrenamiento del autoencoder.
- presentacion-celtian-investigacion-entrega3.pdf: documento de presentación del proyecto.

### Sobre este repositorio

En este repositorio se explora los autoencoders para la reducción de dimensionalidad de imagenes, con una arquitectura basada en redes neuronales convolusionales para la clasificación de gatos y perros a partir de imágenes.

El funcionamiento de un autoencoder para la reducción de dimensionalidad es simple, se crean múltiples capas ocultas cada una más pequeña que la anterior hasta llegar a un umbral en el medio que tiene el tamaño mínimo de características deseadas, nuevamente se añaden capas hasta alcanzar el tamaño inicial, esto causa que los datos primero se “encojan” o codifiquen hasta la llegada del umbral y luego sean reconstruidos a partir de las características más importantes que le constituyen.

El entrenamiento del autoencoder se realiza utilizando back propagation sobre los pesos de cada capa, estos pesos se ajustan a partir de una función de error que calcula que tan diferente es la entrada de datos original con la salida de la misma, ya que se busca maximizar la similitud de estas con la menor cantidad de información posible.


### Como iniciar el ambiente virtual
Las dependencias necesarias para trabajar con el notebook se encuentran en el archivo requirements.pp, facilmente se puede crear un ambiente virtual utilizando la herramienta _venv_ y posteriormente instalar las dependencias utilizando _pip_ de la forma siguiente:

* $_pip install -r requirements.pp_

### Dataset utilizado
El dataset utilizado se encuentra en [Dogs & Cats Images](https://www.kaggle.com/datasets/chetankv/dogs-cats-images)
