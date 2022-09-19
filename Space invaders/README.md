# space-invaders
### Universidad de Costa Rica
### Escuela de Ciencias de la Computación e Informática
### Curso: Aprendizaje Mecánico - CI-0148
### Proyecto Final - Space Invaders
### Estudiantes:
 - Luis Carlos Quesada Rodríguez
 - Gianfranco Bagnarello Hernández 
# 
### Descripción del proyecto:
"Space Invaders" consiste en un juego donde se debe luchar contra naves espaciales y destruir la cantidad máxima posible. Para ganar se deben destruir todas las naves. En el variante implementado, las naves llegan en oleadas de tamaño fijo. El jugador debe disparar proyectiles e intentar destruir la mayor cantidad posible antes de que alguna llegue al límite inferior de la pantalla, caso en el que se pierde el juego y se debe volver a empezar. En esta implementación se utilizó la biblioteca PyGame para la interfaz gráfica. Algunos ejemplos de la interfaz de este juego a través de los años son los siguientes:
![Alt Text](/Results/old1.jpg)
![Alt Text](/Results/old2.png)
![Alt Text](/Results/old3.jpg)

La interfaz gráfica El tipo de algoritmo de aprendizaje mecánico implementado fue Deep Q learning, que utiliza redes neuronales, en este caso una de predicciones y otra de objetivo, para explorar el juego e intentar aproximar un modelo que permita maximizar una recompensa obtenida. En este caso, al agente se le otorga una recompensa cada vez que destruye una nave invasora, y se le penaliza cuando no toma ninguna acción. El agente toma como entrada una imagen de 640x480 que sirve como estado del ambiente, y la salida del agente es el valor Q de tomar una acción. Las acciones disponibles para el agente son: moverse a la izquierda, moverse a la derecha y disparar. El agente interactúa con el ambiente tomando la acción obtenida en su salida. 

Durante el proceso de desarrollo se encontraron varios desafíos:

- En la fase inicial de pruebas, el agente se movía muy lento, y cada iteración del algoritmo tomaba exactamente 3.7 minutos. Se dejó entrenando por alrededor de 11 horas. El modelo generado (guardado y adjunto como model.pth) no trabajaba de la forma deseada, ya que las acciones que tomaba lo llebavan a quedarse en el borde izquierdo o derecho de la pantalla sin disparar, ejecutando constamente la acción de moverse a la derecha o izquierda, de acuerdo del lado donde se quedaba.
Otro factor que tenía es que el modelo tomaba una cantidad de tiempo considerable para entrenar por lo que se tomaron medidas como acelerar la velocidad de aparición de las oleadas, y reducirles la complejidad un poco removiendo los patrones de movimiento. En las siguientes figuras se puede observar el tamaño de las oleadas del primer modelo y algunos de estos problemas:

- Dispara inicialmente en medio, se mueve al borde izquierdo, luego dispara, pero no se mueve y se queda ahí sin derribar ninguna nave hasta perder:
![Alt Text](/Results/desafio1.png)
![Alt Text](/Results/desafio2.png)
- Dispara inicialmente, derriba algunas naves, y luego se va para la derecha y se queda ahí hasta perder.
![Alt Text](/Results/desafio3.png)
- Modelo donde se incrementó el tamaño de las oleadas para que obtener recompensas fuera más fácil. Este modelo (model2.pth) presentó el mismo problema que el anterior, donde el agente no aprendía y se quedaba en una esquina hasta perder. Se realizaron algunos cambios como cambiar la velocidad de los proyectiles, y con esto el modelo 2 aprendió a quedarse en medio disparando fijamente, lo que le dejaba una recompensa constante. Luego de eliminar algunos del medio, se movía un poco a los lados mientras disparaba.
![Alt Text](/Results/desafio4.png)

- Otro desafío presentado en la fase de pruebas fue que al agente le costaba encontrar recompensas ya que la mayoría de los enemigos se encontraban en el centro de la pantalla. Esto se mitigó un poco con el nuevo modelo, donde se ejecutaron alrededor de 450 iteraciones de entrenamiento, cada una de alrededor de 1 minuto, ya que se aceleraron los enemigos. Además, se revisó el código y se encontró un error, ya que la función donde se comparaba el valor aleatorio con el valor epsilon_greedy no estaba. Esto se corrigió lo que llegó a dar mejores resultados:

![Alt Text](/Results/demo.gif)


#

Nota: el código para pruebas se encuentra en el archivo test_code.py en el directorio /Game/.
Por defecto corre 10 iteraciones del juego.

### Clases implementadas:
- Actions: contiene las acciones posibles que pueden ser tomadas en el juego.
- Agent: clase que contiene la red neuronal de DeepQlearning del agente, los hiperparámetros y el valor semilla para que el experimento sea repetible..
- Bullet: maneja detalles del los proyectiles disparados.
- Enemy Manager: manejador de oleadas, movimiento de enemigos y contadores de puntuación la recompensa del agente.
- Enemy: contiene propiedades de los enemigos individualmente, los tipos de enemigos, y sus interacciones con el ambiente.
- Simulation manager: controla el juego importando clases y corriendo el agente.
- Game: Importa clases de entidades para dibujar el juego y correr el game loop.
- Player: clase que controla el movimiento e interacciones de la nave del agente con el ambiente.
- Test code: código de pruebas que permite probar los modelos generados. Por defecto tiene el modelo 3 (el más entrenado y trabajado).

Bibliotecas requeridas:
- Se encuentran en el archivo requirements.txt adjunto.