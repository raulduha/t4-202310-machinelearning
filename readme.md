Reconocimiento de letras manuscritas con Reducción de Dimensionalidad
Este repositorio contiene un proyecto de reconocimiento de letras manuscritas utilizando técnicas de reducción de dimensionalidad, como PCA (Análisis de Componentes Principales) y UMAP (Mapeo Aproximado de Vecinos más Cercanos).

Conjunto de datos
El conjunto de datos utilizado en este proyecto es el EMNIST, que contiene imágenes de letras manuscritas en diferentes estilos y fuentes. El conjunto de datos se divide en conjuntos de entrenamiento y prueba.

Dependencias
Asegúrate de tener instaladas las siguientes bibliotecas para ejecutar el código:

TensorFlow
EMNIST
NumPy
Matplotlib
scikit-learn
Puedes instalar las dependencias ejecutando el siguiente comando:

pip install tensorflow emnist numpy matplotlib scikit-learn

Resultados
Después de ejecutar el código, se mostrarán los siguientes resultados:

Precisión original: Accuracy obtenida utilizando el conjunto de datos sin reducción de dimensionalidad.
Precisión con PCA: Precisión obtenida utilizando PCA con diferentes números de componentes.
Precisión con UMAP: Precisión obtenida utilizando UMAP con diferentes números de componentes.
Además, se generarán gráficas que muestran la precisión obtenida con PCA y UMAP en función del número de componentes utilizados.

Conclusiones
En base a los resultados obtenidos, se mostrará la conclusión del proyecto, indicando el número óptimo de componentes y la precisión correspondiente obtenida tanto con PCA como con UMAP.

Autor: Raul Duhalde
Fecha: 19.06.2023