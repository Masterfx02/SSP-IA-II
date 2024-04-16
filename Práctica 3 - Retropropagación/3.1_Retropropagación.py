import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

class MLP:
    def __init__(self, capas):
        # Inicialización de la red neuronal con el número de neuronas en cada capa
        self.capas = capas
        # Inicialización de pesos con valores aleatorios y sesgos a cero
        self.pesos = [np.random.randn(capas[i], capas[i + 1]) for i in range(len(capas) - 1)]
        self.desviacion = [np.zeros((1, capas[i + 1])) for i in range(len(capas) - 1)]

    def sigmoid(self, x):
        # Función de activación sigmoide
        return 1 / (1 + np.exp(-x))

    def derivada_sigmoid(self, x):
        # Derivada de la función sigmoide
        return x * (1 - x)

    def forward(self, X):
        # Propagación hacia adelante
        self.activaciones = [X]
        self.valores_z = []

        for i in range(len(self.capas) - 1):
            # Calcular el valor ponderado y la activación para cada capa
            z = np.dot(self.activaciones[-1], self.pesos[i]) + self.desviacion[i]
            a = self.sigmoid(z)
            # Almacenar los valores para su uso posterior en la retropropagación
            self.valores_z.append(z)
            self.activaciones.append(a)

    def backward(self, X, y, tasa_aprendizaje):
        # Retropropagación para ajustar pesos y sesgos
        errores = [y - self.activaciones[-1]]
        deltas = [errores[-1] * self.derivada_sigmoid(self.activaciones[-1])]

        for i in range(len(self.capas) - 2, 0, -1):
            # Calcular el error y la delta para cada capa oculta
            error = deltas[-1].dot(self.pesos[i].T)
            delta = error * self.derivada_sigmoid(self.activaciones[i])
            errores.append(error)
            deltas.append(delta)

        for i in range(len(self.capas) - 2, -1, -1):
            # Actualizar pesos y sesgos utilizando los errores y deltas calculados
            self.pesos[i] += self.activaciones[i].T.dot(deltas[len(self.capas) - 2 - i]) * tasa_aprendizaje
            self.desviacion[i] += np.sum(deltas[len(self.capas) - 2 - i], axis=0, keepdims=True) * tasa_aprendizaje

    def entrenar(self, X, y, epocas, tasa_aprendizaje):
        # Entrenamiento de la red durante un número específico de épocas
        for epoca in range(epocas):
            # Propagación hacia adelante y retropropagación en cada época
            self.forward(X)
            self.backward(X, y, tasa_aprendizaje)

    def predecir(self, X):
        # Realizar una predicción utilizando la red neuronal entrenada
        self.forward(X)
        return np.round(self.activaciones[-1])

# Cargar el conjunto de datos
datos = pd.read_csv('concentlite.csv')

# Dividir en características (X) y etiquetas (y)
X = datos.iloc[:, :-1].values
y = datos.iloc[:, -1].values

# Dividir en conjunto de entrenamiento y prueba(20%)
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42)

# Define la arquitectura de la red, por ejemplo, [tamaño_entrada, tamaño_oculto, tamaño_salida]
capas = [X.shape[1], 8, 1]

# Inicializa la red
red_neuronal = MLP(capas)

# Entrenar la red con más épocas y una tasa de aprendizaje más alta
red_neuronal.entrenar(X_entrenamiento, y_entrenamiento.reshape(-1, 1), epocas=5000, tasa_aprendizaje=0.2)

# Realizar predicciones en el conjunto de prueba y redondear a 0 o 1
predicciones = np.round(red_neuronal.predecir(X_prueba))

# Visualizar el resultado
plt.scatter(X_prueba[:, 0], X_prueba[:, 1], c=y_prueba, cmap='coolwarm', label='Clase1', alpha=0.7)
plt.scatter(X_prueba[:, 0], X_prueba[:, 1], c=predicciones.flatten(), cmap='coolwarm', marker='x', label='Clase2', linewidth=1)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Clasificación del Perceptrón Multicapa')
plt.legend()
plt.show()

np.random.seed(0)
X = np.random.rand(1000, 2)  # 1000 puntos en 2 dimensiones
y = np.random.randint(0, 2, size=1000)  # Etiquetas binarias aleatorias (0 o 1)

# Visualizar los puntos
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', alpha=0.7)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Inicialización 1000 Puntos Aleatorios')
plt.show()