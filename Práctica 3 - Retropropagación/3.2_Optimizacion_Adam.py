import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

class MLP:
    def __init__(self, capas):
        # Inicialización de la red neuronal con capas y parámetros de optimización Adam
        self.capas = capas
        self.pesos = [np.random.randn(capas[i], capas[i + 1]) for i in range(len(capas) - 1)]
        self.bias = [np.zeros((1, capas[i + 1])) for i in range(len(capas) - 1)]
        self.m = [np.zeros_like(w) for w in self.pesos]
        self.v = [np.zeros_like(w) for w in self.pesos]
        self.beta1 = 0.9  # Parámetro de decaimiento para el momento
        self.beta2 = 0.999  # Parámetro de decaimiento para la actualización de la media cuadrática

    def sigmoid(self, x):
        # Función de activación sigmoidal
        return 1 / (1 + np.exp(-x))

    def derivada_sigmoid(self, x):
        # Derivada de la función sigmoidal
        return x * (1 - x)

    def forward(self, X):
        # Propagación hacia adelante (forward pass)
        self.activaciones = [X]
        self.valores_z = []
        for i in range(len(self.capas) - 1):
            # Calcular el valor ponderado y la activación para cada capa
            z = np.dot(self.activaciones[-1], self.pesos[i]) + self.bias[i]
            a = self.sigmoid(z)
            self.valores_z.append(z)
            self.activaciones.append(a)

    def backward(self, X, y, tasa_aprendizaje):
        # Propagación hacia atrás (backward pass) y actualización de parámetros
        errores = [y - self.activaciones[-1]]
        deltas = [errores[-1] * self.derivada_sigmoid(self.activaciones[-1])]

        for i in range(len(self.capas) - 2, 0, -1):
            # Cálculo de errores y deltas en capas intermedias
            error = deltas[-1].dot(self.pesos[i].T)
            delta = error * self.derivada_sigmoid(self.activaciones[i])
            errores.append(error)
            deltas.append(delta)

        for i in range(len(self.capas) - 2, -1, -1):
            # Cálculo del gradiente y actualización de pesos y sesgos utilizando Adam
            gradiente = self.activaciones[i].T.dot(deltas[len(self.capas) - 2 - i])
            
            # Actualización de Adam
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * gradiente
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * gradiente**2
            
            m_hat = self.m[i] / (1 - self.beta1**(i + 1))
            v_hat = self.v[i] / (1 - self.beta2**(i + 1))
            
            self.pesos[i] += tasa_aprendizaje * m_hat / (np.sqrt(v_hat) + 1e-8)
            self.bias[i] += tasa_aprendizaje * np.sum(deltas[len(self.capas) - 2 - i], axis=0, keepdims=True)

    def entrenar(self, X, y, epocas, tasa_aprendizaje):
        # Entrenamiento de la red neuronal a lo largo de las épocas especificadas
        for epoca in range(epocas):
            # Propagación hacia adelante y hacia atrás en cada época
            self.forward(X)
            self.backward(X, y, tasa_aprendizaje)

    def predecir(self, X):
        # Predicción de la red neuronal
        self.forward(X)
        return np.round(self.activaciones[-1])

# Cargar el conjunto de datos
datos = pd.read_csv('concentlite.csv')

# Dividir en características (X) y etiquetas (y)
X = datos.iloc[:, :-1].values
y = datos.iloc[:, -1].values

# Dividir en conjunto de entrenamiento y prueba
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42)

# Define la arquitectura de la red para la nueva instancia con Adam
capas = [X.shape[1], 8, 1]
mlp_adam = MLP(capas)

# Entrenar la red con Adam
mlp_adam.entrenar(X_entrenamiento, y_entrenamiento.reshape(-1, 1), epocas=5000, tasa_aprendizaje=0.01)

# Realizar predicciones en el conjunto de prueba y redondear a 0 o 1
predicciones_adam = np.round(mlp_adam.predecir(X_prueba))

# Visualizar el resultado con Adam
plt.scatter(X_prueba[:, 0], X_prueba[:, 1], c=y_prueba, cmap='coolwarm', alpha=0.7)
plt.scatter(X_prueba[:, 0], X_prueba[:, 1], c=predicciones_adam.flatten(), cmap='coolwarm')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Clasificación del Perceptrón Multicapa con Adam')
plt.legend()
plt.show()

np.random.seed(42)
X = np.random.rand(1000, 2)  # 1000 puntos en 2 dimensiones
y = np.random.randint(0, 2, size=1000)  # Etiquetas binarias aleatorias (0 o 1)

# Visualizar los puntos
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', alpha=0.7)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Inicialización 1000 Puntos Aleatorios')
plt.show()