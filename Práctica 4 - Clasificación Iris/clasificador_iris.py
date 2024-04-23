import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

lko_accuracy = []
loo_accuracy = []

class MLP:
    def __init__(self, layers):
        # Inicialización de la red neuronal con el número de neuronas en cada capa
        self.layers = layers
        # Inicialización de pesos con valores aleatorios y sesgos a cero
        self.weights = [np.random.randn(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        self.bias = [np.zeros((1, layers[i + 1])) for i in range(len(layers) - 1)]

    def sigmoid(self, x):
        # Función de activación sigmoide
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivative(self, x):
        # Derivada de la función sigmoide
        return x * (1 - x)

    def forward(self, X):
         # Propagación hacia adelante
        self.activations = [X]
        self.z_values = []

        for i in range(len(self.layers) - 1):
            # Calcular el valor ponderado y la activación para cada capa
            z = np.dot(self.activations[-1], self.weights[i]) + self.bias[i]
            # Seleccionar la función de activación adecuada
            if i == len(self.layers) - 2:
                # Si es la última capa, aplicar la función softmax para la clasificación multiclase
                a = self.softmax(z)
            else:
                # Para capas anteriores, aplicar la función sigmoid para la no linealidad
                a = self.sigmoid(z)
            # Almacenar los valores para su uso posterior en la retropropagación
            self.z_values.append(z)
            self.activations.append(a)

    def softmax(self, x):
        # Resta el valor máximo de cada fila para mejorar la estabilidad numérica
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        # Calcula la exponencial de cada elemento y normaliza por la suma de las exponenciales
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def backward(self, X, y, learning_rate):
        # Retropropagación para ajustar pesos y sesgos
        errors = [y - self.activations[-1]]
        deltas = [errors[-1]]

        for i in range(len(self.layers) - 2, 0, -1):
            # Calcular el error y la delta para cada capa oculta
            error = deltas[-1].dot(self.weights[i].T)
            delta = error * self.sigmoid_derivative(self.activations[i])
            errors.append(error)
            deltas.append(delta)

        for i in range(len(self.layers) - 2, -1, -1):
             # Actualizar pesos y sesgos utilizando los errores y deltas calculados
            self.weights[i] += self.activations[i].T.dot(deltas[len(self.layers) - 2 - i]) * learning_rate
            self.bias[i] += np.sum(deltas[len(self.layers) - 2 - i], axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        # Entrenamiento de la red durante un número específico de épocas
        for epoch in range(epochs):
            # Propagación hacia adelante y retropropagación en cada época
            self.forward(X)
            self.backward(X, y, learning_rate)

    def predict(self, X):
        # Realizar una predicción utilizando la red neuronal entrenada
        self.forward(X)
        return self.activations[-1]

    def evaluate_loo(self, X, y):
        # Evaluación utilizando Leave-One-Out (LOO)
        loo = LeaveOneOut()
        accuracies = []

        # Iterar sobre los conjuntos de entrenamiento y prueba generados por LOO
        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # Entrenar el modelo en el conjunto de entrenamiento
            self.train(X_train, y_train, epochs=1000, learning_rate=0.2)
            # Realizar predicciones en el conjunto de prueba y convertir de one-hot a clase única
            y_pred_onehot = self.predict(X_test)
            y_pred = np.argmax(y_pred_onehot, axis=1)  # Convertir de one-hot a clase única
            y_true = np.argmax(y_test, axis=1)  # Convertir de one-hot a clase única
            # Calcular la precisión y almacenarla
            accuracy = accuracy_score(y_true, y_pred)
            accuracies.append(accuracy)
        # Calcular la media y la desviación estándar de las precisiciones obtenidas
        return np.mean(accuracies), np.std(accuracies)

    def evaluate_lko(self, X, y, k):
        # Evaluación utilizando Leave-k-Out (LKO)
        lko = KFold(n_splits=5)
        accuracies = []
        # Iterar sobre los conjuntos de entrenamiento y prueba generados por LKO
        for train_index, test_index in lko.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # Entrenar el modelo en el conjunto de entrenamiento
            self.train(X_train, y_train, epochs=1000, learning_rate=0.2)
            y_pred_onehot = self.predict(X_test)
            y_pred = np.argmax(y_pred_onehot, axis=1)  # Convertir de one-hot a clase única
            y_true = np.argmax(y_test, axis=1)  # Convertir de one-hot a clase única
            accuracy = accuracy_score(y_true, y_pred)
            accuracies.append(accuracy)
        # Calcular la media y la desviación estándar de las precisiciones obtenidas
        return np.mean(accuracies), np.std(accuracies)


# Cargar el conjunto de datos
data = pd.read_csv('irisbin.csv', header=None)

# Dividir en características (X) y etiquetas (y) 4 entradas, 3 salidas
X = data.iloc[:, :-3].values
y = data.iloc[:, -3:].values

# Crear un objeto StandardScaler
scaler = StandardScaler()
# Normalizar características
X = scaler.fit_transform(X)

# Se define la arquitectura de la red [input_size, hidden_size, output_size]
layers = [X.shape[1], 8, 3]

# Inicializa la red
mlp = MLP(layers)

# Dividir en conjunto de entrenamiento y prueba (80% y 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar la red con más épocas y una tasa de aprendizaje más alta
mlp.train(X_train, y_train, epochs=1000, learning_rate=0.2)

# Realizar predicciones en el conjunto de prueba y redondear a 0 o 1
predictions = mlp.predict(X_test)

# Calcular el error esperado, promedio y desviación estándar
lko_avg_accuracy, lko_std_deviation = mlp.evaluate_lko(X, y, k=5)
loo_avg_accuracy, loo_std_deviation = mlp.evaluate_loo(X, y)
lko_error = 1 - lko_avg_accuracy
loo_error = 1 - loo_avg_accuracy

# Imprimir resultados
print("leave-k-out")
print("Error Esperado:", lko_error)
print("Promedio:", lko_avg_accuracy)
print("Desviación Estándar:", lko_std_deviation)
print("leave-one-out")
print("Error Esperado:", loo_error)
print("Promedio:", loo_avg_accuracy)
print("Desviación Estándar:", loo_std_deviation)


# Imprimir clasificación de los datos de prueba
print("Predicciones y Especies Reales:")
for i in range(len(predictions)):
    species_real = None
    # Determinar la especie real basada en la codificación one-hot (representaciones binarias)
    if y_test[i][2] == 1: #[-1, -1, 1]
        species_real = 'Setosa'
    elif y_test[i][1] == 1: #[-1, 1, -1]
        species_real = 'Versicolor'
    elif y_test[i][0] == 1: #[1, -1, -1]
        species_real = 'Virginica'
    
    species_pred = None
    # Determinar la especie predicha basada en la clase con mayor probabilidad
    if np.argmax(predictions[i]) == 2:
        species_pred = 'Setosa'
    elif np.argmax(predictions[i]) == 1:
        species_pred = 'Versicolor'
    elif np.argmax(predictions[i]) == 0:
        species_pred = 'Virginica'

    print(f"{i+1}: Predicción={species_pred}, Especie real={species_real}")


# Visualizar el resultado
# Pintar puntos para Setosa en rojo
plt.scatter(X_test[y_test[:, 0] == 1, 0], X_test[y_test[:, 0] == 1, 1], color='purple', label='Setosa', alpha=0.7)
# Pintar puntos para Versicolor en verde
plt.scatter(X_test[y_test[:, 1] == 1, 0], X_test[y_test[:, 1] == 1, 1], color='blue', label='Versicolor', alpha=0.7)
# Pintar puntos para Virginica en azul
plt.scatter(X_test[y_test[:, 2] == 1, 0], X_test[y_test[:, 2] == 1, 1], color='pink', label='Virginica', alpha=0.7)

plt.xlabel('Dimensión Petalo')
plt.ylabel('Dimensión Sepalo')
plt.title('Clasificación con MLP: Especies de Iris')
plt.legend(loc='lower right', bbox_transform=plt.gcf().transFigure)
plt.show()
