import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Se define la función del perceptrón
def perceptron(inputs, weights, bias):
    summation = np.dot(inputs, weights) + bias
    # Función de activación
    if summation >= 0:
        return 1
    else:
        return 0

# Leer archivos de datos perturbados
dataset_file = 'spheres2d70.csv'
dataset_name = 'Dataset spheres2d70'
data = pd.read_csv(dataset_file)
inputs = data.iloc[:, :-1].values
outputs = data.iloc[:, -1].values

# Parámetros para entrenamiento
max_epochs = 100
learning_rate = 0.1
convergence_criterion = 0.01  # Alteraciones aleatorias < 5%

# Inicializar validación cruzada estratificada con 10 particiones
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Variables para almacenar la precisión promedio
average_accuracy = 0.0

# Dividir los datos en entrenamiento (80%) y prueba (20%)
from sklearn.model_selection import train_test_split
train_data, test_data, train_labels, test_labels = train_test_split(inputs, outputs, test_size=0.2, random_state=42)


# Iterar sobre las particiones
for fold_idx, (train_indices, test_indices) in enumerate(skf.split(inputs, outputs)):
    train_data, test_data = inputs[train_indices], inputs[test_indices]
    train_labels, test_labels = outputs[train_indices], outputs[test_indices]

    # Entrenamiento del perceptrón
    def train_perceptron(inputs, outputs, learning_rate, max_epochs, convergence_criterion):
        num_inputs = inputs.shape[1]
        num_patterns = inputs.shape[0]

        weights = np.random.rand(num_inputs)
        bias = np.random.rand()
        epochs = 0
        convergence = False

        while epochs < max_epochs and not convergence:
            convergence = True
            for i in range(num_patterns):
                inputt = inputs[i]
                output_prediction = outputs[i]
                output_received = np.dot(weights, inputt) + bias
                error = output_prediction - output_received

                if abs(error) > convergence_criterion:
                    convergence = False
                    weights += learning_rate * error * inputt
                    bias += learning_rate * error
            epochs += 1
        return weights, bias

    trained_weights, trained_bias = train_perceptron(train_data, train_labels, learning_rate, max_epochs, convergence_criterion)
    print(f"Perceptrón entrenado con éxito para {dataset_name} - Partición {fold_idx + 1}.")

    # Probar el perceptrón entrenado en datos de prueba
    def test_perceptron(inputs, weights, bias):
        output_received = np.dot(inputs, weights) + bias
        return np.where(output_received >= 0, 1, 0)

    test_predictions = test_perceptron(test_data, trained_weights, trained_bias)

    # Calcular la precisión
    accuracy = accuracy_score(test_labels, test_predictions)
    print(f"Precisión del perceptrón en datos de prueba (Accuracy) para {dataset_name} - Partición {fold_idx + 1}: {accuracy * 100:.2f}%")

    # Agregar la precisión de esta partición a la precisión promedio
    average_accuracy += accuracy

# Calcular la precisión promedio
average_accuracy /= 10  # Dividir por el número de particiones (k-fold)

print(f"Precisión promedio del perceptrón en {dataset_name}: {average_accuracy * 100:.2f}%")

# Visualización en 3D
def graphic_3d(inputs, outputs, weights, bias, title):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Graficar patrones
    ax.scatter(inputs[:, 0], inputs[:, 1], inputs[:, 2], c=outputs, s=100)

    # Crear plano de separación
    x_min, x_max = inputs[:, 0].min() - 1, inputs[:, 0].max() + 1
    y_min, y_max = inputs[:, 1].min() - 1, inputs[:, 1].max() + 1
    z_min, z_max = inputs[:, 2].min() - 1, inputs[:, 2].max() + 1

    xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01), np.arange(z_min, z_max, 0.01))
    Z = test_perceptron(np.c_[xx.ravel(), yy.ravel(), zz.ravel()], weights, bias)
    Z = Z.reshape(xx.shape)

    # Graficar plano de separación en 3D
    ax.plot_surface(xx, yy, Z, facecolors='gray', alpha=0.5)

    ax.set_xlabel('Entrada X1')
    ax.set_ylabel('Entrada X2')
    ax.set_zlabel('Entrada X3')
    plt.title(title)
    plt.grid(True)
    plt.show()

# Graficar un ejemplo de una de las particiones
graphic_3d(train_data, train_labels, trained_weights, trained_bias, f'Patrones y Plano de Separación (3D) - {dataset_name} - Partición 1')
