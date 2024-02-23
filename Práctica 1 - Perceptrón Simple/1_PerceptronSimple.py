import numpy as np
import matplotlib.pyplot as plt

# Se define la función del perceptrón
def perceptron(inputs, weights, bias):
    summation = np.dot(inputs, weights) + bias
    #Función de activación
    if summation >= 0:
        return 1
    else:
        return 0
#Leer patrones desde el archivo csv
def read_patterns(file):
    data = np.genfromtxt(file, delimiter=',')
    inputs = data[:, :-1]
    outputs = data[:, -1]
    return inputs, outputs

def normalize_data(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)


# Entrenar el perceptrón
def train_perceptron(inputs, outputs, learning_rate, max_epochs, convergence_criterion):
    inputs = normalize_data(inputs)
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

#Testear el perceptrón entrenado
def test_perceptron( inputs, weights, bias):
    output_received = np.dot(inputs, weights) + bias
    return np.sign(output_received)

#Calcular la precisión
def calculate_accuracy(outputs_real, outputs_predictions):
    correct_predictions = np.sum(outputs_real == outputs_predictions)
    total_predictions = len(outputs_real)
    accuracy = correct_predictions / total_predictions
    return accuracy

def graphic(inputs, outputs, weights, bias):
    plt.figure(figsize=(8, 6))
    # Graficar patrones
    plt.scatter(inputs[:, 0], inputs[:, 1], c=outputs, s=100)
    
    # Graficar recta de separación
    x_min, x_max = inputs[:, 0].min() - 1, inputs[:, 0].max() + 1
    y_min, y_max = inputs[:, 1].min() - 1, inputs[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = test_perceptron(np.c_[xx.ravel(), yy.ravel()], weights, bias)
    Z = Z.reshape(xx.shape)
    
    plt.contour(xx, yy, Z, colors='k', linestyles=['-'], levels=[0])
    plt.title('Patrones y Recta Separadora')
    plt.xlabel('Entrada X1')
    plt.ylabel('Entrada X2')
    plt.grid(True)
    plt.show()
    
#Llamado a función principal
if __name__ == "__main__":
    # Lectura de patrones de entrenamiento y prueba desde archivos CSV
    training_file = 'XOR_trn.csv'
    test_file = 'XOR_tst.csv'
    #Patrón de entrenamiento
    inputs_train, outputs_train = read_patterns(training_file)
    #Patrón de prueba
    inputs_test, outputs_test = read_patterns(test_file)

    #Parámetros para entrenamiento
    max_epochs = 100
    learning_rate = 0.1
    convergence_criterion = 0.01  # Alteraciones aleatorias < 5%
    
    # Entrenamiento del perceptrón
    trained_weights, trained_bias = train_perceptron(inputs_train, outputs_train, learning_rate, 
    max_epochs, convergence_criterion)
    print("Perceptrón entrenado con éxito.")
    #Probar el perceptrón entrenado en datos prueba
    outputs_predictions = test_perceptron(inputs_test, trained_weights, trained_bias)
    
    # Calcular la precisión
    accuracy = calculate_accuracy(outputs_test, outputs_predictions)
    print("Precisión del perceptrón en datos de prueba (Accuracy):", accuracy)
    
    # Mostrar resultados
    print("Salidas en prueba:")
    print(outputs_test)
    print("Salidas predichas por el perceptrón:")
    print(outputs_predictions)

    graphic(inputs_train, outputs_train, trained_weights, trained_bias)