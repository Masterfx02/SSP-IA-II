import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

#*Librerias para evaluar las clasificaciones
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

def zoo():

    #cargar y dividir los datos de zoo dataset
    dataset = pd.read_csv('zoo.csv')
    X = dataset.drop(['animal_name','type'],axis=1)
    y = dataset['type']
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
    return X_train, X_test, y_train, y_test



# Modelos de regresión
def logistic_Regression(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=10000)
    # Ajustar el modelo a los datos de entrenamiento
    model.fit(X_train, y_train)
    # Predecir en el conjunto de prueba
    y_pred = model.predict(X_test)
    
    #Cálculos y resultados de las métricas utilizadas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred,average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred,average='weighted', zero_division=1)
    f1 = f1_score(y_test, y_pred,average='weighted')
    #specificity
    confu_matrix =confusion_matrix(y_test,y_pred)
    tn, fp, fn, tp = confu_matrix.ravel()[:4]
    specificity = tn / (tn + fp)

    # Graficar la matriz de confusión como un mapa de calor

    plt.figure(figsize=(8, 6))
    sns.heatmap(confu_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    print("-------- Logistic Regression --------\n")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"Specificity: {specificity:.3f}")
    print(f"F1 Score: {f1:.3f}")
    
    # Graficar las métricas
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'Specifity', 'F1 Score']
    metrics_values = [accuracy, precision, recall, specificity, f1]

    plt.bar(metrics_names, metrics_values, color=['blue', 'green', 'red', 'purple', 'orange'])
    plt.title('Metrics for Logistic Regression')
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    # Añadir valores en cada barra
    for i, value in enumerate(metrics_values):
        plt.text(i, value + 0.01, f'{value:.3f}', ha='center', va='bottom')
    plt.show()

def k_Nearest_Neighbors(X_train, X_test, y_train, y_test):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    #Cálculos y resultados de las métricas utilizadas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred,average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred,average='weighted', zero_division=1)
    f1 = f1_score(y_test, y_pred,average='weighted')
    #specificity
    confu_matrix =confusion_matrix(y_test,y_pred)
    tn, fp, fn, tp = confu_matrix.ravel()[:4]
    specificity = tn / (tn + fp)
    

    # Graficar la matriz de confusión como un mapa de calor
    plt.figure(figsize=(8, 6))
    sns.heatmap(confu_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    print("\n-------- K-Nearest Neighbors --------\n")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"Specificity: {specificity:.3f}")
    print(f"F1 Score: {f1:.3f}")
    
    # Graficar las métricas
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'Specifity', 'F1 Score']
    metrics_values = [accuracy, precision, recall, specificity, f1]

    plt.bar(metrics_names, metrics_values, color=['blue', 'green', 'red', 'purple', 'orange'])
    plt.title('Metrics for K-Nearest Neighbors')
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    # Añadir valores en cada barra
    for i, value in enumerate(metrics_values):
        plt.text(i, value + 0.01, f'{value:.3f}', ha='center', va='bottom')
    plt.show()
    
def support_Vector_Machine(X_train, X_test, y_train, y_test):
    
    model = SVC(C=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    #Cálculos y resultados de las métricas utilizadas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred,average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred,average='weighted', zero_division=1)
    f1 = f1_score(y_test, y_pred,average='weighted')
    #specificity
    confu_matrix =confusion_matrix(y_test,y_pred)
    tn, fp, fn, tp = confu_matrix.ravel()[:4]
    specificity = tn / (tn + fp)
    

    # Graficar la matriz de confusión como un mapa de calor
    plt.figure(figsize=(8, 6))
    sns.heatmap(confu_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    print("\n-------- Support Vector Machine --------\n")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"Specificity: {specificity:.3f}")
    print(f"F1 Score: {f1:.3f}")
    
    # Graficar las métricas
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'Specifity', 'F1 Score']
    metrics_values = [accuracy, precision, recall, specificity, f1]

    plt.bar(metrics_names, metrics_values, color=['blue', 'green', 'red', 'purple', 'orange'])
    plt.title('Metrics for Support Vector Machine')
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    # Añadir valores en cada barra
    for i, value in enumerate(metrics_values):
        plt.text(i, value + 0.01, f'{value:.3f}', ha='center', va='bottom')
    plt.show()
    
    
def naive_Bayes(X_train, X_test, y_train, y_test):
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    #Cálculos y resultados de las métricas utilizadas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred,average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred,average='weighted', zero_division=1)
    f1 = f1_score(y_test, y_pred,average='weighted')
    #specificity
    confu_matrix =confusion_matrix(y_test,y_pred)
    tn, fp, fn, tp = confu_matrix.ravel()[:4]
    specificity = tn / (tn + fp)


    # Graficar la matriz de confusión como un mapa de calor
    plt.figure(figsize=(8, 6))
    sns.heatmap(confu_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    print("\n-------- Naive Bayes --------\n")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"Specificity: {specificity:.3f}")
    print(f"F1 Score: {f1:.3f}")
    
    # Graficar las métricas
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'Specifity', 'F1 Score']
    metrics_values = [accuracy, precision, recall, specificity, f1]

    plt.bar(metrics_names, metrics_values, color=['blue', 'green', 'red', 'purple', 'orange'])
    plt.title('Metrics for Naive Bayes')
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    # Añadir valores en cada barra
    for i, value in enumerate(metrics_values):
        plt.text(i, value + 0.01, f'{value:.3f}', ha='center', va='bottom')
    plt.show()
    
def MLP(X_train, X_test, y_train, y_test, hidden_layer_sizes=(100,50), max_iter=2000):

    
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    #Cálculos y resultados de las métricas utilizadas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred,average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred,average='weighted', zero_division=1)
    f1 = f1_score(y_test, y_pred,average='weighted')
    #specificity
    confu_matrix =confusion_matrix(y_test,y_pred)
    tn, fp, fn, tp = confu_matrix.ravel()[:4]
    specificity = tn / (tn + fp)
    

    # Graficar la matriz de confusión como un mapa de calor
    plt.figure(figsize=(8, 6))
    sns.heatmap(confu_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    print("\n-------- MLP --------\n")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"Specificity: {specificity:.3f}")
    print(f"F1 Score: {f1:.3f}")
    
    # Graficar las métricas
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'Specifity', 'F1 Score']
    metrics_values = [accuracy, precision, recall, specificity, f1]

    plt.bar(metrics_names, metrics_values, color=['blue', 'green', 'red', 'purple', 'orange'])
    plt.title('Metrics for MLP')
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    # Añadir valores en cada barra
    for i, value in enumerate(metrics_values):
        plt.text(i, value + 0.01, f'{value:.3f}', ha='center', va='bottom')
    plt.show()
    

# Evaluar los clasificadores
X_train, X_test, y_train, y_test = zoo()
logistic_Regression(X_train, X_test, y_train, y_test)
k_Nearest_Neighbors(X_train, X_test, y_train, y_test)
support_Vector_Machine(X_train, X_test, y_train, y_test)
naive_Bayes(X_train, X_test, y_train, y_test)
MLP(X_train, X_test, y_train, y_test, hidden_layer_sizes=(16,16), max_iter=1000)
