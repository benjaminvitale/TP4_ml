import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def mean_absolute_error(y_test,y_pred):
    # Calcular el error absoluto
    errors = np.abs(y_test - y_pred)

    # Calcular el MAE
    mae = np.mean(errors)

    return mae

def RMSE(y_test,y_pred):
    errors = (y_test - y_pred)**2
    RMSE = np.mean(errors)
    return np.sqrt(RMSE)

def rsquare(y_test,y_pred):
    y_mean = np.mean(y_test)
    SS_res = np.sum((y_test - y_pred) ** 2)
    SS_tot = np.sum((y_test - y_mean) ** 2)

    # Cálculo de R^2
    return 1 - (SS_res / SS_tot)

def calcular_residuos(y_real, y_pred):
    return np.abs(y_real - y_pred)


def graficar_histograma_residuos(y_real, y_pred_model1, y_pred_model2, y_pred_model3, title, bins=20):
    residuos_model1 = calcular_residuos(y_real, y_pred_model1)
    residuos_model2 = calcular_residuos(y_real, y_pred_model2)
    residuos_model3 = calcular_residuos(y_real, y_pred_model3)
    
    plt.figure(figsize=(12, 8))
    
    # Histograma para el Modelo 1
    sns.histplot(residuos_model1, bins=bins, color='blue', label='Modelo 1', kde=True, stat="density", element="step")
    
    # Histograma para el Modelo 2
    sns.histplot(residuos_model2, bins=bins, color='green', label='Modelo 2', kde=True, stat="density", element="step")
    
    # Histograma para el Modelo 3
    sns.histplot(residuos_model3, bins=bins, color='red', label='Modelo 3', kde=True, stat="density", element="step")
    
    plt.title(f'Histograma de Residuos Absolutos - {title}')
    plt.xlabel('Residuos absolutos')
    plt.ylabel('Densidad')
    plt.legend()
    plt.show()

# Ejemplo de cómo llamar la función con las predicciones para los distintos conjuntos

