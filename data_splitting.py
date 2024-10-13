import numpy as np
import pandas as pd
import preprocessing as pp

def data_split_dev(cd):
    Data2 = []
    Año = cd['ano']
    for column in cd:
        if column != 'Precio' and column != 'colores':
            Data2.append(cd[column].values)
    Data2 = list(zip(*Data2))
    A_test = []
    A_train = []
    Data_train = []
    Data_test = []
    y_train = []
    y_test =  []
    i = 0
    Data_train.append(Data2[i])
    A_train.append(Año[i])
    y_train.append(cd['Precio'][i])
    i += 1

    Data_test.append(Data2[i])
    A_test.append(Año[i])
    y_test.append(cd['Precio'][i])
    i += 1
    #separar los datos de entrenamientos en 80-20 tratando de mantener el promedio de anos igual en ambos lados. 
    while i < 1541: 
        if len(A_test) < 1541 * 0.2:
            if np.mean(A_test) <= np.mean(A_train):
                if Año[i] > np.mean(A_train):
                    Data_test.append(Data2[i])
                    A_test.append(Año[i])
                    y_test.append(cd['Precio'][i])
                    i += 1
                
                if Año[i] <= np.mean(A_train):
                    Data_train.append(Data2[i])
                    A_train.append(Año[i])
                    y_train.append(cd['Precio'][i])
                    i += 1
                
            if np.mean(A_test) > np.mean(A_train):
                if Año[i] < np.mean(A_train):
                    Data_test.append(Data2[i])
                    A_test.append(Año[i])
                    y_test.append(cd['Precio'][i])
                    i += 1
                    
                if Año[i] >= np.mean(A_train):
                    Data_train.append(Data2[i])
                    A_train.append(Año[i])
                    y_train.append(cd['Precio'][i])
                    i += 1
        else:
            Data_train.append(Data2[i])
            A_train.append(Año[i])
            y_train.append(cd['Precio'][i])
            i += 1
        
    y_train = np.array(y_train)
    y_train = np.transpose(y_train)
    y_train = y_train[:, np.newaxis]
    y_test = np.array(y_test)
    y_test = np.transpose(y_test)
    y_test = y_test[:, np.newaxis]
    x_train,vals_min,vals_max = pp.min_max_scaling(np.array(Data_train))
    x_test = pp.normalize_test(np.array(Data_test),vals_min,vals_max)
    return x_train,y_train,x_test,y_test,vals_min,vals_max

def data_create_test(cd,vals_min,vals_max):
    Data2 = []
    
    for column in cd:
        if column != 'Precio' and column != 'colores':
            Data2.append(cd[column].values)
    Data2 = list(zip(*Data2))
    y_test = np.array(cd['Precio'])
    y_test = np.transpose(y_test)
    y_test = y_test[:, np.newaxis]
    x_test = pp.normalize_test(np.array(Data2),vals_min,vals_max)
    return x_test,y_test


import numpy as np

# Funciones auxiliares para calcular las métricas

# Función para hacer k-fold cross-validation y calcular las métricas
def k_fold_cross_validation(model, X, y, k,mae,rmse,r2_score):
    # Shuffle y dividir los datos en k-folds
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    fold_size = len(X) // k
    folds_X = [X_shuffled[i * fold_size: (i + 1) * fold_size] for i in range(k)]
    folds_y = [y_shuffled[i * fold_size: (i + 1) * fold_size] for i in range(k)]
    
    # Inicializar listas para guardar las métricas
    mae_list = []
    rmse_list = []
    r2_list = []
    
    # Iterar sobre los folds
    for i in range(k):
        # Fold actual como conjunto de validación
        X_val = folds_X[i]
        y_val = folds_y[i]
        
        # Resto como conjunto de entrenamiento
        X_train = np.vstack([folds_X[j] for j in range(k) if j != i])
        y_train = np.hstack([folds_y[j] for j in range(k) if j != i])
        if y_train.ndim != 1:
            y_train = y_train.reshape(-1, 1)
        # Entrenar el modelo con los datos de entrenamiento
        model.train(X_train, y_train)
        
        # Predecir con los datos de validación
        y_pred = model.predict(X_val)
        
        # Calcular las métricas
        mae_list.append(mae(y_val, y_pred))
        rmse_list.append(rmse(y_val, y_pred))
        r2_list.append(r2_score(y_val, y_pred))
    
    # Retornar los promedios de las métricas
    return [np.mean(mae_list), np.mean(rmse_list),np.mean(r2_list)]




