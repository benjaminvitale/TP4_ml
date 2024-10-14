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




