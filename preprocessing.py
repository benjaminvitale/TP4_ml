import numpy as np

combu_uniq = ['Nafta','Diésel','Híbrido/Nafta','Híbrido','Eléctrico','Nafta/GNC']
transmisiones = ['Automática','Manual','Automática secuencial']
RAV4_hot = []
SW4_hot = []
C_Cross_hot = []
combust_1 = []
combust_2 = []
combust_3 = []
combust_4 = []
combust_5 = []
combust_6 = []
transm1 = []
transm2 = []
transm3 = []

c1 = []
c2 = []
c3 = []
c4 = []
c5 = []
c6 = []
t1 = []
t2 = []
t3 = []
RAV4_hot22 = []
SW4_hot22 = []
C_Cross_hot22 = []

def one_hot_combustible(combustibless):
    for i in combustibless:
        if i == combu_uniq[0]:
            combust_1.append(1)
            combust_2.append(0)
            combust_3.append(0)
            combust_4.append(0)
            combust_5.append(0)
            combust_6.append(0)
        if i == combu_uniq[1]:
            combust_1.append(0)
            combust_2.append(1)
            combust_3.append(0)
            combust_4.append(0)
            combust_5.append(0)
            combust_6.append(0)
        if i == combu_uniq[2]:
            combust_1.append(0)
            combust_2.append(0)
            combust_3.append(1)
            combust_4.append(0)
            combust_5.append(0)
            combust_6.append(0)
        if i == combu_uniq[3]:
            combust_1.append(0)
            combust_2.append(0)
            combust_3.append(0)
            combust_4.append(1)
            combust_5.append(0)
            combust_6.append(0)
        if i == combu_uniq[4]:
            combust_1.append(0)
            combust_2.append(0)
            combust_3.append(0)
            combust_4.append(0)
            combust_5.append(1)
            combust_6.append(0)
        if i == combu_uniq[5]:
            combust_1.append(0)
            combust_2.append(0)
            combust_3.append(0)
            combust_4.append(0)
            combust_5.append(0)
            combust_6.append(1)
    
def one_hot_transmision(transmision_):
    for i in transmision_:
        if i == transmisiones[0]:
            transm1.append(1)
            transm2.append(0)
            transm3.append(0)
        if i == transmisiones[1]:
            transm1.append(0)
            transm2.append(1)
            transm3.append(0)
        if i == transmisiones[2]:
            transm1.append(0)
            transm2.append(0)
            transm3.append(1)
        
    
#one hot encoding para la marca del auto 
def one_hot_marca(marca):
    for i in marca:
        if i == 'RAV4':
            RAV4_hot.append(1)
            SW4_hot.append(0)
            C_Cross_hot.append(0)
        if i == 'Hilux SW4':
            RAV4_hot.append(0)
            SW4_hot.append(1)
            C_Cross_hot.append(0)
        if i == 'Corolla Cross':
            RAV4_hot.append(0)
            SW4_hot.append(0)
            C_Cross_hot.append(1)


def one_hot_combustible2(combu):


    for i in combu:
        if i == combu_uniq[0]:
            c1.append(1)
            c2.append(0)
            c3.append(0)
            c4.append(0)
            c5.append(0)
            c6.append(0)
        if i == combu_uniq[1]:
            c1.append(0)
            c2.append(1)
            c3.append(0)
            c4.append(0)
            c5.append(0)
            c6.append(0)
        if i == combu_uniq[2]:
            c1.append(0)
            c2.append(0)
            c3.append(1)
            c4.append(0)
            c5.append(0)
            c6.append(0)
        if i == combu_uniq[3]:
            c1.append(0)
            c2.append(0)
            c3.append(0)
            c4.append(1)
            c5.append(0)
            c6.append(0)
        if i == combu_uniq[4]:
            c1.append(0)
            c2.append(0)
            c3.append(0)
            c4.append(0)
            c5.append(1)
            c6.append(0)
        if i == combu_uniq[5]:
            c1.append(0)
            c2.append(0)
            c3.append(0)
            c4.append(0)
            c5.append(0)
            c6.append(1)

def one_hot_transm2(transmisiones_):
    for i in transmisiones_:
        if i == transmisiones[0]:
            t1.append(1)
            t2.append(0)
            t3.append(0)
        if i == transmisiones[1]:
            t1.append(0)
            t2.append(1)
            t3.append(0)
        if i == transmisiones[2]:
            t1.append(0)
            t2.append(0)
            t3.append(1)

def one_hot_marca_2(marcas):
    for i in marcas:
        if i == 'RAV4':
            RAV4_hot22.append(1)
            SW4_hot22.append(0)
            C_Cross_hot22.append(0)
        if i == 'Hilux SW4':
            RAV4_hot22.append(0)
            SW4_hot22.append(1)
            C_Cross_hot22.append(0)
        if i == 'Corolla Cross':
            RAV4_hot22.append(0)
            SW4_hot22.append(0)
            C_Cross_hot22.append(1)
def min_max_scaling(X):
    vals_min = []
    vals_max = []

    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    vals_min.append(min_val)
    vals_max.append(max_val)
    
    # Evitar división por cero
    range_val = max_val - min_val
    range_val[range_val == 0] = 1
    
    X_scaled = (X - min_val) / range_val
    return X_scaled,np.array(vals_min),np.array(vals_max)

def normalize_test(X,vals_min,vals_max):
    return (X - vals_min) / (vals_max - vals_min)