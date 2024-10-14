#curacion de datos
import csv
import pandas as pd
import preprocessing as pp
import numpy as np
import data_splitting as ds
id = []
Tipo = []
Año = []
Colors = []
combustiblex = []
transm = []
motores = []
Kilometros = []
vendedors = []
precios = []

id2 = []
Tipo2 = []
Año2 = []
Colors2 = []
combustiblex2 = []
transm2 = []
motores2 = []
Kilometros2 = []
vendedors2 = []
precios2 = []

Data = []
Data_2_test =[]

with open('toyota_dev.csv', mode='r', newline='') as file:
    csv_reader = csv.reader(file)
    i = 0
    
    for row in csv_reader:
        if i != 0:
            Data.append(row)
        i += 1

with open('toyota_test.csv', mode='r', newline='') as file:
    csv_reader = csv.reader(file)
    i = 0
    
    for row in csv_reader:
        if i != 0:
            Data_2_test.append(row)
        i += 1

for j in Data:
    Tipo.append(j[1])
    Año.append(j[2])
    Colors.append(j[3])
    combustiblex.append(j[4])
    transm.append(j[5])
    motores.append(j[6])
    Kilometros.append(j[7])
    vendedors.append(j[8])
    precios.append(float(j[9]))

for j in Data_2_test:
    Tipo2.append(j[1])
    Año2.append(j[2])
    Colors2.append(j[3])
    combustiblex2.append(j[4])
    transm2.append(j[5])
    motores2.append(j[6])
    Kilometros2.append(j[7])
    vendedors2.append(j[8])
    precios2.append(float(j[9]))
# Convertir los años de str a int
Año = [float(anio) for anio in Año]
Año = [int(anio) for anio in Año]

#test 2
Año2 = [float(anio) for anio in Año2]
Año2 = [int(anio) for anio in Año2]


# Limpiar y convertir los kilómetros de str a int
Kilometros = [int(km.replace(' km', '').replace(',', '')) for km in Kilometros]

#test 2 
Kilometros2 = [int(km.replace(' km', '').replace(',', '')) for km in Kilometros2]
for j in range(len(Colors)):
    if not Colors[j]:
        if Data[j][1] == "RAV4":
            Colors[j] = 'Bordó'
        if Data[j][1] == 'Hilux SW4':
            Colors[j] = 'Bordó'

        if Data[j][1] == 'Corolla Cross':
            Colors[j] = 'Blanco'
        
#test 2
for j in range(len(Colors2)):
    if not Colors2[j]:
        if Data_2_test[j][1] == "RAV4":
            Colors2[j] = 'Bordó'
        if Data_2_test[j][1] == 'Hilux SW4':
            Colors2[j] = 'Bordó'

        if Data_2_test[j][1] == 'Corolla Cross':
            Colors2[j] = 'Blanco'


for i in range(len(transm)):
    if not transm[i]:
        transm[i] = 'Automática'
#test2
for i in range(len(transm2)):
    if not transm2[i]:
        transm2[i] = 'Automática'
# Convertir todos los valores a string y limpiar los datos
motores = [str(motor).strip().replace(',', '.').upper() for motor in motores]



# Función para determinar si un string es un número
def es_numero(valor):
    try:
        return float(valor)
    except ValueError:
        return None  # Indica que el valor no es numérico

# Aplicar la función y reemplazar valores no numéricos
motores_limpios = [es_numero(motor) if es_numero(motor) is not None else 0 for motor in motores]
prom_RAV = 0
prom_corolla = 0
prom_SW4 = 0
for i in range(len(motores_limpios)):

    if Data[i][1] == "RAV4":
        prom_RAV += motores_limpios[i]
    if Data[i][1] == 'Hilux SW4':
        prom_SW4 += motores_limpios[i]

    if Data[i][1] == 'Corolla Cross':
        prom_corolla += motores_limpios[i]
prom_RAV = round(prom_RAV/196,1)
prom_SW4 = round(prom_SW4/905,1)
prom_corolla = round(prom_corolla/440,1)

for i in range(len(motores_limpios)):
    if motores_limpios[i] == 0:
        if Data[i][1] == "RAV4":
            motores_limpios[i] = prom_RAV
        if Data[i][1] == 'Hilux SW4':
            motores_limpios[i] = prom_SW4

        if Data[i][1] == 'Corolla Cross':
            motores_limpios[i] = prom_corolla

motores2 = [str(motor).strip().replace(',', '.').upper() for motor in motores2]





# Aplicar la función y reemplazar valores no numéricos
motores_limpios2 = [es_numero(motor) if es_numero(motor) is not None else 2.0 for motor in motores2]

for i in range(len(motores_limpios2)):
    if motores_limpios2[i] == 0:
        motores_limpios2[i] = np.mean(motores_limpios2)
pp.one_hot_combustible(combustiblex)
pp.one_hot_transmision(transm)
pp.one_hot_marca(Tipo)
cd = pd.DataFrame({
    "ano" : Año,
    'km' : Kilometros,
    'motor' : motores_limpios,
    'C_Cross': pp.C_Cross_hot,
    'SW4' : pp.SW4_hot,
    'RAV4' : pp.RAV4_hot,
    'nafta' : pp.combust_1,
    'Diesel' : pp.combust_2,
    'Hib/Naf' : pp.combust_3, 
    'hibrido' : pp.combust_4,
    "electrico" : pp.combust_5,
    'Nafta/gnc' : pp.combust_6,
    'colores' : Colors,
    'automatica': pp.transm1,
    'manual' : pp.transm2,
    'automatica secuencial' : pp.transm3,
    'Vendedor': vendedors,
    'Precio': precios
})

mean_encoding = cd.groupby('colores')['Precio'].mean()
mean_encoding2 = cd.groupby('Vendedor')['Precio'].mean()
cd['Vendedor'] = cd['Vendedor'].map(mean_encoding2)
cd['colores'] = cd['colores'].map(mean_encoding)


pp.one_hot_combustible2(combustiblex2)
pp.one_hot_transm2(transm2)
pp.one_hot_marca_2(Tipo2)


data_2_frame = pd.DataFrame({
    "ano" : Año2,
    'km' : Kilometros2,
    'motor' : motores_limpios2,
    'C_Cross': pp.C_Cross_hot22,
    'SW4' : pp.SW4_hot22,
    'RAV4' : pp.RAV4_hot22,
    'nafta' : pp.c1,
    'Diesel' : pp.c2,
    'Hib/Naf' : pp.c3, 
    'hibrido' : pp.c4,
    "electrico" : pp.c5,
    'Nafta/gnc' : pp.c6,
    'automatica': pp.t1,
    'manual' : pp.t2,
    'automatica secuencial' : pp.t3,
    'Vendedor': vendedors2,
    'Precio': precios2
})
#mean_encoding = data_2_frame.groupby('colores')['Precio'].mean()
mean_encoding2 = data_2_frame.groupby('Vendedor')['Precio'].mean()
data_2_frame['Vendedor'] = data_2_frame['Vendedor'].map(mean_encoding2)
#data_2_frame['colores'] = data_2_frame['colores'].map(mean_encoding)



data_frame_dev = cd
data_frame_test = data_2_frame
x_train,y_train,x_test,y_test,vals_min,vals_max = ds.data_split_dev(data_frame_dev)
x_test_2,y_test_2 = ds.data_create_test(data_frame_test,vals_min,vals_max)
X_kf,Y_kf = ds.data_create_test(data_frame_dev,vals_min,vals_max)