# Grupo: Pescado rabioso
# Participantes:
#               -Souto, Sebastian Manuel
#               -Sanza, Gian Lucca
#               -Goldfarb, Bruno

from inline_sql import sql, sql_val
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import tree
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import random as rd
import copy 
#bruno
#prefijo = 'C:/Users/Bruno Goldfarb/Downloads/'

#seba
#prefijo = 'C:\\Users\\Sebastián\\Documents\\LaboDeDatos\\TP2\\'

#labo
prefijo = '/home/Estudiante/Descargas/TP2_LDD-main/TP2/'
data = pd.read_csv(prefijo + 'TMNIST_Data.csv')



#%%
X = data.iloc[:,2:]
Y = data['labels']

#%%
#ia)
n = 200
indices = []
for i in range(n):
    indice = rd.randint(0,29899)
    indices.append(indice)        
muestreo = X.iloc[indices]    
promedios = muestreo.mean(axis=0).tolist()
muestreo.loc[len(muestreo)] = promedios

#%%

# muestreo, prom = sacarPromedioAleatorio(X, n)    
def dibujarPromedio(muestreo):
    n = 200
    img = np.array(muestreo.iloc[n]).reshape((28,28))
    plt.imshow(img, cmap='gray')
    plt.show()
dibujarPromedio(muestreo)
#%%
consulta_sql = """
                SELECT DISTINCT * 
                FROM data
                WHERE labels == '1' OR  labels == '0';               
               """             
img1 = sql^consulta_sql
n = 500
indices = []
data1 = img1.iloc[:,2:]
for i in range(n):
    indice = rd.randint(0, data1.shape[0])
    indices.append(indice)
muestreo = data1.iloc[indices]
promedios = muestreo.mean(axis=0).tolist()
img2 = np.array(promedios).reshape((28,28))

# n = data1.shape[0]
# data1.loc[len(muestreo)] = promedios
# img = np.array(promedios).reshape((28,28))
plt.imshow(img2, cmap='gray')
plt.show()



#flag_r






































