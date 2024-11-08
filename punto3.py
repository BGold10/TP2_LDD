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


#bruno
#prefijo = 'C:/Users/Bruno Goldfarb/Downloads/'

#seba
#prefijo = 'C:\\Users\\Sebastián\\Documents\\LaboDeDatos\\TP2\\'
prefijo = '/home/Estudiante/Escritorio/TP2_LDD/TP2/'

data = pd.read_csv(prefijo + 'TMNIST_Data.csv')


#%%

#con esta función imprimimos las primeras 10 imagenes del dataframe

def imprimir_img (data):
    imagenes = data.iloc[:, 2:] #saco las 2 primeras columnas
    for i in range(10):
        img = np.array(imagenes.iloc[i]).reshape((28,28))
        plt.imshow(img, cmap='gray')
        plt.show()
    

imprimir_img(data)

#%%

#1c
consulta_sql = """
                SELECT DISTINCT * 
                FROM data
                WHERE labels == '0';
               """

img0 = sql^consulta_sql

imprimir_img(img0)

#Se puede observar que la mayoría de las imagenes que corresponden al 0 cumplen que en 
#centro tienen como valor 0. Sin embargo en algunos casos no se cumple, por ejemplo para 
#la fuente Mitr-Bold.

#%%

#EJ 2

#a
#Armo df con digitos 0 y 1

consulta_sql = """
                SELECT DISTINCT * 
                FROM data
                WHERE labels == '0' OR  labels == '1';               
               """
               
img01 = sql^consulta_sql

#Queremos ver si esta balanceado.

consulta_sql = """
                SELECT labels, COUNT(*) AS cantXdigito
                FROM img01
                GROUP BY labels;
               """

cantidadXdigito = sql^consulta_sql

#mismo numero de 0s y 1s, esta balanceado.

#%%
#b

#Primero separamos la data de los labels.

X = data.iloc[:,2:]
Y = data['labels']

#%%
#Separamos en datos de train y test y tomo 3 atributos.

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3) # 70% para train y 30% para test

X_train3 = X_train[[ '186','381', '661']]
X_test3 = X_test[[ '186','381', '661']]


#%%

#ahora realizamos el modelo

model = KNeighborsClassifier(n_neighbors = 5)
model.fit(X_train3, Y_train) 
Y_pred = model.predict(X_test3) 
print("Exactitud del modelo:", metrics.accuracy_score(Y_test, Y_pred))
metrics.confusion_matrix(Y_test, Y_pred)

#no tiene buena exactitud, pruebo con todos los atributos.

#%%

model = KNeighborsClassifier(n_neighbors = 5)
model.fit(X_train, Y_train) 
Y_pred = model.predict(X_test) 
print("Exactitud del modelo:", metrics.accuracy_score(Y_test, Y_pred))
metrics.confusion_matrix(Y_test, Y_pred)

#aumneta a 0.96

#hay que seguir probando con distintos atributos, ver las metricas y eso.

#%%



#3. (Clasificación multiclase) 

#separo el modelo en train, testing y validation:
    
X, X_validation, Y, Y_validation = train_test_split(X, Y, test_size = 0.2) # 80% para train y 20% para validation



X_train, x_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size = 0.3) #70% para train y 30% para testing

#%%        

cnombres = ['0', '1', '2', '3', '4', '5', '6', '7','8','9']

#PROBAMOS CON DISTINTAS PROFUNDIDAES:
    
#depth= 1

arbol = tree.DecisionTreeClassifier(criterion = "entropy", max_depth= 1)
arbol = arbol.fit(X_train, Y_train)



plt.figure(figsize= [15,10])
tree.plot_tree(arbol,filled = True, rounded = True, fontsize = 10,class_names=cnombres)


#%%
#depth =5
arbol = tree.DecisionTreeClassifier(criterion = "entropy", max_depth= 5)
arbol = arbol.fit(X_train, Y_train)



plt.figure(figsize= [15,10])
tree.plot_tree(arbol,filled = True, rounded = True, fontsize = 10,class_names=cnombres)

#%%
#depth =10
arbol = tree.DecisionTreeClassifier(criterion = "entropy", max_depth= 10)
arbol = arbol.fit(X_train, Y_train)



plt.figure(figsize= [15,10])
tree.plot_tree(arbol,filled = True, rounded = True, fontsize = 10,class_names=cnombres)
#seba dice: recontra overfil!

#%%
#PELIGRO AL CORRER ESTO TARDA MUCHISIMO!!!!!!!!!!!

Nrep = 5
valores_n = range(1, 11)

resultados_test_gini = np.zeros((Nrep, len(valores_n)))
resultados_train_gini = np.zeros((Nrep, len(valores_n)))

criterion= ["entropy","gini"]

for i in range(Nrep):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)
    for k in valores_n:
        model = tree.DecisionTreeClassifier(criterion = 'gini', max_depth= k)
        model.fit(X_train, Y_train) 
        Y_pred = model.predict(X_test)
        Y_pred_train = model.predict(X_train)
        acc_test = metrics.accuracy_score(Y_test, Y_pred)
        acc_train = metrics.accuracy_score(Y_train, Y_pred_train)
        resultados_test_gini[i, k-1] = acc_test
        resultados_train_gini[i, k-1] = acc_train
        
resultados_test_entropy = np.zeros((Nrep, len(valores_n)))
resultados_train_entropy = np.zeros((Nrep, len(valores_n)))
        
for i in range(Nrep):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)
    for k in valores_n:
        model = tree.DecisionTreeClassifier(criterion = 'gini', max_depth= k)
        model.fit(X_train, Y_train) 
        Y_pred = model.predict(X_test)
        Y_pred_train = model.predict(X_train)
        acc_test = metrics.accuracy_score(Y_test, Y_pred)
        acc_train = metrics.accuracy_score(Y_train, Y_pred_train)
        resultados_test_entropy[i, k-1] = acc_test
        resultados_train_entropy[i, k-1] = acc_train
#%%

promedios_train_gini = np.mean(resultados_train_gini,axis=0)
promedios_test_gini= np.mean(resultados_test_gini,axis=0)

promedios_train_entropy = np.mean(resultados_train_entropy,axis=0)
promedios_test_entropy= np.mean(resultados_test_entropy,axis=0)
#En todos los casos, mientrsa mas preguntas hacmoes, mejor precision tiene el modelo.pero con 10 tarda mucho. 
#Definimos por usar depth = 9
#%%


#Ahora salgamos al mundo real, y con el modelo de depth = 9 intentemos predecir:
    
    

    
model = tree.DecisionTreeClassifier(criterion = 'gini', max_depth= 9)
model.fit(X, Y) 
Y_pred = model.predict(X_validation)
Y_pred_train = model.predict(X)
acc_test = metrics.accuracy_score(Y_validation, Y_pred)
acc_train = metrics.accuracy_score(Y, Y_pred_train)
metrics.confusion_matrix(Y, Y_pred_train)





















#%%








#%%

datonuevo= pd.DataFrame([dict(zip(iris['feature_names'], [6.8,3,4.5, 2.15]))])
clf_info.predict(datonuevo)


#%%
# otra forma de ver el arbol
r = tree.export_text(clf_info, feature_names=iris['feature_names'])
print(r)













