# Grupo: Pescado rabioso
# Participantes:
#               -Souto, Sebastian Manuel
#               -Sanza, Gian Lucca
#               -Goldfarb, Bruno

import pandas as pd
from inline_sql import sql, sql_val
import numpy as np
import matplotlib.pyplot as plt # Para graficar series multiples
from   matplotlib import ticker   # Para agregar separador de miles
import seaborn as sns  

bruno = 'C:/Users/Bruno Goldfarb/Downloads/'


data = pd.read_csv(bruno + 'TMNIST_Data.csv')
imagenes = data.iloc[:, 2:] #saco las 2 primeras columnas
for i in range(10):
    img = np.array(imagenes.iloc[i]).reshape((28,28))
    plt.imshow(img, cmap='gray')
    plt.show()
    