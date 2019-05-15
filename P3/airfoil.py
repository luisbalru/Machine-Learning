# Práctica 3 Aprendiza Automático 2019
# Autor: Luis Balderas Ruiz

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print("AIRFOIL SELF-NOISE")

names = ['Frequency','Angle-Attack','Chord-Length','Free-stream-velocity','Suction-thickness','SSPresure-level']
data = pd.read_csv('./datos/airfoil_self_noise.dat',names = names,sep="\t")


print("PREPROCESADO")


print("Matriz de correlación")
corr_matrix = data.corr()
k = 6 #number of variables for heatmap
cols = corr_matrix.nlargest(k, 'Frequency')['Frequency'].index
cm = np.corrcoef(data[cols].values.T)
plt.subplots(figsize=(9,9))
sns.set(font_scale=0.75)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

input("\n--- Pulsa una tecla para continuar ---\n")

print("Pairplot")

sns.set(font_scale=0.75)
sns.pairplot(data[names],height=1.65)
plt.show()

input("\n--- Pulsa una tecla para continuar ---\n")

print("Resumen estadístico")

tabla = data.describe()
print(tabla)

input("\n--- Pulsa una tecla para continuar ---\n")

plt.scatter(data['Frequency'],data['SSPresure-level'])
plt.xlabel("Frequency")
plt.title("Buscando outliers a través de Frequency")
plt.ylabel("SSPresure-level")
plt.show()
