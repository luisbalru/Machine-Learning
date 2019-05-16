# Práctica 3 Aprendiza Automático 2019
# Autor: Luis Balderas Ruiz

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats import skew

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

print("Outliers")

plt.scatter(data['Frequency'],data['SSPresure-level'])
plt.xlabel("Frequency")
plt.title("Buscando outliers a través de Frequency")
plt.ylabel("SSPresure-level")
plt.show()

input("\n--- Pulsa una tecla para continuar ---\n")

print("Eliminando posibles outliers")

data = data.drop(data[data['SSPresure-level']<105].index)
data = data.drop(data[data['SSPresure-level']>=140].index)

plt.scatter(data['Frequency'],data['SSPresure-level'])
plt.xlabel("Frequency")
plt.title("Tras la eliminación de outliers")
plt.ylabel("SSPresure-level")
plt.show()

input("\n--- Pulsa una tecla para continuar ---\n")

print("Distribuciones de las variables: Buscando asimetría y transformaciones")

plt.subplots(figsize=(9,9))
sns.distplot(data['Frequency']).set_title("Distribución de Frequency")
plt.show()
print("Asimetría de Frequency")
print(skew(data['Frequency']))

input("\n--- Pulsa una tecla para continuar ---\n")

plt.subplots(figsize=(9,9))
sns.distplot(data['Angle-Attack']).set_title("Distribución de Angle-Attack")
plt.show()
print("Asimetría de Angle-Attack")
print(skew(data['Angle-Attack']))

input("\n--- Pulsa una tecla para continuar ---\n")

plt.subplots(figsize=(9,9))
sns.distplot(data['Chord-Length']).set_title("Distribución de Chord-Length")
plt.show()
print("Asimetría de Chord-Length")
print(skew(data['Chord-Length']))

input("\n--- Pulsa una tecla para continuar ---\n")

plt.subplots(figsize=(9,9))
sns.distplot(data['Free-stream-velocity']).set_title("Distribución de Free-stream-velocity")
plt.show()
print("Asimetría de Free-stream-velocity")
print(skew(data['Free-stream-velocity']))

input("\n--- Pulsa una tecla para continuar ---\n")

plt.subplots(figsize=(9,9))
sns.distplot(data['Suction-thickness']).set_title("Distribución de Suction-thickness")
plt.show()
print("Asimetría de Suction-thickness")
print(skew(data['Suction-thickness']))

input("\n--- Pulsa una tecla para continuar ---\n")

print("Distribución de SSPresure-level")

# Distribución con histograma
plt.subplots(figsize=(9,9))
sns.distplot(data['SSPresure-level']).set_title("Distribución de SSPresure-level")
plt.show()
# Gráfica de probabilidad

input("\n--- Pulsa una tecla para continuar ---\n")

figura = plt.figure()
res = stats.probplot(data['SSPresure-level'],plot=plt)
plt.show()

input("\n--- Pulsa una tecla para continuar ---\n")

print("Skewness")
print(skew(data['SSPresure-level']))
