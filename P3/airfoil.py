# Práctica 3 Aprendiza Automático 2019
# Autor: Luis Balderas Ruiz

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNetCV
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR



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


input("\n--- Pulsa una tecla para continuar ---\n")

# Gráfica de probabilidad
figura = plt.figure()
res = stats.probplot(data['SSPresure-level'],plot=plt)
plt.show()

input("\n--- Pulsa una tecla para continuar ---\n")

print("Asimetría de SSPresure-level")
print(skew(data['SSPresure-level']))

input("\n--- Pulsa una tecla para continuar ---\n")

print("Aplicando transformaciones")
#logarítmica para la asimetría positiva y transformación cuadrática para la negativa

data['Frequency'] = np.log(data['Frequency'])
# sqrt porque contiene 0
data['Angle-Attack'] = np.sqrt(data['Angle-Attack'])
data['Chord-Length'] = np.log(data['Chord-Length'])
data['Free-stream-velocity'] = np.log(data['Free-stream-velocity'])
data['Suction-thickness'] = np.log(data['Suction-thickness'])
data['SSPresure-level'] = np.square(data['SSPresure-level'])

input("\n--- Pulsa una tecla para continuar ---\n")

print("Normalización de los datos")

# Definición del objeto para escalar
scaler = StandardScaler()
# Calcula la media y la std de los datos
scaler.fit(data)
# Transforma los datos
data = scaler.transform(data)

input("\n--- Pulsa una tecla para continuar ---\n")

# Separación de datos

dataset = data
X = dataset[:,0:4]
Y = dataset[:,5]

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state = 77145416)


print("REGRESIÓN")


clf = LinearRegression().fit(X_train,y_train)


print("Coeficiente de R^2 asociado a la predicción dentro de la muestra")
print(clf.score(X_train,y_train))
input("\n--- Pulsa una tecla para continuar ---\n")
# E_out

prediccion = clf.predict(X_test)
print("Accuracy fuera de la muestra", clf.score(X_test,y_test))

input("\n--- Pulsa una tecla para continuar ---\n")

print("ElasticNet")

regr = ElasticNetCV(cv=10,random_state=77145416)
regr.fit(X_train,y_train)
print(regr.score(X_train,y_train))
en_pred = regr.predict(X_test)

print("Accuracy fuera de la muestra", regr.score(X_test,y_test))

input("\n--- Pulsa una tecla para continuar ---\n")

print("Redes neuronales")

redes = MLPRegressor()
redes.fit(X_train,y_train)
print(redes.get_params())
print("In R^2")
print(redes.score(X_train,y_train))

red_pred = redes.predict(X_test)
print("Out R^2")
print(redes.score(X_test,y_test))

input("\n--- Pulsa una tecla para continuar ---\n")

print("SVR")

svr = SVR(C=1,epsilon=0.2)
svr.fit(X_train,y_train)
print("In R^2")
print(svr.score(X_train,y_train))

svr_pred = svr.predict(X_test)
print("Out R^2")
print(svr.score(X_test,y_test))
