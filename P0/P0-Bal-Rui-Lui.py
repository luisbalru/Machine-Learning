# Práctica 0
# Autor: Luis Balderas Ruiz

# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math

# Parte 1

iris = load_iris()
X = iris.data
y = iris.target
fn = iris.feature_names
names = iris.target_names


last_2features = X[:,2:4]
setosa = []
versicolor = []
virginica = []
for i in range(0,len(y)):
    if y[i] == 0:
        setosa.append(last_2features[i])
    elif y[i] == 1:
        versicolor.append(last_2features[i])
    else:
        virginica.append(last_2features[i])

setosa = np.array(setosa)
versicolor = np.array(versicolor)
virginica = np.array(virginica)

plt.scatter(setosa[:,0],setosa[:,1],c='b',label="Setosa")
plt.scatter(versicolor[:,0],versicolor[:,1],c='r', label="Versicolor")
plt.scatter(virginica[:,0], virginica[:,1],c='g', label="Virginica")
#plt.scatter(last_2features[:,0], last_2features[:,1], c = y)
plt.legend()
plt.xlabel(fn[-2])
plt.ylabel(fn[-1])
plt.show()

input("Pulsa enter para seguir")

# Parte 2: Lo hago de dos formas distintas

# Para hacerlo automáticamente con scikit-learn
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# De forma manual, a través de la función random, genero una permutación
# (asumiendo que se utiliza una distribución uniforme) y me quedo con el 80% de
# los primeros datos para el training y el resto para el test

np.random.seed(77145416)
iris_l = list(zip(X,y))
iris_un = np.random.permutation(iris_l)
train = iris_un[0:int(0.8*len(iris_un))]
test = iris_un[int(0.8*len(iris_un)):]

# COMPROBACIÓN
print("Separación en training y test")
print('Número de instancias en training:', len(train))
print('Número de instancias en test:', len(test))

s=0
v=0
vir=0
for i in range(0,len(train)):
    if(train[i][1] == 0):
        s=s+1
    elif train[i][1] == 1:
        v = v+1
    else:
        vir = vir+1
# Por problemas de redondeo, no sale exactamente 33% en cada clase pero está realmente cerca:
# Entorno a 33%, 32%, 35%
print("#--------------------------#")
print("Distribución de clases en training")
print('Training. Setosa: ',s/len(train))
print('Training. Versicolor: ',v/len(train))
print('Training. Virginica: ',vir/len(train))

s=0
v=0
vir=0
for i in range(0,len(test)):
    if(test[i][1] == 0):
        s=s+1
    elif test[i][1] == 1:
        v = v+1
    else:
        vir = vir+1

# En test se exacerba el problema del redondeo al haber una proporción de datos mucho menor.
# De estos resultados se deduce que es mucho mejor utilizar funciones ya definidas como
# train_test_split de sklearn
print("#--------------------------#")
print("Distribución de clases en test")
print('Training. Setosa: ',s/len(test))
print('Training. Versicolor: ',v/len(test))
print('Training. Virginica: ',vir/len(test))

input("Pulsa enter para seguir")

# Parte 3

# Función para equiespaciar un número determinado de valores entre un límite inferior y otro superior
equiespaciados = np.linspace(0, 2*math.pi,num=100)
cosins = []
sins = []
sum = []


for i in range(0, len(equiespaciados)):
    cosins.append(math.cos(equiespaciados[i]))
    sins.append(math.sin(equiespaciados[i]))
    sum.append(cosins[i]+sins[i])

# Establezco el máximo valor en la variable independiente como 2pi
max_val = 2*math.pi
plt.plot(equiespaciados,cosins,'k--',label="coseno")
plt.plot(equiespaciados,sins,'b--',label="seno")
plt.plot(equiespaciados,sum,'r--',label="seno+coseno")
plt.xlabel("Variable independiente")
plt.ylabel("Variable dependiente")
plt.title("Parte 3")
plt.legend()
plt.axis([0,max_val,-3,3])
plt.show()
