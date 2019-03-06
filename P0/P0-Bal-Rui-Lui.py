# Práctica 0
# Autor: Luis Balderas Ruiz
# Aprendizaje Automático 2019
# Doble Grado en Ingeniería Informática y Matemáticas

# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math

# Parte 1
print("PARTE 1")


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

# Doy estructura de array a los distintos contenedores
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
print("PARTE 2")

# Para hacerlo automáticamente con scikit-learn
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# De forma manual, a través de la función random, genero una permutación
# (asumiendo que se utiliza una distribución uniforme) y clasifico los elementos
# de dicha permutación. Mientras que haya menos de 40 instancias para cada clase en training,
# las instancias se almacenan en train. Si no, irá a test.

#Fijo semilla para poder reproducir los casos
np.random.seed(77145416)

# Unión de datos y etiquetas
iris_l = list(zip(X,y))
# Permutación del dataset para conseguir la aleatoriedad en la partición
iris_un = np.random.permutation(iris_l)
train = []
test = []
v_tr=0
s_tr=0
vir_tr=0

for i in range(0, len(iris_un)):
    if iris_un[i][1] == 0:
        if s_tr < 40:
            train.append(iris_un[i])
            s_tr = s_tr + 1
        else:
            test.append(iris_un[i])
    elif iris_un[i][1] == 1:
        if v_tr < 40:
            train.append(iris_un[i])
            v_tr = v_tr + 1
        else:
            test.append(iris_un[i])
    elif iris_un[i][1] == 2:
        if vir_tr < 40:
            train.append(iris_un[i])
            vir_tr = vir_tr + 1
        else:
            test.append(iris_un[i])

# COMPROBACIÓN
print("Separación en training y test")
print('Número de instancias en training:', len(train))
print('Número de instancias en test:', len(test))

train = np.array(train)
test = np.array(test)

# Hago un conteo de los elementos únicos en la columna de la etiqueta de clase (es decir, las distintas clases que hay)
# y el número de repeticiones de cada una. Para la representación por pantalla utilizo names (nombre de las clases)
unique_elements, count_elements = np.unique(train[:,1],return_counts=True)

# Resultados en training
print("#--------------------------#")
print("Distribución de clases en training")
print(dict(zip(names,count_elements)))

unique_elements, count_elements = np.unique(test[:,1],return_counts=True)

# Resultados en test
print("#--------------------------#")
print("Distribución de clases en test")
print(dict(zip(names,count_elements)))

input("Pulsa enter para seguir")

# Parte 3

print("PARTE 3")

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
# Rango de X: 0,2*pi. Rango de Y: -3 a 3
plt.axis([0,max_val,-3,3])
plt.show()
