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

# Parte 2

# Para hacerlo automáticamente con scikit-learn
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# Parte 3

equiespaciados = np.linspace(0, 2*math.pi,num=100)
cosins = []
sins = []
sum = []

for i in range(0, len(equiespaciados)):
    cosins.append(math.cos(equiespaciados[i]))
    sins.append(math.sin(equiespaciados[i]))
    sum.append(cosins[i]+sins[i])

cosins = np.array(cosins)
sins = np.array(sins)
sum = np.array(sum)

max_val = 2*math.pi
plt.plot(equiespaciados,cosins,'g--',label="coseno")
plt.plot(equiespaciados,sins,'b--',label="seno")
plt.plot(equiespaciados,sum,'r--',label="seno+coseno")
plt.xlabel("Variable independiente")
plt.ylabel("Variable dependiente")
plt.title("Parte 3")
plt.legend()
plt.axis([0,max_val,-3,3])
plt.show()
