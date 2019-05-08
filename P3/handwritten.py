# Práctica 3 Aprendiza Automático 2019
# Autor: Luis Balderas Ruiz


import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import cm as cm
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn import linear_model
from sklearn.cross_validation import StratifiedKFold

def correlation_matrix(df):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Correlación de las características')
    fig.colorbar(cax, ticks=[-.6,-.5,-.4,-.3,-.2,-.1,0,.1,.2,.3,.4,.5,.6,.7,.8,.90,1])
    plt.show()


print("Recognition of Handwritten Digits")

print("Realizo la lectura de los datos")
training = pd.read_csv('./datos/optdigits.tra',header=None)
test = pd.read_csv('./datos/optdigits.tes', header = None)


print("PREPROCESADO")

print("Veamos si las clases están balanceadas")
clases = sns.countplot(training[64])
clases.plot()

input("\n--- Pulsa una tecla para continuar ---\n")

print("Descripción estadística de las variables")
tabla = training.describe()
print(tabla)

input("\n--- Pulsa una tecla para continuar ---\n")


matriz = training.values
X_train_pca = matriz[:,0:63]
Y_train_pca = matriz[:,64]
matriz_test = test.values
X_test = matriz_test[:,0:63]
Y_test = matriz_test[:,64]

print("Represento la matriz de correlación de las características")
correlation_matrix(training)

input("\n--- Pulsa una tecla para continuar ---\n")

print("Matriz de correlación completa")
print(training.corr())
input("\n--- Pulsa una tecla para continuar ---\n")


# PCA

# Necesidad de normalizar para garantizar el buen funciónamiento de PCA
# La normalización es necesaria tanto en training como en test

print("Escalado estándar: preparación de variables para PCA y clasificación")
scaler = StandardScaler()
scaler.fit(X_train_pca)
X_train_pca = scaler.transform(X_train_pca)
X_test_pca = scaler.transform(X_test)

input("\n--- Pulsa una tecla para continuar ---\n")
# 0.95 y 'full' para garantizar el mínimo número de componentes principales
# con una varianza explicada mayor del 95%

print("PCA: 0.95 Y svd_solver = full")
pca = PCA(0.95, svd_solver='full')
pca.fit(X_train_pca)

# Aplico las transformaciones tanto a training como a test
X_train_pca1 = pca.transform(X_train_pca)
X_test_pca1 = pca.transform(X_test_pca)

input("\n--- Pulsa una tecla para continuar ---\n")

print("Número de componentes principales:", pca.n_components_)

input("\n--- Pulsa una tecla para continuar ---\n")

"""
# REGRESIÓN LOGÍSTICA
logisticRegr = LogisticRegression(solver = 'lbfgs')
logisticRegr.fit(X_train_pca1, Y_train_pca)
prediccion = logisticRegr.predict(X_test_pca1)
print(logisticRegr.score(X_test_pca1, Y_test))
"""

print("CLASIFICACIÓN")

# SGDClassifier

print("Caso 1: SGDClassifier")

print("Validación cruzada estratificada con 10 particiones para garantizar los resultados")
# Validación cruzada
skf = StratifiedKFold(Y_train_pca, n_folds = 10)
scores = []
for train_index, test_index in skf:
    clf = linear_model.SGDClassifier(max_iter=10000, tol = 1e-6)
    X_train = X_train_pca1[train_index]
    Y_train = Y_train_pca[train_index]
    X_test1 = X_train_pca1[test_index]
    Y_test1 = Y_train_pca[test_index]
    clf.fit(X_train,Y_train)
    scores.append(clf.score(X_test1,Y_test1))
    
print(scores)

# E_out

print("Accuracy fuera de la muestra", clf.score(X_test_pca1,Y_test))


