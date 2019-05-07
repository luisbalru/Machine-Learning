# Práctica 3 Aprendiza Automático 2019
# Autor: Luis Balderas Ruiz


import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import cm as cm
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

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


print("Descripción estadística de las variables")
tabla = training.describe()
print(tabla)



matriz = training.values
X_train_pca = matriz[:,0:63]
Y_train_pca = matriz[:,64]
matriz_test = test.values
X_test = matriz_test[:,0:63]
Y_test = matriz_test[:,64]

print("Represento la matriz de correlación de las características")
correlation_matrix(training)


# PCA

# Necesidad de normalizar
scaler = StandardScaler()
scaler.fit(X_train_pca)
X_train_pca = scaler.transform(X_train_pca)
X_test_pca = scaler.transform(X_test)
pca = PCA(0.95, svd_solver='full')
pca.fit(X_train_pca)

X_train_pca1 = pca.transform(X_train_pca)
X_test_pca1 = pca.transform(X_test_pca)
print(pca.n_components_)


# REGRESIÓN LOGÍSTICA
logisticRegr = LogisticRegression(solver = 'lbfgs')
logisticRegr.fit(X_train_pca1, Y_train_pca)
prediccion = logisticRegr.predict(X_test_pca1)
print(logisticRegr.score(X_test_pca1, Y_test))
