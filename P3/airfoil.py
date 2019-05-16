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
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LinearRegression



#
# plot_confusion_matrix
# @brief: Función encargada de computar y preparar la impresión de la matriz de confusión. Se puede extraer los resultados normalizados o sin normalizar. Basada en un ejemplo de scikit-learn
# @param: y_true. Etiquetas verdaderas
# @param: y_pred. Etiquetas predichas
# @param: classes. Distintas clases del problema (vector)
# @param: normalize. Booleano que indica si se normalizan los resultados o no
# @param: title. Título del gráfico
# @param: cmap. Paleta de colores para el gráfico
#

def plot_confusion_matrix(y_true, y_pred, classes,normalize=False,title=None,cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Matriz de confusión normalizada'
        else:
            title = 'Matriz de confusión sin normalizar'

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    # Clases
    classes = [0,1,2,3,4,5,6,7,8,9]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Matriz de confusión normalizada")
    else:
        print('Matriz de confusión sin normalizar')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Etiquetas verdaderas',
           xlabel='Etiquetas predecidas')

    # Rotar las etiquetas para su posible lectura
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Creación de anotaciones
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax






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

print("REGULARIZACIÓN")
"""
print("Lasso")
print("Validación cruzada de 10 particiones para ajustar hiperparámetros")

reg = LassoCV(cv=10,random_state=77145416).fit(X_train,y_train)

print("Parámetros obtenidos")
print(reg.get_params())
print("Score para el ajuste en CV")
print(reg.score(X_train,y_train))

lasso_pred = reg.predict(X_test)

print("Accuracy fuera de la muestra: ", reg.score(X_test,y_test))

input("\n--- Pulsa una tecla para continuar ---\n")

print("Matriz de confusión")

np.set_printoptions(precision=2)
plot_confusion_matrix(y_test, lasso_pred, classes=names,normalize = True,title='Matriz de confusión para Regularización Lasso')
plt.show()

input("\n--- Pulsa una tecla para continuar ---\n")

print("Resumen de la clasificación")

print(classification_report(y_test,lasso_pred,target_names = names))
"""

print("REGRESIÓN")

print("Validación cruzada estratificada con 10 particiones para garantizar los resultados")
# Validación cruzada
skf = StratifiedKFold(y_train, n_folds = 10,shuffle = True)

scores = []
for train_index, test_index in skf:
    clf = LinearRegression()
    X_train = X_train[train_index]
    Y_train = y_train[train_index]
    X_test1 = X_train[test_index]
    Y_test1 = y_train[test_index]
    clf.fit(X_train,Y_train)
    scores.append(clf.score(X_test1,Y_test1))

print("Valores de las clasificaciones en la validación cruzada:")
print(scores)


input("\n--- Pulsa una tecla para continuar ---\n")

print("Descripción estadística de los resultados de la validación cruzada:")
print(stats.describe(scores))


input("\n--- Pulsa una tecla para continuar ---\n")
# E_out

prediccion = clf.predict(X_test)
print("Accuracy fuera de la muestra", clf.score(X_test,y_test))

input("\n--- Pulsa una tecla para continuar ---\n")

print("Matriz de confusión")

np.set_printoptions(precision=2)
plot_confusion_matrix(y_test, prediccion, classes=names,normalize = True,title='Matriz de confusión para Regresión lineal')
plt.show()

input("\n--- Pulsa una tecla para continuar ---\n")

print("Resumen de la clasificación:")

print(classification_report(y_test,prediccion,target_names = names))

input("\n--- Pulsa una tecla para continuar ---\n")
