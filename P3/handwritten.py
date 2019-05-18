# Práctica 3 Aprendiza Automático 2019
# Autor: Luis Balderas Ruiz


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import cm as cm
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn import linear_model
from sklearn.model_selection import StratifiedKFold
from scipy import stats
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

#
# correlation_matrix
# @brief: Función para ver gráficamente la matriz de correlación. Los colores implican
#   el nivel de correlación (véase lateral derecho de la gráfica)
# @param df. Matriz de datos para calcular correlación
#
def correlation_matrix(df):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Correlación de las características')
    fig.colorbar(cax, ticks=[-.6,-.5,-.4,-.3,-.2,-.1,0,.1,.2,.3,.4,.5,.6,.7,.8,.90,1])
    plt.show()


# detect_outlier
# @brief: Detector rudimentario de outliers. Se define un límite, se calcula la media y la desviación
#   típica de todo el dataset y se calcula z_score de la media de cada instancia. Si z_score
#   en valor absoluto es mayor que el límite definido, se dice que esa instancia es un outlier.
# @param data. Dataset completo
def detect_outlier(data):
    outliers = []
    lim = 3
    media = np.mean(data)
    std = np.std(data)
    for i in range(len(data)):
        media_i = np.mean(data[i])
        z = (media_i - media)/std
        if np.abs(z) > lim:
            outliers.append(i)
    return outliers

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
           xlabel='Etiquetas predichas')

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


print("Recognition of Handwritten Digits")

print("Realizo la lectura de los datos")
training = pd.read_csv('./datos/optdigits.tra',header=None)
test = pd.read_csv('./datos/optdigits.tes', header = None)


print("PREPROCESADO")

# Balanceo de clases con sns
print("Veamos si las clases están balanceadas")
clases = sns.countplot(training[64])
clases.plot()

input("\n--- Pulsa una tecla para continuar ---\n")


# Variabilidad estadística de las características
print("Descripción estadística de las variables")
tabla = training.describe()
print(tabla)


input("\n--- Pulsa una tecla para continuar ---\n")

# Organización en training y test temporal (luego se harán divisiones con validación cruzada)
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


print("Detección de outliers")
index_outlier = []
index_outlier = detect_outlier(matriz)
if len(index_outlier) == 0:
    print("No se han encontrado outliers")
else:
    print("Se han detectado los siguientes outliers (índices): ", index_outlier)

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


nombres = ['0','1','2','3','4','5','6','7','8','9']



print("CLASIFICACIÓN")

# SGDClassifier

print("Caso 1: SGDClassifier")

print("Validación cruzada estratificada con 10 particiones para garantizar los resultados")
# Validación cruzada
skf = StratifiedKFold(n_splits = 10,shuffle = True)

scores = []
for train_index, test_index in skf.split(X_train_pca1,Y_train_pca):
    clf = linear_model.SGDClassifier(max_iter=10000, tol = 1e-6)
    X_train = X_train_pca1[train_index]
    Y_train = Y_train_pca[train_index]
    X_test1 = X_train_pca1[test_index]
    Y_test1 = Y_train_pca[test_index]
    clf.fit(X_train,Y_train)
    scores.append(clf.score(X_test1,Y_test1))

print("Valores de las clasificaciones en la validación cruzada:")
print(scores)


input("\n--- Pulsa una tecla para continuar ---\n")

print("Descripción estadística de los resultados de la validación cruzada:")
print(stats.describe(scores))


input("\n--- Pulsa una tecla para continuar ---\n")
# E_out

prediccion = clf.predict(X_test_pca1)
print("Accuracy fuera de la muestra", clf.score(X_test_pca1,Y_test))

input("\n--- Pulsa una tecla para continuar ---\n")

print("Matriz de confusión")

np.set_printoptions(precision=2)
plot_confusion_matrix(Y_test, prediccion, classes=nombres,normalize = True,title='Matriz de confusión para SGD')
plt.show()

input("\n--- Pulsa una tecla para continuar ---\n")

print("Resumen de la clasificación:")

print(classification_report(Y_test,prediccion,target_names = nombres))

input("\n--- Pulsa una tecla para continuar ---\n")

# REGRESIÓN LOGÍSTICA

print("Regresión logística")


print("Validación cruzada estratificada con 10 particiones para garantizar los resultados")
# Validación cruzada

logisticRegr = LogisticRegressionCV(cv=10, random_state = 77145416, multi_class = 'multinomial', max_iter=2500)
logisticRegr.fit(X_train_pca1,Y_train_pca)

input("\n--- Pulsa una tecla para continuar ---\n")

# Predicción fuera de la muestra
prediccion = logisticRegr.predict(X_test_pca1)


print("Accuracy fuera de la muestra", logisticRegr.score(X_test_pca1,Y_test))

input("\n--- Pulsa una tecla para continuar ---\n")

print("Matriz de confusión")

np.set_printoptions(precision=2)
plot_confusion_matrix(Y_test, prediccion, classes=nombres,normalize = True,title='Matriz de confusión para Regresión Logística')
plt.show()

input("\n--- Pulsa una tecla para continuar ---\n")

print("Resumen de la clasificación")

print(classification_report(Y_test,prediccion,target_names = nombres))


input("\n--- Pulsa una tecla para continuar ---\n")

print("Clasificación con SVM")

C_range = np.logspace(-2,10,4)
gamma_range = np.logspace(-9,3,4)
kernel = ['linear','rbf']
param_grid = dict(gamma=gamma_range, C=C_range, kernel = kernel)

# Defino arrays para almacenar las distintas configuraciones y luego elegir la mejor
configuraciones = []
resultados = []

for c in C_range:
    for g in gamma_range:
        for k in kernel:
            skf = StratifiedKFold(n_splits = 10,shuffle = True)
            scores = []
            for train_index, test_index in skf.split(X_train_pca1,Y_train_pca):
                svmp = SVC(C=c, gamma = g, kernel= k)
                X_train = X_train_pca1[train_index]
                Y_train = Y_train_pca[train_index]
                X_test1 = X_train_pca1[test_index]
                Y_test1 = Y_train_pca[test_index]
                svmp.fit(X_train,Y_train)
                scores.append(svmp.score(X_test1,Y_test1))
            configuraciones.append([c,g,k])
            resultados.append(scores)

print(resultados)
mean_results = []

for i in range(len(resultados)):
    mean_results.append(np.mean(resultados[i]))

pos = np.argmax(mean_results)

print("Mejor configuración tras el GridSearch con Validación Cruzada de 10 particiones:")
print("C: ",configuraciones[pos][0])
print("gamma: ",configuraciones[pos][1])
print("kernel: ", configuraciones[pos][2])

input("\n--- Pulsa una tecla para continuar ---\n")

print("Resultados finales de la clasificación con SVM y mejores parámetros")
svm = SVC(C = configuraciones[pos][0], gamma = configuraciones[pos][1], kernel = configuraciones[pos][2])
svm.fit(X_train_pca1,Y_train_pca)
y_pred = svm.predict(X_test_pca1)

print("Accuracy fuera de la muestra: ", svm.score(X_test_pca1,Y_test))

input("\n--- Pulsa una tecla para continuar ---\n")

print("Matriz de confusión")

np.set_printoptions(precision=2)
plot_confusion_matrix(Y_test, y_pred, classes=nombres,normalize = True,title='Matriz de confusión para SVM')
plt.show()

input("\n--- Pulsa una tecla para continuar ---\n")

print("Resumen de la clasificación")

print(classification_report(Y_test,y_pred,target_names = nombres))
