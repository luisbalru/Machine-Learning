# -*- coding: utf-8 -*-
"""
TRABAJO 2.
Nombre Estudiante: Luis Balderas Ruiz
Aprendizaje Automático 2019
"""
import numpy as np
import matplotlib.pyplot as plt
import warnings
#warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


# Fijamos la semilla
np.random.seed(7)


def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

def simula_gaus(N, dim, sigma):
    media = 0
    out = np.zeros((N,dim),np.float64)
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para
        # la primera columna se usará una N(0,sqrt(5)) y para la segunda N(0,sqrt(7))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)

    return out


def simula_recta(intervalo):
	points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
	x1 = points[0,0]
	x2 = points[1,0]
	y1 = points[0,1]
	y2 = points[1,1]
	a = (y2-y1)/(x2-x1)
	b = y1 - a*x1
	return a, b


def f_ej12(a,b,x,y):
	return np.sign(y-a*x-b)

"""
	@brief Dibuja los puntos coloreados según etiqueta y el ajuste lineal realizado
	@param x: Conjunto de datos
	@param y: etiquetas
	@param a: Pendiente del ajuste
	@param b: Ordenada en el origen
	@param titulo: Título para el plot
	@param eje_x: Etiqueta para el eje X
	@param eje_y: Etiqueta para el eje Y
"""
def dibujaRecta(x, y, a, b, titulo='Scatter plot de datos', eje_x='X', eje_y='Y'):
	X = []
	for i in range(len(x)):
		X.append(np.array([x[i][1],x[i][2]]))
	X = np.array(X)
	min_xy = X.min(axis=0)
	max_xy = X.max(axis=0)
	border_xy = (max_xy-min_xy)*0.01

    # Ajuste de los límites del dibujo y del plano necesario para representar de la forma
	# más visual posible los datos
	xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0],min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
	grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]

    #plot
	plt.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=2,cmap="RdYlBu", edgecolor='white', label='Datos')
	plt.plot(grid[:, 0], a*grid[:, 0]+b, 'black', linewidth=2.0, label='Ajuste lineal')
	plt.xlim(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0])
	plt.ylim(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1])
	plt.xlabel(eje_x)
	plt.ylabel(eje_y)
	plt.legend()
	plt.title(titulo)
	plt.show()

"""
#Ejercicio 1: SOBRE LA COMPLEJIDAD DE H Y EL RUIDO
"""


print("####################################################")
print("####################################################")
print("EJERCICIO 1: SOBRE LA COMPLEJIDAD DE H Y EL RUIDO")
print("####################################################")
print("####################################################")

print("Ejercicio 1. Dibujar una gráfica con la nube de puntos correspondiente")
print("a) N=50, dim=2, rango = [-50,50] con simula_unif")

listaN = simula_unif(50,2,[-50,50])
plt.scatter(listaN[:,0],listaN[:,1])
plt.title("Nube de puntos generada con simula_unif")
plt.show()

input("\n--- Pulsa una tecla para continuar ---\n")

print("b) N=50, dim=2, sigma=[5,7] con simula_gaus")

gauss = simula_gaus(50,2,np.array([5,7]))
plt.scatter(gauss[:,0],gauss[:,1])
plt.title("Nube de puntos generada con simula_gaus")
plt.show()

input("\n--- Pulsa una tecla para continuar ---\n")


print("Ejercicio 2")
# En primer lugar, simulo los parámetros a, b para la recta y = ax+b en [-50,50]
print("Apartado a)")
print("Calculando la pendiente (a) y la ordenada en el origen (b) de la recta...")
a,b = simula_recta([-50,50])
print("Pendiente (a): ", a)
print("Término independiente (b): ", b)
print("Generando la nube de 50 puntos uniformemente")
puntos = simula_unif(50,2,np.array([-50,50]))

print("Añadiendo las etiquetas correspondientes")
signos = f_ej12(a,b,puntos[:,0],puntos[:,1])
# Necesita tener la misma dimensión que puntos para poder hacer append
signos = signos.reshape(len(signos),1)
puntos_un = np.append(puntos,signos, axis=1)
print("Conjunto de datos a representar gráficamente")
print(puntos_un)
plt.scatter(puntos_un[:,0],puntos_un[:,1], c = puntos_un[:,2])
plt.plot([-50,50],[-50*a+b,50*a+b])
plt.title("Conjunto aleatorio de datos y recta que los separa")
plt.show()

input("\n--- Pulsa una tecla para continuar ---\n")

print("Apartado b)")
#Separo los datos por etiquetas
arriba = []
debajo = []
for i in range(len(puntos_un)):
	if puntos_un[i][2] == 1.0:
		arriba.append(puntos_un[i])
	else:
		debajo.append(puntos_un[i])

arriba = np.array(arriba)
debajo = np.array(debajo)
# Defino los índices en cada subconjunto para introducir ruido en el 10% de las tuplas
index1 = np.random.choice(len(arriba),int(0.1*len(arriba)),replace = False)
print(index1)
for i in index1:
	arriba[i][2] = -arriba[i][2]
#arriba[index1][:,2] = -arriba[index1][:,2]

index2 = np.random.choice(len(debajo), int(0.1*len(debajo)),replace = False)
print(index2)

for i in index2:
	debajo[i][2] = -debajo[i][2]
#debajo[index2][:,2] = -debajo[index2][:,2]

# Concatenación de los conjuntos y plot de los resultados. Ahora hay puntos
# mal clasificados
puntos_ruido = np.concatenate((arriba,debajo))
print("Modificación de  los puntos anteriores añadiendo ruido en el 10% de las etiquetas")
print(puntos_ruido)
plt.scatter(puntos_ruido[:,0],puntos_ruido[:,1], c = puntos_ruido[:,2])
plt.plot([-50,50],[-50*a+b,50*a+b])
plt.title("Introducción de ruido sobre el conjunto de datos.\n Ahora hay puntos mal clasificados")
plt.show()


input("\n--- Pulsa una tecla para continuar ---\n")

# Con la ayuda de la función contour dibujo las funciones implícitas pedidas

print("Ejercicio 3")
delta = 0.25
xrange = np.arange(-50, 50, delta)
yrange = np.arange(-50, 50, delta)
X, Y = np.meshgrid(xrange,yrange)

print("Representando f(x,y) = (x-10)**2+(y-20)**2 - 400")
F = (X-10)**2 + (Y-20)**2 - 400
plt.contour(X, Y, F , [0], colors=['red'])
plt.scatter(puntos_ruido[:,0],puntos_ruido[:,1], c = puntos_ruido[:,2])
plt.title("f(x,y) = (x-10)**2+(y-20)**2 - 400")
plt.show()

input("\n--- Pulsa una tecla para continuar ---\n")

print("Representando f(x,y) = 0.5(x+10)**2+(y-20)**2 - 400")
G = 0.5*(X+10)**2 + (Y-20)**2 - 400
plt.contour(X, Y, G , [0], colors=['red'])
plt.scatter(puntos_ruido[:,0],puntos_ruido[:,1], c = puntos_ruido[:,2])
plt.title("f(x,y) = 0.5(x+10)**2+(y-20)**2 - 400")
plt.show()

input("\n--- Pulsa una tecla para continuar ---\n")

print("Representando f(x,y) = 0.5(x-10)**2-(y+20)**2 - 400")
H = 0.5*(X-10)**2 - (Y+20)**2 - 400
plt.contour(X, Y, H , [0], colors=['red'])
plt.scatter(puntos_ruido[:,0],puntos_ruido[:,1], c = puntos_ruido[:,2])
plt.title("f(x,y) = 0.5(x-10)**2-(y+20)**2 - 400")
plt.show()

input("\n--- Pulsa una tecla para continuar ---\n")

# Cambiar rango
xrange = np.arange(-50, 50, delta)
yrange = np.arange(-50, 60, delta)
X, Y = np.meshgrid(xrange,yrange)
print("Representando f(x,y) = y-20x**2-5x+3")
I = Y-20*X**2-5*X+3
plt.contour(X, Y, I , [0], colors=['red'])
plt.scatter(puntos_ruido[:,0],puntos_ruido[:,1], c = puntos_ruido[:,2])
plt.title("f(x,y) = y-20x**2-5x+3")
plt.show()

input("\n--- Pulsa una tecla para continuar ---\n")


print("####################################################")
print("####################################################")
print("EJERCICIO 2: MODELOS LINEALES")
print("####################################################")
print("####################################################")

print("Definiendo la función ajusta_PLA...")

def ajusta_PLA(datos,label,vini,max_iter = -1):
	w = vini
	w_old = vini
	datos = np.concatenate((datos,np.ones((datos.shape[0],1),np.float64)),axis=1)
	it = 0
	changes = True
	if max_iter != -1:
		while(it < max_iter and changes):
			changes = False
			for i in range(len(datos)):
				if np.sign(w.dot(datos[i])) != label[i]:
					w_old = w
					w = w_old + label[i]*datos[i]
					changes = True
			it = it +1
	else:
		while(changes):
			changes = False
			for i in range(len(datos)):
				if np.sign(w.dot(datos[i])) != label[i]:
					w_old = w
					w = w_old + label[i]*datos[i]
					changes = True
			it = it +1
			w = np.array(w)
	return w,it



print("Ejercicio 1")

print("a) Ejecuto PLA con los datos de 2a de la sección anterior. Parámetros:")
print("a) 1. Vector 0")

# 92 iteraciones w [  36.85168076   47.132749    605.        ]
w21a1, it21a1 = ajusta_PLA(puntos,signos,np.zeros(puntos.shape[1]+1,np.float64))
print("w obtenido: ", w21a1)
print("Número de iteraciones necesarias para la convergencia: ", it21a1)

input("\n--- Pulsa una tecla para continuar ---\n")

print("a) 2. Vector de números aleatorios [0,1]")
# 130 iteraciones w  [  41.12023131   54.91502475  758.21062629]
w21a2, it21a2 = ajusta_PLA(puntos,signos,np.array([np.random.rand(),np.random.rand(),np.random.rand()]))
print("w obtenido: ",w21a2)
print("Número de iteraciones necesarias para la convergencia: ",it21a2)

input("\n--- Pulsa una tecla para continuar ---\n")

print("b) PLA con los datos de 2b de la sección anterior. Parámetros:")
print("1. Vector 0")
# No converge. 15000 iteraciones w [  22.7135259   -29.86849622  231.        ]
w21b1, it21b1 = ajusta_PLA(puntos,puntos_ruido[:,2],np.zeros(puntos.shape[1]+1,np.float64),15000)
print("w obtenido: ",w21b1)
print("Número de iteraciones necesarias para la convergencia: ",it21b1)

input("\n--- Pulsa una tecla para continuar ---\n")

print("2. Vector de números aleatorios [0,1]")
# No converge 15000 iteraciones w [  33.65887969  -42.0748706   243.0822101 ]
w21b2, it21b2 = ajusta_PLA(puntos,puntos_ruido[:,2],np.array([np.random.rand(),np.random.rand(),np.random.rand()],np.float64),15000)
print("w obtenido: ",w21b2)
print("Número de iteraciones necesarias para la convergencia: ",it21b2)

input("\n--- Pulsa una tecla para continuar ---\n")


# Función que utilizamos para la regresión logística
def sigmoide(x):
	return 1/(np.exp(-x)+1)

# REGRESIÓN LOGÍSTICA CON GRADIENTE DESCENDENTE ESTOCÁSTICO
# Parámetros:
#       - x: Conjunto de datos a evaluar
#       - y: Verdaderos valores de la etiqueta asociada a cada tupla
#       - learning_rate
#       - max_iters: Número máximo de iteraciones
#       - minibatch_size: Tamaño del minibatch
#       - epsilon: Cota del error
# Aclaraciones:
# 		Este algoritmo se distingue de SGD en la forma en la que se calcula
# 		el gradiente de E_in:
#			$\nabla E_{in} = -1/N \sum_{n=1}^{N} = \frac{y_n x_n}{1+ exp(y_n w^T(t)x_n)}


def sgd_logistic_regression(x,y,learning_rate,minibatch_size = 1, epsilon = 0.01, max_iters = 15000):
	x = np.hstack((np.ones(shape=(x.shape[0],1)),x))
	w = np.zeros(len(x[0]),np.float64)
	last_w = np.ones(len(w),np.float64)
	indices = np.array(range(0,x.shape[0]))
	np.random.shuffle(indices)
	it = 0
	i=0
	while np.linalg.norm(w-last_w) > epsilon and it < max_iters:
		for j in range(int(len(x)/minibatch_size)-1):
			X_minib = x[indices[j*minibatch_size:(j+1)*minibatch_size:1],:]
			Y_minib = y[indices[j*minibatch_size:(j+1)*minibatch_size:1]]
			last_w = w
			X_minib = np.array(X_minib)
			Y_minib = np.array(Y_minib)
			suma = -1/len(x[0])*Y_minib*X_minib*(sigmoide(-Y_minib*X_minib.dot(last_w))[0])
			suma = suma[0]
			w = w - learning_rate * suma
			i=j
		if len(x) % minibatch_size != 0:
			resto = (len(x) % minibatch_size)*minibatch_size
			indices = np.append(indices[-resto:],indices[:minibatch_size-resto])
			last_w = w
			X_minib = x[indices[i*minibatch_size:(i+1)*minibatch_size:1],:]
			Y_minib = y[indices[i*minibatch_size:(i+1)*minibatch_size:1]]
			X_minib = np.array(X_minib)
			Y_minib = np.array(Y_minib)
			suma = -1/len(x[0])*Y_minib*X_minib*(sigmoide(-Y_minib*X_minib.dot(last_w))[0])
			suma = suma[0]
			w = w - learning_rate * suma
		np.random.shuffle(indices)
		it = it+1
	return w

def reetiquetar(y):
	y_ = np.array(y)
	for i in range(0, y_.size):
		if y_[i] == -1:
			y_[i] = 0
	return y_


print("Ejercicio 2")
print("Generando 100 puntos en [0,2]x[0,2]")
puntos2 = simula_unif(100,2,[0,2])
print("Simulando recta que los separa...")
a,b = simula_recta([0,2])
print("Pendiente (a): ", a)
print("Término independiente (b): ", b)

input("\n--- Pulsa una tecla para continuar ---\n")

print("Añadiendo las etiquetas correspondientes")
etiquetas = f_ej12(a,b,puntos2[:,0],puntos2[:,1])
etiquetas = etiquetas.reshape(len(etiquetas),1)
y = reetiquetar(etiquetas)
p = np.append(puntos2,etiquetas, axis=1)
plt.scatter(puntos2[:,0],puntos2[:,1],c=p[:,2])
plt.plot([0,2],[b,(2*a+b)])
plt.ylim([0,2])
plt.title("Regresión logística")
plt.show()

input("\n--- Pulsa una tecla para continuar ---\n")

print("Ejecutando SGD con Regresión Logística")
print("Parámetros: Learning_rate 0.01, epsilon 0.01")

w = sgd_logistic_regression(puntos2,etiquetas,15000,64)
print("El valor de w es: ", w)
# [[ 0.04156455  0.08628088  0.08090566]]


def error_acierto(X,y,w):
	x = np.concatenate((X, np.ones((X.shape[0], 1), np.float64)), axis=1)
	tam = y.size
	suma = 0

	for i in range(0,tam):
		if np.abs(sigmoide(x[i].dot(w.T))-y[i]) > 0.5:
			suma += 1

	return suma/tam


print("Ein: ", error_acierto(puntos2,y,w))


plt.scatter(puntos2[:,0],puntos2[:,1],c=p[:,2])
plt.plot([0,2], [-w[0]/w[2], -w[0]/w[2]-2*w[1]/w[2]])
plt.title("Recta generada con el w de la RL-SGD")
plt.show()


###############################################################################
###############################################################################
###############################################################################
###############################################################################

print('EJERCICIO BONUS\n')

label4 = 1
label8 = -1

# Funcion para leer los datos
def readData(file_x, file_y):
	# Leemos los ficheros
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []
	# Solo guardamos los datos cuya clase sea la 4 o la 8
	for i in range(0,datay.size):
		if datay[i] == 4 or datay[i] == 8:
			if datay[i] == 4:
				y.append(label4)
			else:
				y.append(label8)
			x.append(np.array([1, datax[i][0], datax[i][1]]))

	x = np.array(x, np.float64)
	y = np.array(y, np.float64)

	return x, y


# PSEUDOINVERSA
# Calcula la pseudoinversa (Moore-Penrose) de una matriz. Se basa en la utilización
# de la descomposición en valores singulares (SVD) vista en teoría.
# En general, la descomposición en valores singulares de una matriz devuelve una expresióm
# del tipo X=UDV^t, siendo U, V matrices ortogonales y D una matriz (no necesariamente cuadrada)
# que contiene en la diagonal los valores singulares de X. En el intento de calcular
# (X^tX)^(-1) = VD^tDV^t, por lo que se compatibiliza el producto de las "matrices diagonales"
# En el caso de Python, np.linalg.svd devuelve la matriz U (array2D), D (array1D con los valores singulares)
# y V_t (array2D).
# Parámetros: x --> Dataset (matriz)
#             y --> valores reales para la clasificación
# Return: Coeficientes calculados

def pseudoinverse(x,y):
    U,D,V_t = np.linalg.svd(x)
    # D es un array1D con los valores singulares (distinto de 0), luego es posible
    # calcular la inversa. En este caso, el inverso de cada elemento de la diagonal
    inverse_D = np.linalg.inv(np.diag(D))
    V = V_t.transpose()
    # El producto D^t D teórico genera una matriz cuadrada que coincide con el producto
    # matricial de dos matrices iguales cuya diagonal contiene a los valores singulares
    inverse_X = V.dot(inverse_D).dot(inverse_D).dot(V.transpose()).dot(x.transpose())
    w = inverse_X.dot(y)
    return w


#
# errorBonus
# @brief: Cálculo del error. Para ello contabilizo la proporción de individuos
# 		  mal clasificados (np.sign(X[i].dot(w)) != y[i]) respecto del total.
# @param X: conjunto de datos
# @param y: etiquetas
# @param w: w ajustado
# @return Proporción de individuos mal clasificados
#

def errorBonus(X,y,w):
	tam = y.size
	suma = 0

	for i in range(0,tam):
		if np.sign(X[i].dot(w)) != y[i]:
			suma += 1

	return suma/tam

#
# PLA_pocket
# @brief: Modificación de PLA en la que se va acumulando el mejor error conseguido
#		  hasta el momento (y también la mejor w asociada a ese error)
# @param x Conjunto de datos
# @param y Etiquetas
# @param max_iters
# @param vini w inicial
#

def PLA_pocket(x,y,max_iters,vini):
	w = vini
	w_best = w
	# Comienzo con el error
	mejor_error = errorBonus(x,y,w)

	for i in range(max_iters):
		change = True
		for j in range(y.size):
			if np.sign(w.dot(x[j])) != y[j]:
				w = w + y[j]*x[j]
				change = False
		if change:
			break
		else:
			error_actual = errorBonus(x,y,w)
			# Actualizo el error si se ha encontrado uno menor
			if error_actual < mejor_error:
				mejor_error = error_actual
				w_best = np.array(w)

	return w_best

print("Leyendo los datos para el problema de clasificación")

x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')

input("\n--- Pulsa una tecla para continuar ---\n")

print("Ejecutando Pseudoinversa como ajuste lineal y PLA-Pocket para extraer resultados")
# Se pedía un ajuste lineal. Como en la práctica anterior Pseudoinversa tuvo
# muy buenos resultados, la utilizo aquí para calcular el w inicial a pasarle
# al PLA_pocket
w_ini = pseudoinverse(x,y)

# Ejecuto PLA_pocket con 1000 iteraciones y w_ini la salida de Pseudoinversa
w = PLA_pocket(x, y, 1000, w_ini)

print("w obtenida: ", w)

input("\n--- Pulsa una tecla para continuar ---\n")

print("Dibujo las rectas obtenidas juunto al conjunto de datos de training y test")

# Dibujo el conjunto de entrenamiento junto a la recta generada por medio del w
# calculado en PLA_pocket. Téngase en cuenta que -w[1]/w[2] y -w[0]/w[2] son
# la pendiente de la recta y la ordenada en el origen, respectivamente
dibujaRecta(x, y, -w[1]/w[2], -w[0]/w[2], 'Training de PLA_pocket', 'Intensidad', 'Simetria')

input("\n--- Pulsa una tecla para continuar ---\n")

# Misma representación pero en el conjunto de test
dibujaRecta(x_test, y_test, -w[1]/w[2], -w[0]/w[2], 'Test de PLA_pocket', 'Intensidad', 'Simetria')

input("\n--- Pulsa una tecla para continuar ---\n")

print("A continuación, calculamos los errores cometidos en training y test")

# Calculo el error cometido en el conjunto de entrenamiento
E_in = errorBonus(x,y,w)
# Calculo el error cometido en test
E_test = errorBonus(x_test,y_test,w)

print("Error medio en entrenamiento y test")
print("Error interior medio: ",E_in)
print("Error exterior medio: ",E_test)

input("\n--- Pulsa una tecla para continuar ---\n")

print("Por último, calculamos las cotas de error fuera de la muestra a través")
print("de la interpretación de la desigualdad de Hoeffding y su adaptación")
print("vía la dimensión de Vapnik-Chervonenkis (3 por ser lineal)")

cotaein = E_in + np.sqrt((8/y.size)*np.log(4*((2*y.size)**3 + 1)/0.05))
cotaetest = E_test + np.sqrt((8/y_test.size)*np.log(4*((2*y_test.size)**3 +1 )/0.05))

print ('\nCotas sobre el valor del error fuera de la muestra\n')
print ('Cota basada en Ein:', cotaein)
print ('Cota basada en Etest:', cotaetest)
