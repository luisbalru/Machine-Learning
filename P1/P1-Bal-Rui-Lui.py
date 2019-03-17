# -*- coding: utf-8 -*-

"""
Práctica 1. Aprendizaje Automático
Doble Grado en Ingeniería Informática y Matemáticas. Universidad de Granada

@author: Luis Balderas Ruiz
GNU GENERAL PUBLIC LICENSE
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from sympy import symbols, diff
from sympy.functions import exp, sin


# Fijo la semilla para garantizar la reproducibilidad de los resultados
np.random.seed(77145416)

"""
Ejercicio 1: Ejercicio sobre la búsqueda iterativa de óptimos.
"""

"""
# ALGORITMO DE GRADIENTE DESCENDENTE
# - El parámetro E se refiere a una empresión de la librería de cálculo simbólico
#   sympy para calcular las derivadas parciales y generalizar la función lo máximo posible.
# - gradient es una función que calcula el gradiente. Es necesario definir esa función
#   para cada caso que queramos considerar, puesto que depende directamente del nombre
#   y número de variables del caso en cuestión.
# - f es la función que evalúa la expresión en los diferentes valores de w. También hay que
#   definirla en cada caso.

def GD(E,w,learning_rate,gradient,f,epsilon, max_iters = 15000000):
    num_iter = 0
    diferencia = 1
    medidas = [f(w)]
    while num_iter <= max_iters and diferencia > epsilon:
        last_w = w
        w = w - learning_rate * gradient(last_w)
        diferencia = abs(f(w) - f(last_w))
        num_iter = num_iter + 1
        medidas.insert(len(medidas),f(w))

    return w, num_iter, medidas


################## Ejercicio 1.2 #################################

# Definición de la función matemática a considerar en el 1.2. Asociada a ella,
# las correspondientes que calculan el gradiente y evalúa.

u, v = symbols('u v', real=True)
E = (u**2*exp(v)-2*v**2*exp(-u))**2

def gradientE(w):
    der_u = diff(E,u)
    der_v = diff(E,v)
    return np.array([der_u.subs([(u,w[0]),(v,w[1])]),der_v.subs([(u,w[0]),(v,w[1])])])

def evalE(w):
    return E.subs([(u,w[0]),(v,w[1])])

print("#################################################")
print("EJERCICIO 1.2")
# Apartado a)
print("Apartado a)")
print("El gradiente de E es: ", diff(E,u),",",diff(E,v))
input("\n--- Pulsa una tecla para continuar ---\n")

# Apartado b)
print("Apartado b)")
w,k,data = GD(E,np.array([1.0,1.0],np.float64),0.01,gradientE,evalE,10**(-14))
print("El número de iteraciones necesarias para epsilon = 10**(-14) es ",k)
input("\n--- Pulsa una tecla para continuar ---\n")

# Apartado c)
print("Apartado c)")
print("COORDENADAS:")
print("Coordenada X: ", w[0])
print("Coordenada Y: ", w[1])
input("\n--- Pulsa una tecla para continuar ---\n")


################## Ejercicio 1.3 #################################

x, y = symbols('x y', real=True)
f = x**2 + 2*y**2 + 2*sin(2*math.pi*x)*sin(2*math.pi*y)

def gradientf(w):
    der_x = diff(f,x)
    der_y = diff(f,y)
    return np.array([der_x.subs([(x,w[0]),(y,w[1])]),der_y.subs([(x,w[0]),(y,w[1])])])

def evalf(w):
    return f.subs([(x,w[0]),(y,w[1])])

print("#################################################")
print("EJERCICIO 1.3")
# Apartado a)
print("Apartado a)")

w,k,data = GD(f,np.array([0.1,0.1],np.float64),0.01,gradientf,evalf,10**(-20),50)
print("Coordenadas del mínimo: ", w)
plt.plot(range(0,k+1),data,'bo')
plt.xlabel('Número de iteraciones')
plt.ylabel('f(x,y)')
plt.show()
input("\n--- Pulsa una tecla para continuar ---\n")

## para learning_rate = 0.1
print("Learning rate 0.1")
w,k,data = GD(f,np.array([0.1,0.1],np.float64),0.1,gradientf,evalf,10**(-20),50)
plt.plot(range(0,k+1),data,'bo')
plt.xlabel('Número de iteraciones')
plt.ylabel('f(x,y)')
plt.show()
input("\n--- Pulsa una tecla para continuar ---\n")

# Apartado b)

print("Apartado b)")

datos = []
array = np.array([(0.1,0.1),(1,1),(-0.5,-0.5),(-1,-1)])
for n in array:
    w,k,data = GD(f,n,0.01,gradientf,evalf,10**(-20),50)
    datos.append([w,evalf(w)])

datos = np.array(datos)

for i in range(0,len(datos)):
    print("Punto de inicio: ", array[i])
    print('(x,y) = ', datos[i][0])
    print('Valor mínimo: ',datos[i][1])
"""

#print(datos)

"""
Ejercicio 2. Regresión lineal
"""

################# Ejercicio 1 ##########################

print('EJERCICIO SOBRE REGRESION LINEAL\n')
print('Ejercicio 1\n')

label5 = 1
label1 = -1

# Funcion para leer los datos
def readData(file_x, file_y):
	# Leemos los ficheros
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []
	# Solo guardamos los datos cuya clase sea la 1 o la 5
	for i in range(0,datay.size):
		if datay[i] == 5 or datay[i] == 1:
			if datay[i] == 5:
				y.append(label5)
			else:
				y.append(label1)
			x.append(np.array([1, datax[i][0], datax[i][1]]))

	x = np.array(x, np.float64)
	y = np.array(y, np.float64)

	return x, y

# FUNCIÓN PARA CALCULAR EL ERROR
# Efectúa el producto escalar de x y w, lo que genera la salida
# de la función lineal que hemos interpolado. Después, calcula la
# norma al cuadrado de la diferencia entre la salida estimada y la real.
# Finalmente, divide por el número de datos
def Err(x,y,w):
    return (1/y.size)*np.linalg.norm(x.dot(w)-y)**2

# GRADIENTE DESCENDENTE ESTOCÁSTICO
# Parámetros:
#       - x: Conjunto de datos a evaluar
#       - y: Verdaderos valores de la etiqueta asociada a cada tupla
#       - learning_rate
#       - max_iters: Número máximo de iteraciones
#       - minibatch_size: Tamaño del minibatch
#       - epsilon: Cota del error
# Aclaraciones:
#       - Para garantizar la aleatoriedad de las muestras cogidas en el SGD,
#         declaro un array de índices desde el 0 hasta la longitud de una tupla
#         que será permutado (shuffle) cada vez. El conjunto de índices permutado, sujeto
#         al tamaño del minibatch, definirá los elementos del dataset que se evalúan
#         en el algoritmo.

def sgd(x,y,learning_rate,max_iters,minibatch_size,epsilon):
    w = np.zeros(len(x[0]),np.float64)
    indices = np.array(range(0,x.shape[0]))
    iter = 0
    while Err(x,y,w) > epsilon and iter < max_iters:
        last_w = w
        for i in range(0,w.size):
            sum = 0
            np.random.shuffle(indices)
            suma = (np.sum(x[indices[0:minibatch_size:1],i] * (x[indices[0:minibatch_size:1]].dot(last_w)-y[indices[0:minibatch_size:1]])))
            w[i] = w[i] - (2.0/minibatch_size) * learning_rate * suma
        iter = iter + 1
    return w


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


# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')

"""
# CALCULANDO EL MEJOR LEARNING RATE PARA SGD
# Miro entre 0.01 y 0.1 100 iteraciones.
# Miro entre 0.01 y 0.12 100 iteraciones
# Miro entre 0.01 y 0.04 --> Mejores resultados con error 0.081...
# Miro entre 0.55 y 0.65 200 iters


rates = np.linspace(0.01,0.04,num=200)
datos = []
for i in rates:
    w = sgd(x,y,i,500,64,10**(-3))
    e = Err(x,y,w)
    datos.append([i,e])

datos = np.array(datos)
print(datos)
plt.scatter(datos[:,0],datos[:,1])
plt.show()
"""
# Utilización del Gradiente descendente estocástico para nuestro dataset
"""
w = sgd(x, y, 0.01, 500, 64,10**(-3))
print("Resultados del error en el Gradiente Descendente Estocástico")
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))

input("\n--- Pulsa una tecla para continuar ---\n")

# Separando etiquetas para poder escribir leyenda en el plot
label1 = []
label5 = []
for i in range(0,len(y)):
    if y[i] == 1:
        label5.append(x[i])
    else:
        label1.append(x[i])
label5 = np.array(label5)
label1 = np.array(label1)

# Plot de la separación de datos SGD

plt.scatter(label5[:,1],label5[:,2],c='g',label="5")
plt.scatter(label1[:,1],label1[:,2],c='r',label="1")
plt.plot([0, 1], [-w[0]/w[2], -w[0]/w[2]-w[1]/w[2]])
plt.xlabel('Intensidad')
plt.ylabel('Simetria')
plt.legend()
plt.title('SGD')
plt.show()

# Aplicación del algoritmo de la pseudoinversa

w = pseudoinverse(x,y)
print("Resultados del error en el algoritmo de pseudoinversa")
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))

input("\n--- Pulsa una tecla para continuar ---\n")

# Plot de la separación de datos pseudoinversa

plt.scatter(label5[:,1],label5[:,2],c='g',label="5")
plt.scatter(label1[:,1],label1[:,2],c='r',label="1")
plt.plot([0, 1], [-w[0]/w[2], -w[0]/w[2]-w[1]/w[2]])
plt.xlabel('Intensidad')
plt.ylabel('Simetria')
plt.legend()
plt.title('Pseudoinversa')
plt.show()
"""
################### EJERCICIO 2 ###################################

# Función que genera datos en el cuadrado [-size,size]x[-size,size]
def simula_unif(N,d,size):
    return np.random.uniform(-size,size,(N,d))

#### EXPERIMENTO

# a) Generar una muestra de entrenamiento de 1000 puntos en [-1,1]x[-1,1]

muestra = simula_unif(1000,2,1)
plt.scatter(muestra[:,0],muestra[:,1])
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")
