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



# ALGORITMO DE GRADIENTE DESCENDENTE
# - Función que implementa Gradiente Descendente.
# - Parámetros:
#   - w: vector de partida que almacenará el mínimo.
#   - learning rate
#   - gradient es una función que calcula el gradiente. Es necesario definir esa función
#   para cada caso que queramos considerar, puesto que depende directamente del nombre
#   y número de variables del caso en cuestión.
#   - f es la función que evalúa la expresión en los diferentes valores de w. También hay que
#   definirla en cada caso.
#   - epsilon fija el criterio de parada, de forma que si la diferencia de la
#     imagen de dos w consecutivos es menor que epsilon, el algoritmo para.


def GD(w,learning_rate,gradient,f,epsilon, max_iters = 15000000):
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
# Calculando el mínimo

# Apartado a)
print("Apartado a)")
print("El gradiente de E es: ", diff(E,u),",",diff(E,v))
input("\n--- Pulsa una tecla para continuar ---\n")

# Apartado b)
print("Apartado b)")
w,k,data = GD(np.array([1.0,1.0],np.float64),0.01,gradientE,evalE,10**(-14))
print("El número de iteraciones necesarias para epsilon = 10**(-14) es ",k)
input("\n--- Pulsa una tecla para continuar ---\n")

# Apartado c)
print("Apartado c)")
print("COORDENADAS:")
print("Coordenada X: ", w[0])
print("Coordenada Y: ", w[1])
input("\n--- Pulsa una tecla para continuar ---\n")


plt.plot(range(0,k+1),data,'bo')
plt.xlabel('Número de iteraciones')
plt.ylabel('f(x,y)')
plt.show()
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

w,k,data_gd1 = GD(np.array([0.1,0.1],np.float64),0.01,gradientf,evalf,10**(-20),50)
print("Coordenadas del mínimo: ", w)
plt.plot(range(0,k+1),data_gd1)
plt.xlabel('Número de iteraciones')
plt.ylabel('f(x,y)')
plt.show()
input("\n--- Pulsa una tecla para continuar ---\n")

## para learning_rate = 0.1
print("Learning rate 0.1")
w,k,data_gd2 = GD(np.array([0.1,0.1],np.float64),0.1,gradientf,evalf,10**(-20),50)
plt.plot(range(0,k+1),data_gd2)
plt.xlabel('Número de iteraciones')
plt.ylabel('f(x,y)')
plt.show()
input("\n--- Pulsa una tecla para continuar ---\n")

# Apartado b)

print("Apartado b)")

datos = []
array = np.array([(0.1,0.1),(1,1),(-0.5,-0.5),(-1,-1)])
for n in array:
    w,k,data = GD(n,0.01,gradientf,evalf,10**(-20),50)
    datos.append([w,evalf(w)])

datos = np.array(datos)

for i in range(0,len(datos)):
    print("Punto de inicio: ", array[i])
    print('(x,y) = ', datos[i][0])
    print('Valor mínimo: ',datos[i][1])


input("\n--- Pulsa una tecla para continuar ---\n")

"""
#Ejercicio 2. Regresión lineal

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


# CALCULANDO EL MEJOR LEARNING RATE PARA SGD
# Miro entre 0.01 y 1 100 iteraciones.
# Miro entre 0.01 y 0.12 100 iteraciones
# Miro entre 0.01 y 0.04 --> Mejores resultados con error 0.081...
# Miro entre 0.55 y 0.65 200 iters

rates = np.linspace(0.01,1,num=200)
datos = []
for i in rates:
    w = sgd(x,y,i,500,64,10**(-3))
    e = Err(x,y,w)
    datos.append([i,e])

datos = np.array(datos)
plt.scatter(datos[:,0],datos[:,1])
plt.xlabel("Learning rate")
plt.ylabel("Ein cometido")
plt.show()

rates = np.linspace(0.01,0.1,num=200)
datos = []
for i in rates:
    w = sgd(x,y,i,500,64,10**(-3))
    e = Err(x,y,w)
    datos.append([i,e])

datos = np.array(datos)
plt.scatter(datos[:,0],datos[:,1])
plt.xlabel("Learning rate")
plt.ylabel("Ein cometido")
plt.show()
"""
"""
print("Calculando el mejor learning rate")
rates = np.linspace(0.01,0.04,num=200)
datos = []
for i in rates:
    w = sgd(x,y,i,500,64,10**(-3))
    e = Err(x,y,w)
    datos.append([i,e])

datos = np.array(datos)
plt.scatter(datos[:,0],datos[:,1])
plt.xlabel("Learning rate")
plt.ylabel("Ein cometido")
plt.show()

input("\n--- Pulsa una tecla para continuar ---\n")

# Probando distintos tamaños para el minibatch
# 80 es el mejor tamaño
print("Calculando el mejor tamaño de minibatch")
mb = [32,50,64,80,100,110,128]
datos = []
for i in mb:
    w = sgd(x,y,0.01,500,i,10**(-3))
    e = Err(x,y,w)
    datos.append([i,e])

datos = np.array(datos)
print(datos)
plt.scatter(datos[:,0],datos[:,1])
plt.xlabel("Tamaño minibatch")
plt.ylabel("Ein cometido")
plt.show()

input("\n--- Pulsa una tecla para continuar ---\n")
# Utilización del Gradiente descendente estocástico para nuestro dataset

w = sgd(x, y, 0.01, 500, 80,10**(-3))
print("W",w)
print("Resultados del error en el Gradiente Descendente Estocástico")
print("W",w)
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
print("W",w)
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

################### EJERCICIO 2 ###################################

# Función que genera datos en el cuadrado [-size,size]x[-size,size]
def simula_unif(N,d,size):
    return np.random.uniform(-size,size,(N,d))

input("\n--- Pulsa una tecla para continuar ---\n")

#### EXPERIMENTO

# a) Generar una muestra de entrenamiento de 1000 puntos en [-1,1]x[-1,1]

print("Generación aleatoria de una muestra de entrenamiento de 1000 en [-1,1]x[-1,1]")
muestra = simula_unif(1000,2,1)
plt.scatter(muestra[:,0],muestra[:,1])
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# b) Definir la función f(x_1,x_2) = sign((x_1-0.2)^2 + x_2^2 -0.6) y asignar etiquetas
#    Introducir ruido

print("Definición de la función f(x_1,x_2) = sign((x_1-0.2)^2 + x_2^2 -0.6)")
# Defino una función general de signo
def sign(x):
    if x>=0:
        return 1
    return -1

# Función pedida en el enunciado
def f_1(x_1,x_2):
    return sign((x_1-0.2)**2 + x_2**2 - 0.6)

column = []
for i in range(0,len(muestra)):
    column.append(f_1(muestra[i][0],muestra[i][1]))

column = np.array(column)


input("\n--- Pulsar tecla para continuar ---\n")


# Definiendo dataset --> matriz con la muestra generada y con la nueva columna asociada a cada tupla según su signo
muestra = np.array(muestra)

print("Evaluando los elementos de la muestra en f e introduciendo ruido... Creando dataset")

# Introduciendo ruido aleatorio en el 10% de los datos
# Para ello, defino un conjunto de índices a lo largo de los índices de column, tomando un 10% de ellos.
# Es importante no reemplazar para que salgan índices únicos
# Tras ello, cambio el signo de las etiquetas situadas en esos índices
indices = np.random.choice(len(column),int(0.1*len(column)),replace=False)
column[indices] = -column[indices]

print("Dibujando las etiquetas obtenidas...")
plt.scatter(muestra[:,0],muestra[:,1],c=column)
plt.xlabel("x_1")
plt.ylabel("x_2")
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# Generado dataset uniendo los datos generados a una columna de 1, tal y como
# se define en teoría
print("Aplicando SGD sobre el nuevo dataset")
dataset = np.hstack((np.ones(shape=(muestra.shape[0],1)),muestra))
w = sgd(dataset, column, 0.01, 1000, 64,10**(-15))
print("Resultados del error en el Gradiente Descendente Estocástico sobre el nuevo dataset")
print("W",w)
print("Ein: ", Err(dataset,column,w))
print("Modelo de regresión lineal")

plt.scatter(muestra[:,0],muestra[:,1],c=column)
plt.plot([-1, 1], [-w[0]/w[2]+w[1]/w[2], -w[0]/w[2]-w[1]/w[2]])
plt.xlabel("x_1")
plt.ylabel("x_2")
plt.title("Modelo de regresión lineal")
plt.axis([-1,1,-1,1])
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# d) Ejecutar los apartados anteriores 1000 veces

print("Ejecutando el experimento 1000 veces")

error_in = 0
error_out = 0

for i in range(0,1000):
    np.random.seed()
    muestra = simula_unif(1000,2,1)
    column = []
    for i in range(0,len(muestra)):
        column.append(f_1(muestra[i][0],muestra[i][1]))

    column = np.array(column, np.float64)
    muestra = np.array(muestra)
    indices = np.random.choice(len(column),int(0.1*len(column)),replace=False)
    column[indices] = -column[indices]
    dataset = np.hstack((np.ones(shape=(muestra.shape[0],1)),muestra))
    w = sgd(dataset, column, 0.01, 1000, 64,10**(-15))
    test = simula_unif(1000,2,1)
    col_test = []
    for i in range(0,len(test)):
        col_test.append(f_1(test[i][0],test[i][1]))
    col_test = np.array(col_test, np.float64)
    test = np.array(test)
    indices = np.random.choice(len(col_test),int(0.1*len(col_test)),replace=False)
    col_test[indices] = -col_test[indices]
    data_test = np.hstack((np.ones(shape=(test.shape[0],1)),test))
    error_in += Err(dataset,column,w)
    error_out += Err(data_test,col_test,w)

error_in = error_in/1000
error_out = error_out/1000

print("Error medio interior y exterior")
print("Error interior medio: ",error_in)
print("Error exterior medio: ",error_out)

input("\n--- Pulsar tecla para continuar ---\n")

################# BONUS #################

print("########### EJERCICIO BONUS ############")

# Definiendo f de nuevo
print("Calculando f y su hessiana")

# En el ejercicio 1 utilicé sympy para definir de forma simbólica f
# Ahora, con el cálculo de la hessiana, me he encontrado con muchos errores
# así que, por simplicidad, he decidido definir los distintos órdenes de la función
# f y escribir explícitamente las derivadas (así cubro las dos posibilidades y, por
# otra parte, se gana muchísimo en eficiencia).

def f(w):
	x = w[0]
	y = w[1]
	return x**2+2*y**2+2*math.sin(2*math.pi*x)*math.sin(2*math.pi*y)

# Derivada parcial de f respecto de x
def f_x(w):
	x = w[0]
	y = w[1]
	return 2*x+4*math.pi*math.sin(2*math.pi*y)*math.cos(2*math.pi*x)

# Derivada parcial de f respecto de y
def f_y(w):
	x = w[0]
	y = w[1]
	return 4*y+4*math.pi*math.sin(2*math.pi*x)*math.cos(2*math.pi*y)

# Gradiente de f
def gradienteF(w):
	return np.array([f_x(w), f_y(w)])

# Derivada segunda de f respecto de x dos veces
def f_xx(w):
	x = w[0]
	y = w[1]
	return 2 - 8*(math.pi)**2*math.sin(2*math.pi*y)*math.sin(2*math.pi*x)

# Derivada segunda de f respecto de x, y luego, y (mismo valor que el caso análogo por el teorema de Schwarz)
def f_xy(w):
	x = w[0]
	y = w[1]
	return 8*(math.pi)**2*math.cos(2*math.pi*x)*math.cos(2*math.pi*y)

# Derivada segunda de f respecto de y dos veces
def f_yy(w):
	x = w[0]
	y = w[1]
	return 4 - 8*(math.pi)**2*math.sin(2*math.pi*x)*math.sin(2*math.pi*y)

# Matriz Hessiana de f
def hessianaF(w):
	return np.array([np.array([f_xx(w), f_xy(w)]), np.array([f_xy(w), f_yy(w)])])


print("Definiendo el método de Newton")

## Función para comprobar si una matriz es definida positiva

def is_positiveDefinite(matrix):
    det = matrix[0][0]*matrix[1][1] - matrix[1][0]*matrix[0][1]
    if(matrix[0][1]>0 and det>0):
        return True
    else:
        return False

## MÉTODO DE NEWTON
# Parámetros:
#   - w: vector inicial w
#   - learning_rate
#   - gradientf: Gradiente de la función f
#   - f: Función a la que le aplicamos el método de minimización
#   - hessf: Matriz hessiana de f
# Método con gran parecido al Gradiente Estocástico, introduciendo la Hessiana de f,
# por lo que la convergencia, para las funciones que son suficientemente regulares,
# es mucho más potente.

def MetodoNewton(w,learning_rate,gradientf,f_2,hessf, max_iters = 15000000):
    medida = [f_2(w)]
    for i in range(0,max_iters):
        last_w = w
        print("Definida positiva en la iteración ", i, is_positiveDefinite(hessf(w)))
        w = w - learning_rate * np.linalg.inv(hessf(w)).dot(gradientf(w))
        medida.insert(len(medida),f_2(w))
    return w, medida

print("EJECUCIONES EN LOS PUNTOS DEL APARTADO ANTERIOR")


print('(1.0, 1.0), LR=0.01')
w, g = MetodoNewton(np.array([1.0,1.0],np.float64), 0.01, gradienteF, f, hessianaF, 50)
print("Punto donde se alcanza el mínimo: (", w[0],",",w[1],")")
plt.plot(range(0,51),g)
plt.xlabel("Número de iteraciones")
plt.ylabel("f(w)")
plt.show()
print("Mínimo de f: " , f(w))

input("\n--- Pulsar tecla para continuar ---\n")

print('(0.1, 0.1), LR=0.01')
w, g = MetodoNewton(np.array([0.1,0.1],np.float64), 0.01, gradienteF, f, hessianaF, 50)
print("Punto donde se alcanza el mínimo: (", w[0],",",w[1],")")
plt.plot(range(0,51),g,label="Método de Newton")
plt.plot(range(0,25),data_gd1,label="Gradiente descendente")
plt.xlabel("Número de iteraciones")
plt.legend()
plt.ylabel("f(w)")
plt.show()
print("Mínimo de f: " , f(w))

input("\n--- Pulsar tecla para continuar ---\n")

print('(-1.0, -1.0), LR=0.01')
w, g = MetodoNewton(np.array([-1.0,-1.0],np.float64), 0.01, gradienteF, f, hessianaF, 50)
print("Punto donde se alcanza el mínimo: (", w[0],",",w[1],")")
plt.plot(range(0,51),g)
plt.xlabel("Número de iteraciones")
plt.ylabel("f(w)")
plt.show()
print("Mínimo de f: " , f(w))

input("\n--- Pulsar tecla para continuar ---\n")

print('(-0.5, -0.5), LR=0.01')
w, g = MetodoNewton(np.array([-0.5,-0.5],np.float64), 0.01, gradienteF, f, hessianaF, 50)
print("Punto donde se alcanza el mínimo: (", w[0],",",w[1],")")
plt.plot(range(0,51),g)
plt.xlabel("Número de iteraciones")
plt.ylabel("f(w)")
plt.show()
print("Mínimo de f: " , f(w))

input("\n--- Pulsar tecla para continuar ---\n")

print('(0.1, 0.1), LR=0.1')
w, g = MetodoNewton(np.array([0.1,0.1],np.float64), 0.1, gradienteF, f, hessianaF,50)
print("Punto donde se alcanza el mínimo: (", w[0],",",w[1],")")
plt.plot(range(0,51),g,label="Método de Newton")
plt.plot(range(0,52),data_gd2,label="Gradiente descendente")
plt.xlabel("Número de iteraciones")
plt.ylabel("f(w)")
plt.legend()
plt.show()
print("Mínimo de f: " , f(w))

input("\n--- Pulsar tecla para continuar ---\n")
