# -*- coding: utf-8 -*-
"""
TRABAJO 2.
Nombre Estudiante: Luis Balderas Ruiz
Aprendizaje Automático 2019
"""
import numpy as np
import matplotlib.pyplot as plt


# Fijamos la semilla
np.random.seed(77145416)


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
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.

    return a, b

def f_ej12(a,b,x,y):
	return np.sign(y-a*x-b)

"""
Ejercicio 1: SOBRE LA COMPLEJIDAD DE H Y EL RUIDO
"""

class Ejercicio1:

	def __init__(self):
		print("####################################################")
		print("####################################################")
		print("EJERCICIO 1: SOBRE LA COMPLEJIDAD DE H Y EL RUIDO")
		print("####################################################")
		print("####################################################")

	def Ej1(self):
		print("Ejercicio 1. Dibujar una gráfica con la nube de puntos correspondiente")
		print("a) N=50, dim=2, rango = [-50,50] con simula_unif")

		listaN = simula_unif(50,2,[-50,50])
		plt.scatter(listaN[:,0],listaN[:,1])
		plt.title("Nube de puntos generada con simula_unif")
		plt.show()

		input("\n--- Pulsa una tecla para continuar ---\n")

		print("b) N=50, dim=2, sigma=[5,7] con simula_gaus")

		gauss = simula_gaus(50,2,[5,7])
		plt.scatter(gauss[:,0],gauss[:,1])
		plt.title("Nube de puntos generada con simula_gaus")
		plt.show()

		input("\n--- Pulsa una tecla para continuar ---\n")

	def Ej2(self):
		print("Ejercicio 2")

		# En primer lugar, simulo los parámetros a, b para la recta y = ax+b en [-50,50]
		print("Calculando la pendiente (a) y la ordenada en el origen (b) de la recta...")
		a,b = simula_recta([-50,50])
		print("Pendiente (a): ", a)
		print("Término independiente (b): ", b)
		print("Generando la nube de 50 puntos uniformemente")
		puntos = simula_unif(50,2,[5,7])
		print("Añadiendo las etiquetas correspondientes")
		signos = f_ej12(a,b,puntos[:,0],puntos[:,1])
		# Necesita tener la misma dimensión que puntos para poder hacer append
		signos = signos.reshape(len(signos),1)
		puntos = np.append(puntos,signos, axis=1)
		plt.scatter(puntos[:,0],puntos[:,1], c = puntos[:,2])
		plt.plot([5,7],[5*a+b,7*a+b])
		plt.show()



Ejerc1 = Ejercicio1()
#Ejerc1.Ej1()
Ejerc1.Ej2()


###############################################################################
###############################################################################
###############################################################################
###############################################################################
#print('EJERCICIO BONUS\n')

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
