# -*- coding: utf-8 -*-

"""
Ejercicio 1: Ejercicio sobre la búsqueda iterativa de óptimos. Práctica 1.
Aprendizaje Automático
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
input("\nPulse enter para continuar")

# Apartado b)
print("Apartado b)")
w,k,data = GD(E,np.array([1.0,1.0]),0.01,gradientE,evalE,10**(-14))
print("El número de iteraciones necesarias para epsilon = 10**(-14) es ",k)
input("\nPulse enter para continuar")

# Apartado c)
print("Apartado c)")
print("COORDENADAS:")
print("Coordenada X: ", w[0])
print("Coordenada Y: ", w[1])
input("\nPulse enter para continuar")


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

w,k,data = GD(f,np.array([0.1,0.1]),0.01,gradientf,evalf,10**(-14),50)
plt.plot(range(0,k+1),data,'bo')
plt.xlabel('Número de iteraciones')
plt.ylabel('f(x,y)')
plt.show()
input("\nPulse enter para continuar")

## para learning_rate = 0.1
print("Learning rate 0.1")
w,k,data = GD(f,np.array([0.1,0.1]),0.1,gradientf,evalf,10**(-14),50)
plt.plot(range(0,k+1),data,'bo')
plt.xlabel('Número de iteraciones')
plt.ylabel('f(x,y)')
plt.show()
input("\nPulse enter para continuar")

# Apartado b)

print("Apartado b)")

datos = []

for n in np.array([(0.1,0.1),(1,1),(-0.5,-0.5),(-1,-1)]):
    w,k,data = GD(f,n,0.01,gradientf,evalf,10**(-14),50)
    datos.append([w,evalf(w)])

datos = np.array(datos)
print(datos)
