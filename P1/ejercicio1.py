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
from sympy.functions import exp

# Fijo la semilla para garantizar la reproducibilidad de los resultados
np.random.seed(77145416)

u, v = symbols('u v', real=True)
E = (u**2*exp(v)-2*v**2*exp(-u))**2

def gradientE(w):
    der_u = diff(E,u)
    der_v = diff(E,v)
    return np.array([der_u.subs([(u,w[0]),(v,w[1])]),der_v.subs([(u,w[0]),(v,w[1])])])


def f(expr,w):
    return expr.subs([(u,w[0]),(v,w[1])])


def GD(E,w,learning_rate,gradient,f,epsilon, max_iters = 15000000):
    num_iter = 0
    while num_iter <= max_iters and (f(E,w) - epsilon).is_positive:
        w = w - learning_rate * gradient(w)
        num_iter = num_iter + 1

    return w, num_iter

w,k = GD(E,np.array([1.0,1.0]),0.01,gradientE,f,10**(-14))
print(w)
print(k)
