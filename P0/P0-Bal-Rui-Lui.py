# Práctica 0
# Autor: Luis Balderas Ruiz

# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt


# Parte 1
X,y = load_iris(True)
X_n = np.array(X)
last_2features = X_n[:,2:4])
