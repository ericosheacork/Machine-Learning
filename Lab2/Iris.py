# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 16:22:20 2022

@author: erico
"""
import sklearn as sk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn import datasets

iris = datasets.load_iris()

print(iris.data)

print(iris.target)

print(iris.data[:,[2,3]])

print(iris.data.shape)

print(iris. DESCR)


plt.title("Example for classification", fontsize = 'small')

x1 , y1 = datasets.make_classification(n_features=2, n_redundant=0, n_informative=2)

plt.scatter(x1[:,0],x1[:,1],marker='o', c=y1)
plt.plot()
plt.show()
