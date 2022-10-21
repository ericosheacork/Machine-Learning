# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 16:22:20 2022

@author: erico
"""
#Task 3.
#In this task you will generate a random dataset with n clusters each containing k data points. The
#dataset should contain 2 features and a label for each data point. It should be compatible with the
#classifier input format for Scikit-Learn.

#A. Generate the cluster mean, i.e. n vectors of size 2 uniformly distributed in the unit box. (Hint:
#use np.random.rand)

#B. Generate the data points for each cluster by adding Normal distributed noise to the cluster
#means. (Hint: use np.random.randn)

#C. Create the data and target vector formatted as required by Scikit-Learn classifiers.

#D. Visualise the data in a scatter plot


import sklearn as sk
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt

#A
X = np.random.rand(3,1)
print(X)

#B i think
x1 =   1 * np.random.randn(100,2) + X[0,0]
x2 =  1 * np.random.randn(100,2) + X[1,0]
x3 =  1 * np.random.randn(100,2) + X[2,0]

#C
names = ['savings' , 'earnings']

#plt.plot(x1 , x2 , x3)
#plt.ylabel('test numbers')
#plt.show()

plt.scatter(x1 , x2 , c = x3 )
plt.xlabel('savings')
plt.ylabel('earnings')
plt.show()

#christians

no_clusters = 3
cluster_mean = np.random.rand(no_clusters , 2)

data = np.array([[]])
target = np.array([[]] , dtype= 'int')

points_per_cluster = 100
sigma = 0.1

for i in range(no_clusters):
    noise = sigma * np.random.randn(points_per_cluster, 2)
    cluster = cluster_mean[i, : ] + noise
    data = np.append(data, cluster).reshape((i + 1) * points_per_cluster, 2)
    target = np.append(target, [i] * points_per_cluster)


plt.figure()
plt.scatter(data[:, 0], data[:,1], c = target)
plt.show()