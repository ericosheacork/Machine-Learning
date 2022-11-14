# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 11:14:37 2022

@author: erico
"""

import sklearn as sk
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import tree
from sklearn import model_selection
from sklearn import metrics

iris = datasets.load_iris()
kf = model_selection.KFold()n_splits
train_data, test_data, train_target, test_target =model_selection.train_test_split(iris.data, iris.target,test_size=0.2)

for i in range(1 , 5):
    print(i)
    clf = tree.DecisionTreeClassifier(max_depth=(i))
    for j in range(len(train_data)):
        clf.fit(train_data[j] , train_target[j])
        prediction = clf.predict(train_data[j])
        score = metrics.accuracy_score(train_target[j], prediction)
        print("Score at: " , i , " is: ", score)
    
