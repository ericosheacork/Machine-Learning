# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 11:08:32 2022

@author: erico
"""

import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import neighbors
from sklearn import metrics

df = pd.read_csv("titanic.csv")

column_sex = df['Sex']
print(column_sex)

df["Sex"] = df['Sex'].map({'male':0, 'female':1})

data = df[["Sex", "Pclass"]].to_numpy()
target = df["Survived"].to_numpy()

survivors = len(data[target ==1])
casualties = len(data[target == 0])

kf = model_selection.StratifiedKFold(n_splits = min(survivors, casualties), shuffle=True)
#kf = model_selection.KFold(n_splits = 3)

correct = 0
incorrect = 0
for train_index,test_index in kf.split(data,target):
    true_casualties = []
    true_survivors = []
    false_casualties = []
    false_survivors = []
    print("Next iteration:")
    print(train_index)
    print()
    print(test_index)
    print()
    
    clf = neighbors.KNeighborsClassifier()
    clf.fit(data[train_index,:], target[train_index])
    predicted_labels = clf.predict(data[test_index,:])
    
    print(target[test_index])
    print(predicted_labels)
    print("----")
    print("Confusion Matrix")
    
    c = metrics.confusion_matrix(test_index, predicted_labels)
    true_casualties.append(c[0,0])
    true_survivors.append(c[1,1])
    false_casualties.append(c[])
   
    
    correct +=sum(predicted_labels==target[test_index])
    incorrect+=sum(predicted_labels!=target[test_index])

print(correct)
print(incorrect)
print(correct/(incorrect+correct))
