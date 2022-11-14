# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 11:06:30 2022

@author: erico
"""

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

def kernel_function(x , y):
    print(x.shape , y.shape)
    result = np.zeros(x , y)
    

def main():
    iris = datasets.load_iris()
    
    #print(iris)
    
    model = SVC()
    
    x = pd.DataFrame(data = iris['data'], columns = iris['feature_names'])
    y=pd.DataFrame(data = iris['target'] ,columns = ['target'])
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
    
    model.fit(x_train, y_train)
    
    pred = model.predict(x_test)
    
    print(confusion_matrix(y_test,pred))
    
    print(classification_report(y_test, pred))
    
    #kfold cross validation
    
    k_folds = KFold(n_splits = 50, shuffle = True)
    
    for train_index, train_target in k_folds.split(x):
        X_train, X_target = x.iloc[train_index, :], x.iloc[train_target,:]
        Y_train, Y_target = y.iloc[train_index], y.iloc[train_target]
        model.fit(X_train, Y_train)
        
        val_preds = model.predict(X_target)
        print(confusion_matrix(Y_target, val_preds))
        print(classification_report(Y_target, pred))
main()        