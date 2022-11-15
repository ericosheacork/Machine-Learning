# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 10:39:50 2022

@author: erico
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics
from sklearn import svm
from sklearn import neighbors
from sklearn import tree
import timeit

dataframe = pd.read_csv("product_images.csv")
def task1(df):
    #seperating the dataframe into sneakers and ankleboots sub frames
    sneaker_set = df[df["label"] == 0]
    ankleboot_set = df[df["label"] == 1]
    
    
    #taking the lables of the sets 
    sneaker_labels = sneaker_set["label"]
    ankleboot_labels = ankleboot_set["label"]
        
    #these 2 lines remove the label column from the dataframes    
    sneaker_set = sneaker_set.drop(['label'], axis=1)
    ankleboot_set = ankleboot_set.drop(['label'], axis=1)
    
    #this block of code calcualtes the amount images of both sneakers and ankleboots in the dataset
    sneaker_rows = (len(sneaker_set))
    ankleboots_rows=(len(ankleboot_set))    
    print("Number of Sneakers in dataset: " , sneaker_rows)
    print("Number of Ankleboots in dataset: " , ankleboots_rows)
    
    #I use this for loop to plot both images as i ran into issues with subplots
    for i in range(0,2):
        if i == 1:
            plt.figure()
            plt.imshow(sneaker_set.iloc[1].values.reshape(28,28))
            plt.show()
        else:
            plt.imshow(ankleboot_set.iloc[1].values.reshape(28,28))
            plt.show()
    labels = df["label"]
    data = df.drop(['label'], axis=1)
    return labels , data

def task2(labels , data):
    
   train_data,test_data,train_target,test_target = model_selection.train_test_split(data, labels,test_size = 0.3)
   perceptron_best_score = []
   svm_best_score = []
   knn_best_score=[]
   tree_best_score=[]
   for k in range(1,4):
       print(k)
        
   return 0

def kernel_function(x , y):
    result = np.zeros((len(x) , len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            result[i,j] = sum([(x[i][k]*y[j][k])**d for k in range(4) for d in range(1,4)])
    return result

def perceptron_classifier(train_data , train_labels , test_data, test_labels):
    
    clf = linear_model.Perceptron()
    clf.fit(train_data , train_labels)
    prediction = clf.predict(test_data)
    score = metrics.accuracy_score(test_labels, prediction)
    return 0
def svm_classifier(train_data , train_labels , test_data , test_labels):
    clf = svm.SVC(kernel = kernel_function)
    clf.fit(train_data,train_labels)
    prediction = clf.predict(test_data)
    svm_score = metrics.accuracy_score(test_labels , prediction)
    
    return 0
def knn_classifier(train_data , train_labels , test_data , test_labels , k, train_index, test_index):
    knn_scores = []
    for i in range(1,k):
        scores = []
        clf = neighbors.KNeighborsClassifier(n_neighbors=k)
        clf.fit(train_data[train_index], train_labels[train_index])
        prediction = clf.predict(train_data[test_index])
        score = metrics.accuracy_score(train_labels[test_index] , prediction)
        scores.append(score)
    knn_scores.append(np.mean(scores))
    best_k = np.argmax(knn_scores)+1
    print("Best K: ", best_k)
    
    clf = neighbors.KNeighborsClassifier(n_neighbors = best_k)
    clf.fit(train_data,train_labels)
    prediction = clf.predict(test_data)
    knn_score = metrics.accuracy_score(test_labels , prediction)
    print("kNN score: ", knn_score)   
    return 0;
def decision_tree_classifier(train_data , train_labels , test_data, test_labels , md , train_index , test_index):
# =============================================================================
#     min_ = np.min(train_data, axis=0)
#     max_ = np.max(train_data, axis=0)
#     granularity = (max_ - min_)/100
#     g2,g3 = np.meshgrid(np.arange(min_[2], max_[2], granularity[2]), np.arange(min_[3], max_[3], granularity[3]))
# =============================================================================
    clf = tree.DecisionTreeClassifier(max_depth = md)
    clf.fit(train_data[train_index], train_labels[train_index])
    prediction_train = clf.predict
    return 0
labels, data = task1(dataframe)
