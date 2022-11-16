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

def kfold_cross_validator(labels , data):
    
   
   perceptron_scores = []
   perceptron_best_size = 0
   svm_scores = []
   svm_best_size = 0
   knn_scores=[]
   knn_best_size = 0
   tree_scores=[]
   tree_best_size = 0
   kf = model_selection.KFold(n_splits = 12)
   sizes = [300,600,1000,4000,10000,14000]
   
   #parameterizing the individual subframes of the data and labels dataframe into test sizes of rows
   data_sizes = [data[:300],data[:600], data[:1000] , data[:4000] , data[:10000]] #,data]
   label_sizes = [labels[:300], labels[:600] , labels[:1000], labels[:4000] , labels[:10000]] #,labels]
   
   #testing the perceptron on different dataframe sizes
   for i in range(0 , len(data_sizes)):
       print("I is : " , i)
       #getting the data and labels from their respective arrays
       temp_data = data_sizes[i]
       temp_labels = label_sizes[i]
       #splitting the data into train data and test data, same for labels
       train_data,test_data,train_target,test_target = model_selection.train_test_split(temp_data, temp_labels,test_size = 0.3)
       #calling the perceptron classifier
       starttime = timeit.default_timer()
       print("The start time is: " , starttime)
       x = perceptron_classifier(train_data, train_target, test_data, test_target) 
       perceptron_time = timeit.default_timer() -starttime
       print("Time taken for perceptron training: " , perceptron_time)
       svm = svm_classifier(train_data, train_target, test_data, test_target)
       print("Time taken for svm training: " , timeit.default_timer() - perceptron_time)
       print('svm score is: ' ,svm)
       #printing the scores
       print("Perceptron Score :" , x)
       perceptron_scores.append(x)
       svm_scores.append(svm)
       if x == max(perceptron_scores):
           perceptron_best_size = sizes[i]
       if svm == max(svm_scores):
           svm_best_size = sizes[i]
# =============================================================================
#        for train_index, test_index in kf.split(temp_data, temp_labels):
#            print()
# =============================================================================
           
           
            
             
   print("Best Perceptron Score: " , max(perceptron_scores), " At Size: ", perceptron_best_size)
   print("Best SVM Score: " , max(svm_scores) , " At Size: " , svm_best_size)
     
        
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
    return score
def svm_classifier(train_data , train_labels , test_data , test_labels):
    clf = svm.SVC(kernel = 'linear')
    clf.fit(train_data,train_labels)
    prediction = clf.predict(test_data)
    svm_score = metrics.accuracy_score(test_labels , prediction)
    return svm_score
    
    return svm_score
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
    
    for d in range(1,md):
        clf = tree.DecisionTreeClassifier(max_depth = md)
        clf.fit(train_data[train_index], train_labels[train_index])
        prediction_train = clf.predict(train_data[train_index])
        prediction_test = clf.predict(test_data[test_index])
    return 0
labels, data = task1(dataframe)
kfold_cross_validator(labels, data)