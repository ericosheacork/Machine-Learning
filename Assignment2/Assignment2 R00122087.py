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
   perceptron_t_times = []
   perceptron_p_times = []
   svm_scores = []
   svm_best_size = 0
   svm_t_times = []
   svm_p_times = []
   knn_scores=[]
   knn_best_k = []
   knn_t_times = []
   knn_p_times = []
   tree_scores= []
   tree_t_times = []
   tree_p_times = []
   sizes = [300,600,1000,4000,10000]
   
   #parameterizing the individual subframes of the data and labels dataframe into test sizes of rows
   data_sizes = [data.iloc[:300],data.iloc[:600], data.iloc[:1000] , data.iloc[:4000] , data.iloc[:10000]] #,data]
   label_sizes = [labels.iloc[:300], labels.iloc[:600] , labels.iloc[:1000], labels.iloc[:4000] , labels.iloc[:10000]] #,labels]

   #testing the perceptron on different dataframe sizes
   for i in range(0 , len(data_sizes)):
       print("I is : " , i)
       #getting the data and labels from their respective arrays
       temp_data = data_sizes[i]


       temp_labels = label_sizes[i]
       #splitting the data into train data and test data, same for labels
       train_data,test_data,train_target,test_target = model_selection.train_test_split(temp_data, temp_labels,test_size = 0.3)
       #calling the perceptron classifier
       perceptron , p_t_time , p_p_time = perceptron_classifier(train_data, train_target, test_data, test_target)
       svm , s_t_time , s_p_time= svm_classifier(train_data, train_target, test_data, test_target)
       tree_t_time , tree_p_time , tree_accuracy = decision_tree_classifier(train_data , train_target , test_data , test_target)
       tree_t_times.append(tree_t_time)
       tree_p_times.append(tree_p_time)
       tree_scores.append(tree_accuracy)
       

       perceptron_scores.append(perceptron)
       perceptron_t_times.append(p_t_time)
       perceptron_p_times.append(p_p_time)

       svm_scores.append(svm)
       svm_t_times.append(s_t_time)
       svm_p_times.append(s_p_time)
       
       knn_k , k_score  , k_t_time , k_p_time= knn_classifier(train_data,train_target ,test_data , test_target)
       knn_scores.append(k_score)
       knn_best_k.append(knn_k)
       knn_t_times.append(k_t_time)
       knn_p_times.append(k_p_time)

       if perceptron == max(perceptron_scores):
           perceptron_best_size = sizes[i]
       if svm == max(svm_scores):
           svm_best_size = sizes[i]
       
   print("Average Perceptron Score: " , average(perceptron_scores))
   print("Minimum Perceptron Score: ", min(perceptron_scores))
   print("Maximum Perceptron Score: ", max(perceptron_scores))
   print("Best Perceptron Score: " , max(perceptron_scores), " At Size: ", perceptron_best_size)
   print("=======================================================================================================================")
   print("Average SVM Score: ", average(svm_scores))
   print("Minimum SVM Score: ", min(svm_scores))
   print("Maximum SVM Score: ", max(svm_scores))
   print("SVM Scores", svm_scores)
   print("Best SVM Score: ", max(svm_scores), " At Size: ", svm_best_size)
   print("=======================================================================================================================")
   print("Decision Tree Scores: ", tree_scores)
   print("=======================================================================================================================")
   print("KNN Scores", knn_scores)

   for j in range(0,5):
    if j ==0:
        plt.xlabel("Sample Sizes")
        plt.ylabel("Time taken")
        plt.plot(sizes, perceptron_t_times ,label = "Training Time")
        plt.plot( sizes,perceptron_p_times, label = "Prediction Time")
        plt.legend()
        plt.title("Perceptron times over sample size")
        plt.show()

    if j ==1:
        plt.xlabel("Sample Sizes")
        plt.ylabel("Time taken")
        plt.plot(sizes,svm_t_times, label = "Training Time")
        plt.plot(sizes,svm_p_times, label = "Prediction Time")
        plt.legend()
        plt.title("SVM times over sample size")
        plt.show()
    if j ==2:
            plt.xlabel("Sample Size:")
            plt.ylabel("Accuracy Score")
            plt.title("Decision Tree Mean Accuracy over sample size")
            plt.plot(sizes, tree_scores, label = "accuracies")
            plt.legend()
            plt.show()
    if j == 3:
        plt.xlabel("Sample SIzes")
        plt.ylabel("Time taken")
        plt.title("Decision tree computation times over sample size")
        plt.plot(sizes, tree_t_times , label = "Training times")
        plt.plot(sizes , tree_p_times , label = "Prediction times")
        plt.legend()
        plt.show()
    if j ==4:
        plt.xlabel("Sample SIzes")
        plt.ylabel("Time taken")
        plt.title("KNN classifier times over sample size")
        plt.plot(sizes , knn_t_times , label = "Training TImes")
        plt.plot(sizes, knn_p_times , label = "Prediciton Times")
        plt.legend()
        plt.show()
        
        
   return 0


def perceptron_classifier(train_data , train_labels , test_data, test_labels):
    
    clf = linear_model.Perceptron()
    starttime = timeit.default_timer()
    clf.fit(train_data , train_labels)
    training_time = timeit.default_timer() - starttime
    print("Time taken for perceptron training: ", training_time)
    starttime = timeit.default_timer()
    prediction = clf.predict(test_data)
    prediction_time = timeit.default_timer() - starttime
    print("Time taken for perceptron prediction: " , prediction_time)
    score = metrics.accuracy_score(test_labels, prediction)
    return score , training_time , prediction_time
def svm_classifier(train_data , train_labels , test_data , test_labels):
    gamma_ranges = [1e1,1e2,1e3,1e4,1e5]
    best_scores=[]
    
    for gamma_range in gamma_ranges:
        scores = []
        clf = svm.SVC(kernel = 'rbf' ,gamma=gamma_range )
        starttime = timeit.default_timer()
        clf.fit(train_data,train_labels)
        training_time = timeit.default_timer() -starttime
        #print("Time taken for svm training: ", timeit.default_timer() - training_time)
    
        starttime = timeit.default_timer()
        prediction = clf.predict(test_data)
        prediction_time = timeit.default_timer() - starttime
        #print("Time taken for svm prediction: ", timeit.default_timer() - prediction_time)
        svm_score = metrics.accuracy_score(test_labels , prediction)
        scores.append(svm_score)
        #print('svm score is: ', svm_score)
    best_scores.append(max(scores))
    print("Best Score is :" , max(best_scores) , " At Gamma: ", gamma_ranges[best_scores.index(max(best_scores))])
    return svm_score , training_time , prediction_time
def knn_classifier(train_data , train_labels , test_data , test_labels):

    kf = model_selection.KFold(n_splits=4)
    knn_scores = []
    k = 20
    starttime = timeit.default_timer()
    for i in range(1,k):
        scores = []

        for train_index, test_index in kf.split(train_data):
            clf = neighbors.KNeighborsClassifier(n_neighbors=k)
            clf.fit(train_data.iloc[train_index].values, train_labels.iloc[train_index].values)          
            prediction = clf.predict(train_data.iloc[test_index].values)
            score = metrics.accuracy_score(train_labels.iloc[test_index].values , prediction)
            scores.append(score)
        knn_scores.append(np.mean(scores))
        best_k_time = timeit.default_timer() - starttime
    print("best value: ",np.argmax(knn_scores))
    print("Time Taken To find Best K: ",best_k_time)
    best_k = np.argmax(knn_scores)+1
    print("Best K: ", best_k)
    
    clf = neighbors.KNeighborsClassifier(n_neighbors = best_k)
    starttime = timeit.default_timer()
    clf.fit(train_data,train_labels)
    train_time = timeit.default_timer() - starttime
    starttime = timeit.default_timer()
    prediction = clf.predict(test_data)
    predicton_time = timeit.default_timer() - starttime
    knn_score = metrics.accuracy_score(test_labels , prediction)
    print("Best kNN score: ", knn_score, "At K: " , best_k)
    print("Training Time for KNN: ", train_time)
    print("Prediction Time for KNN: ", predicton_time)
    return best_k, knn_score , train_time , predicton_time
def decision_tree_classifier(train_data , train_labels , test_data, test_labels):
    
    clf = tree.DecisionTreeClassifier(random_state=(0))
    starttime = timeit.default_timer()
    clf.fit(train_data,train_labels)
    train_time = timeit.default_timer() - starttime
    
    starttime = timeit.default_timer()
    prediction =clf.predict(test_data)
    pred_time = starttime - timeit.default_timer()
    score = metrics.accuracy_score(test_labels, prediction)
    return score , train_time , pred_time
def average(array):
    return sum(array)/len(array)

labels, data = task1(dataframe)
kfold_cross_validator(labels, data)