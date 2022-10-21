# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 11:40:10 2022

@author: erico
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 16:09:35 2022

@author: erico
"""

import sklearn as sk
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import metrics
import matplotlib.pyplot as plt
import math
from sklearn import model_selection

def task_1(data_frame , check):    
    print("Task 1 Executing:")
    #Creating the training data list and training label list from the dataframe.
    data_frame["Sentiment"] = data_frame["Sentiment"].map({"positive":1, "negative":0})
   
    #Creating the train data sets needed.
    #train_set contains the reviews indicated by the split feature as training data
    train_set = data_frame[data_frame["Split"] == "train"]
    
    train_data = train_set["Review"]
    train_labels = train_set["Sentiment"]
    
    
    #Creating the evaluation data list and evaluation label list from the dataframe.
    test_set = data_frame[data_frame["Split"] == "test"]    
    test_data = test_set["Review"]  
    test_labels = test_set["Sentiment"]
   
    
   
    #Positive and negative reviews in the training set
    positive_train = len(train_set[train_set["Sentiment"] == 1])
    negative_train = len(train_set[train_set["Sentiment"] == 0])
    
    #positive and negative reviews in the testing set
    positive_test = len(test_set[test_set["Sentiment"] == 1])
    negative_test = len(test_set[test_set["Sentiment"] == 0])
    
    if check == 1:
        #Printing the positive and negative reviews in both the training and testing dataframes
        print("Positive Reviews in Training Dataset: ",positive_train)
        print("Negative Reviews in Training Dataset: ",negative_train)
        print("Positive Reviews in Testing Dataset: ",positive_test)
        print("Negative Reviews in Testing Dataset: ",negative_test)

   
    #Converting the dataframes to lists in accordance with the spec for task 1.
    train_data = train_data.to_list()
    train_labels = train_labels.to_list()
    test_data = test_data.to_list()
    test_labels = test_labels.to_list()
    
   
    return train_data,train_labels,test_data,test_labels,positive_train,negative_train


    
def task_2(data, min_woccur , word_length):
   print("Task 2 Executing:")
   #print("data type is :" , type(data))
   # instantiating a list of tuples to hold the words and corresponding occurences
   word_occurences = {}
   #criteria for the function
   min_word_length = word_length
   min_occurences = min_woccur
   # instantiating a list object to hold the words that meet the function criteria
   target_words = []
   
   #this for loop will loop through the reviews and clean the data allowinf for further processing 
   for i in range(len(data)):
      data[i] = data[i].lower().replace("'" , "").replace('[^a-zA-Z]', ' ').replace('"', '').replace(".", "").replace(",", "")
      #print(data[i])
      data[i] = data[i].split()
      #print(data[i])
      list = data[i]
             
       #inside this nested for loop the individual words in the list of words is examined and compared to the criteria similar to what we did in Lab 1 Task 2.
      for word in list:
           #checking the length of the word meets the min_word_length criteria to add it to the word_occurences list.
           #if statements check wether to add word to the dict or increment its corresponding integer value
           if(len(word) >= min_word_length ):
               if(word in word_occurences):
                  
                   word_occurences[word] = word_occurences[word] + 1
               else:
                   
                   word_occurences[word] = 1
   #Now checks the words that pass the length criteria against the min_occurence criteria                
   for word in word_occurences:
       #if word passes criteria print the word and its value, then append the word to the targe_word list that we return at the end of the function
      if word_occurences[word] >= min_occurences:
       #print(word + ":" , word_occurences[word])
       #append the target_word list with the words that meet the word length and word occurence criteria
       target_words.append(word)
  
   #print(target_words)
   #returning the list of target words   
   return target_words
  
def task_3(train_data , train_sentiment , word_list):
    print("Task 3 Executing:")
    #instatiating 2 dictionaries to hold tha probability values of the negative review words and positive review words
    word_appearences_pos = {}
    word_appearences_neg = {}
    
    #Populating both dictionaries with all the words from the word list created in task 3 with the value 0 to avoid null values
    for i in range( len(word_list)):
        word_appearences_pos[word_list[i]] = 0
        word_appearences_neg[word_list[i]] = 0
   #the below for loop runs through all training reviews 
    for i in range(len(train_data)):
        #print("training sentiment: ",train_sentiment[i])
      
        # checking the sentiment feature vector to append the positive and negative word dictionaries
        if(train_sentiment[i] == 1):
            for word in train_data[i]:
                for item in word_list:
                    if(word == item):
                        word_appearences_pos[word] = word_appearences_pos[word] + 1
        else:
            for word in train_data[i]:
                for item in word_list:
                    if(word == item):
                        word_appearences_neg[word] = word_appearences_neg[word] + 1
                        
   
    #once the dictionaries are populated return them to be used in task 4            
    return word_appearences_pos,word_appearences_neg      
            
def task_4(pos_wordset , neg_wordset , num_pos , num_neg):
    print("Task 4 Executing:")
    #this smoothing vairable will be used to apply laplace smoothing on the probability values of the negative and positive word sets
    smoothing = 1
   
    positive_dict = {}    
    negative_dict = {}
    
    #running the words through a for loop to get the fraction values for the words in the positive word set
    for word in pos_wordset:
        for i in range(len(pos_wordset)):
            positive_dict[word] = (pos_wordset.get(word) + smoothing) /( num_pos + smoothing)
            
         
    
    #repeating the same for loop but for the negative wordset     
    for word in neg_wordset:    
        for i in range(len(neg_wordset)):
            negative_dict[word] = (neg_wordset.get(word) + smoothing) / (num_neg + smoothing)
        
    #calculating the priors here simple calculation
    positive_prior = num_pos / (num_neg + num_pos) 
    negative_prior = num_neg  / (num_neg + num_pos) 
    
    #print(positive_prior, "  " , negative_prior)

    return positive_dict,negative_dict , positive_prior,negative_prior


def task_5(pos_prior , neg_prior , review , pos_probs , neg_probs):
    positive_likelihood = 0
    negative_likelihood = 0
   #here we instatiate two values for the positive and negative likelihood
    
    #for loop used to first check that the value of the words of the review are not none in the wordlist
    for word in review:
        pos_value = pos_probs.get(word)
        neg_value = neg_probs.get(word)
        #once the word pases the check the words logarithm value is added to the probability values of whichever list they exist in respectively
        if pos_value is not None:
            positive_likelihood = positive_likelihood + math.log(pos_value)
        if neg_value is not None:
            negative_likelihood = negative_likelihood + math.log(neg_value)
             
    #final check of the classifier using logarithms once again to determine if the classifier thinks the review is positive or negative-
    #depending on the words contained in the review the function will return a prediciton of 1 for positive and 0 for negative  
    
    if positive_likelihood - negative_likelihood > math.log(pos_prior) - math.log(neg_prior):
         #print(1)
         return 1
    else:
         #print(0)
         return 0

    
def task_6(df):
    print("Task 6 Executing")    
    #creating a mean accuracy variable
    mean_accuracy = 0
    check = 1
    most_accurate = 0
    k_most_accurate = 0
    
    for k in range(1,5):
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatves = 0
        word_length = k
        word_occurs = 1000
       
        print("Word length is: " , k)
        
        
        data_frame = df.copy(deep=True)
        #using task 1 to generate the lists
        training_data, training_labels, testing_data, testing_labels ,positive_train_num , negative_train_num= task_1(data_frame , check)
        check = 0
            
        #using task 2 to generate the target words from the training review set and returns a list of words
        training_words = task_2(training_data , 1000, k)

        
        #this line does the same as above for the testing review data
        #test_words = task_2(testing_data, 1000 , k)
              
        training_set_pos_review_words, training_set_neg_review_words = task_3(training_data , training_labels , training_words)
        
        pos_probabilities, neg_probabilities, pos_prior , neg_prior = task_4(training_set_pos_review_words, training_set_neg_review_words, positive_train_num,negative_train_num)
        
        data = training_data
        target = training_labels
        for i  in range(len(data)):
           
          prediction = task_5(pos_prior, neg_prior, data[i], pos_probabilities, neg_probabilities)
         
          
          
          if(prediction == 0 and target[i] == 0):
              true_negatives = true_negatives +1
          elif(prediction == 0 and target[i] == 1):
              false_negatves = false_negatves + 1
          elif(prediction == 1 and target[i] == 1):
             true_positives = true_positives + 1
          elif(prediction == 1 and target[i] == 0):
              false_positives = false_positives + 1           
        
          
        print("True Positives: " , true_positives)
        print("True Negatives: ",true_negatives)
        print("False Positives: ",false_positives)
        print("False Negatives: ",false_negatves)
        accuracy = (true_negatives + true_positives)/(false_negatves + false_positives + true_negatives + true_negatives)
        print("Accuracy :", accuracy )
        mean_accuracy = mean_accuracy + accuracy
        if accuracy > most_accurate:
            most_accurate = accuracy
            k_most_accurate = k
          
    print("Mean Accuracy :", (mean_accuracy/10))
    print("Most accurate was at word length", k_most_accurate)
    print("With Accuracy of ", most_accurate)
    print("====================================================")
    print("Executing word length ", k_most_accurate, " on testing data")
    
    data = testing_data
    target = testing_labels
    data_frame = df.copy(deep=True)
    #using task 1 to generate the lists
    training_data, training_labels, testing_data, testing_labels ,positive_train_num , negative_train_num= task_1(data_frame , check)
    check = 0
    test_words = task_2(training_data, 1000 , k_most_accurate)
    for i in range(len(data)):
        prediction = task_5(pos_prior,neg_prior,data[i],pos_probabilities,neg_probabilities)
        if(prediction == 0 and target[i] == 0):
            true_negatives = true_negatives +1
        elif(prediction == 0 and target[i] == 1):
            false_negatves = false_negatves + 1
        elif(prediction == 1 and target[i] == 1):
           true_positives = true_positives + 1
        elif(prediction == 1 and target[i] == 0):
            false_positives = false_positives + 1           
      
        
    print("True Positives: " , true_positives)
    print("True Negatives: ",true_negatives)
    print("False Positives: ",false_positives)
    print("False Negatives: ",false_negatves)
    accuracy = (true_negatives + true_positives)/(false_negatves + false_positives + true_negatives + true_negatives)
    print("Accuracy :", accuracy )
    





    
    

df = pd.read_excel("movie_reviews.xlsx")



task_6(df)