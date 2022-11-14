# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 10:39:50 2022

@author: erico
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


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
    return 1

task1(dataframe)