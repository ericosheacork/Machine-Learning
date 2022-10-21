# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 12:00:26 2022

@author: erico
"""
import pandas as pd
import numpy as np

df = pd.read_csv("titanic.csv")

#Task A

# num peope
count = df.shape[0]
print("Number of People " , count)


# survived
survived = df[df['Survived'] == 1]
print("People Survived" , survived.shape[0])

print("Survival Rate" , survived.shape[0] / count)


#Task B
men = df[ df['Sex']  == "male"]
menCount = men.shape[0]
menPercentage = men[men['Survived'] ==1]
survivalInt = menPercentage.shape[0]
#print("Men onboard" , men.shape[0])
#print("Men" ,menPercentage)
#print("Men survived",survivalInt)
print("Male Survival Rate" , (survivalInt / menCount))

women = df[df['Sex'] == "female"]
womenCount = women.shape[0]
womenPrecent = women[women['Survived'] == 1].shape[0]
print("Female Survival Rate ",(womenPrecent / womenCount))

#Task C
casualties = df[df['Survived'] == 0]
print("Average fair of casualties" , casualties['Fare'].mean())

survived = df[df['Survived'] == 1]
print("Average fair of survivor" , survived['Fare'].mean())