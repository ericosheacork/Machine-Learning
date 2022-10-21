# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 11:22:04 2022

@author: erico
"""

import pandas as pd
import numpy as np

df = pd.read_csv("day.csv")
#print(df.head(10))
compareCasual = df['casual']
compareRegistered = df['registered']
holTrue = df['holiday'] == 1
holFalse = df['holiday']  == 0

dfTrue = df[compareRegistered & compareCasual & holTrue]

dfFalse = df[compareRegistered & compareCasual & holFalse]

#Task A
#print(dfTrue)
print('Casual Mean when Holiday' , dfTrue['casual'].mean())
print( 'Registered Mean when Holiday', dfTrue['registered'].mean())
print('\n')

#rint(dfFalse)
print('Casual Mean when not Holiday' , dfFalse['casual'].mean())
print( 'Registered Mean when not Holiday', dfFalse['registered'].mean())

#Task B
print("Max Temp " , df['temp'].max())
print("Min Temp" , df['temp'].min())

#Task C
higherCas = df[df['casual'] > df['registered']]
print(higherCas)

#Task D
plotTemp = df[['temp' , 'atemp' , 'casual' , 'registered']]
plt.figure()
pl
print(plotTemp)
