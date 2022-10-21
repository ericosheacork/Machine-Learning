# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 17:38:57 2022

@author: erico
"""
import numpy as np
import pandas as pd


df = pd.read_csv('day.csv')
print(df)
x = df.columns
print(x)

emptyFrame = df.loc[df["weathersit"] == 1]

print(int(emptyFrame.shape[0]))
print(emptyFrame)

print('Mean Bicycle Rentals on clear days is ',emptyFrame["cnt"].mean())