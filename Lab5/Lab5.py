# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:07:07 2022

@author: erico
"""

import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import datasets,svm,metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
 

    
digits = datasets.load_digits()
print(digits.data.shape)
print(digits.DESCR) 

targets = set(digits.target)
print(targets)

for digit in targets:
    plt.figure()
    for i in range(3):
        for j in range(3):
            plt.subplot(3,3,i*3+j+1)
            index = np.random.randint(0,sum(digits.target==digit)-1)
            plt.imshow(digits.data[digits.target == digit][index].reshape(8,8))
    train_data,test_data,train_target,test_target = model_selection.train_test_split(digits.data,digits.target,)
            



    
import matplotlib.pyplot as plt
plt.gray()
for i in range(10):
    
    plt.matshow(digits.images[i])
    
    
    plt.show()

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
clf = svm.SVC(gamma=0.001)
print(data[1])

Xtrain, Xtest, Ytrain,Ytest = train_test_split(data,digits.target, test_size=0.5,shuffle = False)

clf.fit(Xtrain, Ytrain)

predicted = clf.predict(Xtest)
reshaped = []
_, axes = plt.subplots(nrows = 1, ncols = 4, figsize=(10,3))
for ax, image,label in zip(axes, digits.images,digits.target):
    ax.set_axis_off()
    image = image.reshape(8,8)
    reshaped.append(image)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation = "nearest")
    ax.set_title("Training: %i" % label)
print(reshaped)

