# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 16:41:36 2022

@author: eric o shea
"""
import numpy as np
import pandas as pd


wordList = []
minWordLength = int(input("Eneter the min word length now!\n"))
minWordOccurence = int(input("Enter min word occurence\n"))

with open('dracula.txt', 'r') as file:
    for line in file:
        for word in line.split():
            if word.isalpha() and len(word) >= minWordLength :
                wordList.append(word)
    print(len(wordList))    
    for word in wordList:
        if wordList.count(word) >= minWordOccurence:
            print(word)
            print(wordList.count(word))
            for item in wordList:
                if item == word:
                    wordList.remove(item)

#Christians
# =============================================================================
# minlength = 5
# minoccur = 300
# 
# f = open("dracula.txt", "r")
# doc = f.read()
# f.close()
# allWords = doc.split()
#     wordoccurences = {}
#     for word in allWords:
#         if(len(word) >=minlength):
#             if (word in wordoccurences):
#                 wordoccurences[word] = wordoccurences[word] + 1
#             else:
#                 wordoccurences[word] = 1
#     for word in wordoccurences:
#         if wordoccurences[word] >=minoccur:
#             print(wordoccurences[word])
# =============================================================================


    
        
