# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 16:08:41 2022

@author: erico
"""

# Task 1 fibinacci

FibArray = [0, 1]

def fibonacci(n):
    if n < 0:
        print("Incorrect input")
    elif n < len(FibArray):
        return FibArray[n]
    else:
        FibArray.append(fibonacci(n -1) + fibonacci(n -2))
        return FibArray[n]

fibonacci(40)
#print(FibArray)

choice = int(input("Slect a number between 1 and 40\n"))
print(FibArray[choice -1])

#christians

x = 40
y = [0,1]
for i in range(2, x):
    f.append(f[i-1] + f(i-2))
i = input('enter fib number')
print("The " + i+ "th fib number is " + str(f[int(i) -1]))