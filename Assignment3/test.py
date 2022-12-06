# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 13:23:20 2022

@author: erico
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D

def regression():
    data = pd.read_csv("energy_performance.csv")
    print(data.head(0))
    targets = data[["Heating load" , "Cooling load"]]
    heat_loads=data["Heating load"]
    cool_loads=data["Cooling load"]
    features = data[["Relative compactness","Surface area","Wall area","Roof area","Overall height","Orientation","Glazing area","Glazing area distribution"]]
    print("MIN LOADS :" , targets.min())
    print("MAX LOADS :" ,targets.max())
    coefficients = len(features.columns)
    
    print("Number of coefficients: " , coefficients)
    
    return heat_loads,cool_loads , features , coefficients
   
# I create a polynomial model function that takes a degree value for the degree of the ploynomial
# and a list of features that we extract from task1 in this case 8 therefore there will be 8 nested for loops in total
#this function will provide the shape that needs to be fitted to the data to make sense
#
def model_function(data , parameter_vector):
    
    result = np.zeros(len(data))
    
    deg = int(-(3./2.) + math.sqrt(3.*3/4 -2 + 2*len(parameter_vector)))
    k= 0
    
    for i in range(deg + 1):
        for j in range(i+1):
            #we produce a 2d data vector here
            result += parameter_vector[k]*(data[:,0]**i) *(data[:,1]**(i-j))
            k+=1
    
    return result

def linearize(data, p0):
    f0 = model_function(data,p0)
    J = np.zeros((len(f0), len(p0)))
    
    epsilon = 1e-6
    for i in range(len(p0)):
        p0[i] += epsilon
        fi = model_function(data,p0)
        p0[i] -= epsilon
        di = (fi - f0)/epsilon
        J[:,i] = di
    
    
    return f0, J

def calculate_update(y,f0,J):
    l=1e-2
    N = np.matmul(J.T,J) + l*np.eye(J.shape[1])
    r= y-f0
    n=np.matmul(J.T,r)
    dp = np.linalg.solve(N,n)
    return dp
def calculate_covariance(y,f0,J):
    l=1e-2
    N=np.matmul(J.T,J) + l*np.eye(J.shape[1])
    r = y-f0
    sigma0_squared = np.matmul(r.T,r)/(J.shape[0] - J.shape[1])
    cov = sigma0_squared * np.linalg.inv(N)
    return cov
def main():    
    heat_loads,cool_loads , features , deg = regression()
    # I enlarge the deg parameter vector to accomodate all coefficients
    p0 = np.zeros(int((deg+2)*(deg+1)/2))
    print(len(p0))
    function_result_set = model_function(features , p0)
    

main()