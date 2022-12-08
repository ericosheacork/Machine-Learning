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
    targets = data[["Heating load", "Cooling load"]]
    heat_loads = data["Heating load"]
    cool_loads = data["Cooling load"]
    features = data[["Relative compactness", "Surface area", "Wall area", "Roof area", "Overall height", "Orientation",
                     "Glazing area", "Glazing area distribution"]]
    print("MIN LOADS :", targets.min())
    print("MAX LOADS :", targets.max())
    coefficients = len(features.columns)

    print("Number of coefficients: ", coefficients)

    return heat_loads, cool_loads, features


# I create a polynomial model function that takes a degree value for the degree of the ploynomial
# and a list of features that we extract from task1 in this case 8 therefore there will be 8 nested for loops in total
# this function will provide the shape that needs to be fitted to the data to make sense
#
def model_function(deg, data, parameter_vector):
    result = np.zeros(data.shape[0])
    k = 0
    for i in range(deg + 1):
        for j in range(i + 1):
            for k in range(i + 1):
                for l in range(i + 1):
                    for m in range(i + 1):
                        for n in range(i + 1):
                            for o in range(i + 1):
                                for p in range(i + 1):
                                    for q in range(i + 1):
                                        if j + k + l + m + n + o + p + q == i:
                                            result += (p[k] * (data[:, 0] ** j) *
                                                        (data[:, 1] ** k) * (data[:, 2] ** l) *
                                                        (data[:, 3] ** m) * (data[:, 4] ** n) *
                                                        (data[:, 5] ** o) * (data[:, 6] ** p) *
                                                        (data[:, 7] ** q))
                                            k += 1

    return result


# the coefficient function will return the amount of coefficients needed for the degree value
# 8 for nested for loops for the 8 feature vectors
def num_coefficients(d):
    t = 0
    for i in range(d + 1):
        for j in range(i + 1):
            for k in range(i + 1):
                for l in range(i + 1):
                    for m in range(i + 1):
                        for n in range(i + 1):
                            for o in range(i + 1):
                                for p in range(i + 1):
                                    for q in range(i + 1):
                                        if j + k + l + m + n + o + p + q == i:
                                            t = t + 1
    return t


# this linearization function uses the black-box linearization procedure
def linearize(deg, data, p0):
    # first we calculate the model function at the linearization point p0
    f0 = model_function(deg, data, p0)
    J = np.zeros((len(f0), len(p0)))

    epsilon = 1e-6
    # linearisation occurs here
    # Occurs by iterating through all components of p0
    for i in range(len(p0)):
        # add a small perturbation to the individual component
        p0[i] += epsilon
        # now calculate the model function again with the perutrbed value
        fi = model_function(deg, data, p0)
        # now remove the value added from the pertrurbation value epsilon
        p0[i] -= epsilon
        # the partial derivatives for the jacobian matrix are calculated below
        di = (fi - f0) / epsilon
        # J is the jacobian matrix and we add the value di to it
        J[:, i] = di

    return f0, J


def calculate_update(y, f0, J):
    l = 1e-2
    # multiplying 𝐽𝑇 J here then adding the regularisation matrix which requires the same size
    N = np.matmul(J.T, J) + l * np.eye(J.shape[1])
    # residual is calculates in the line below
    # we subtract the target data values from the f0 function model data
    # e.g y=[1 2 4 8 16] , f0= [0. 0. 0. 0. 0. ]therefore r = [1. 2. 3. 4. 8. 16.]
    # requires the target data and the f0 data values to be the same length
    # i.e. the size of the residual vector is the number of data points
    r = y - f0
    # normalisation occurs here
    n = np.matmul(J.T, r)
    # solving for Δ𝜃 here faster than matrix inversion
    dp = np.linalg.solve(N, n)
    return dp


#this function will calculate the accuracy of the regression algorithm
#to do this we first estimate the variance factor
#then i can calculate the covariance matrix
def calculate_covariance(y, f0, J):
    l = 1e-2
    #first i calculate the normal equation matrix and residuals
    N = np.matmul(J.T, J) + l * np.eye(J.shape[1])
    r = y - f0
    # the variance factor is calculated by squaring the residuals
    sigma0_squared = np.matmul(r.T, r) / (J.shape[0] - J.shape[1])
    #finally the covariance matrix is calculated, it is proportional to the inverse of N
    cov = sigma0_squared * np.linalg.inv(N)
    return cov


def regression(deg ,target , data ):
    max_iter=10
    p0 = np.zeros(num_coefficients(deg))
    for i in range(max_iter):
        f0, J = linearize(deg,data , p0)
        dp = calculate_update(target, f0 , J)
        p0 += dp
    return p0

def main():
    heat_loads, cool_loads, data, = regression()
    data = np.array(data)
    heat_loads = np.array(heat_loads)
    cool_loads = np.array(cool_loads)
    max_iter = 10
    for x in range(2):
        if x == 0:
            target = heat_loads
            print("HEATING LOADS")
            print("===============================================================")
        else:
            target = cool_loads
            print("COOLING LOADS")
            print("===============================================================")
        for deg in range(3):
           p0 = regression(deg, target, data)
                x, y, = np.meshgrid(np.arange(np.min(data[:, 0]), np.max(data[:, 0]), 0.1),
                                    np.arange(np.min(data[:, 1]), np.max(data[:, 1]), 0.1))
                test_data = np.array([x.flatten(), y.flatten()]).transpose()
                test_target = model_function(deg, test_data, p0)
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(data[:, 0], data[:, 1], target, c='r')
                ax.plot_surface(x, y, test_target.reshape(x.shape))


main()
