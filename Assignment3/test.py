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
def model_function(deg,data, parameter_vector):
    result = np.zeros(len(data))
    k = 0
    for i in range(deg + 1):
        for j in range(i + 1):
            # we produce a 2d data vector here
            result += parameter_vector[k] * (data[:, 0] ** i) * (data[:, 1] ** (i - j))
            k += 1

    return result


# the coefficient function will return the amount of coefficients needed for the degree value
#8 for nested for loops for the 8 feature vectors
def num_coefficients(d):
    t = 0
    for i in range(d + 1):
        for j in range(d + 1):
            for k in range(d + 1):
                for l in range(d + 1):
                    for m in range(d + 1):
                        for n in range(d + 1):
                            for o in range(d + 1):
                                for p in range(d + 1):
                                    for q in range(d + 1):
                                        t = t + 1
    return t


# this linearize function uses the black-box linearization procedure
def linearize(deg, data, p0):
    f0 = model_function(deg,data, p0)
    J = np.zeros((len(f0), len(p0)))

    epsilon = 1e-6
    for i in range(len(p0)):
        p0[i] += epsilon
        fi = model_function(deg,data, p0)
        p0[i] -= epsilon
        di = (fi - f0) / epsilon
        J[:, i] = di

    return f0, J


def calculate_update(y, f0, J):
    l = 1e-2
    N = np.matmul(J.T, J) + l * np.eye(J.shape[1])
    r = y - f0
    n = np.matmul(J.T, r)
    dp = np.linalg.solve(N, n)
    return dp


def calculate_covariance(y, f0, J):
    l = 1e-2
    N = np.matmul(J.T, J) + l * np.eye(J.shape[1])
    r = y - f0
    sigma0_squared = np.matmul(r.T, r) / (J.shape[0] - J.shape[1])
    cov = sigma0_squared * np.linalg.inv(N)
    return cov


def main():
    heat_loads, cool_loads, data, = regression()
    # I enlarge the deg parameter vector to accomodate all coefficients
    data = np.array(data)
    heat_loads = np.array(heat_loads)
    cool_loads=np.array(cool_loads)
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
            p0 = np.zeros(num_coefficients(deg))
            for i in range(max_iter):
                f0, J = linearize(deg, data, p0)
                dp = calculate_update(target, f0, J)
                print(dp)
                p0 += dp
                x, y, = np.meshgrid(np.arange(np.min(data[:, 0]), np.max(data[:, 0]), 0.1),
                                    np.arange(np.min(data[:, 1]), np.max(data[:, 1]), 0.1))
                test_data = np.array([x.flatten(), y.flatten()]).transpose()
                test_target = model_function(deg, test_data, p0)
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(data[:, 0], data[:, 1], target, c='r')
                ax.plot_surface(x, y, test_target.reshape(x.shape))


main()
