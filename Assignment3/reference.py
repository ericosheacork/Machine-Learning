import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


# TASK2
# Here I create a polynomial model function that takes as input parameters the
# degree of the polynomial, a list of feature vectors as extracted in task 1,
# and a parameter vector of coefficients and calculates the estimated target
# vector using a multi-variate polynomial of the specified degree
def calculate_model_function(deg, data, p):
    result = np.zeros(data.shape[0])
    t = 0
    for n in range(deg + 1):
        for i in range(n + 1):
            for j in range(n + 1):
                for k in range(n + 1):
                    for x in range(n + 1):
                        for y in range(n + 1):
                            for z in range(n + 1):
                                for a in range(n + 1):
                                    for s in range(n + 1):
                                        if i + j + k + x + y + z + a + s == n:
                                            result += (p[t] * (data[:, 0] ** i) *
                                                       (data[:, 1] ** j) * (data[:, 2] ** k) *
                                                       (data[:, 3] ** x) * (data[:, 4] ** y) *
                                                       (data[:, 5] ** z) * (data[:, 6] ** a) *
                                                       (data[:, 7] ** s))
                                            t = t + 1
    return result


# TASK2
# Here I created a second function that determines the correct size for the
# parameter vector from the degree of the multi-variate polynomial
def num_coefficients_3(d):
    t = 0
    for n in range(d + 1):
        for i in range(n + 1):
            for j in range(n + 1):
                for k in range(n + 1):
                    for x in range(n + 1):
                        for y in range(n + 1):
                            for z in range(n + 1):
                                for a in range(n + 1):
                                    for s in range(n + 1):
                                        if i + j + k + x + y + z + a + s == n:
                                            t = t + 1
    return t


# TASK3
# Here I created a function that calculates the value of the model function implemented
# in task 2 and its Jacobian at a given linearization point using the numerical
# linearisation procedure. The function takes the degree of the polynomial,
# a list of feature vectors as extracted in task 1, and the coefficients of the
# linearization point as input and calculate the estimated target vector and the
# Jacobian at the linearization point as output.
def linearize(deg, data, p0):
    f0 = calculate_model_function(deg, data, p0)
    J = np.zeros((len(f0), len(p0)))
    epsilon = 1e-6
    for i in range(len(p0)):
        p0[i] += epsilon
        fi = calculate_model_function(deg, data, p0)
        p0[i] -= epsilon
        di = (fi - f0) / epsilon
        J[:, i] = di
    return f0, J


# Indicate and explain where the model function implemented in task 2 is called
# and why [2 points], and where the partial derivatives for the Jacobian are
# calculated and how [2 points]


# TASK4
# I created a function that calculates the optimal parameter update from the
# training target vector extracted in task 1 and the estimated target vector and
# Jacobian calculated in task 3. I start with calculating the normal equation
# matrix; I add a regularisation term to prevent the normal equation system from
# being singular. I calculated the residual and built the normal equation system.
# I solved the normal equation system to obtain the optimal parameter update.
# The function takes the training target vector and the estimated target vector
# and Jacobian at the linearization point as input and calculates the optimal
# parameter update vector as output
def calculate_update(y, f0, J):
    l = 1e-2
    N = np.matmul(J.T, J) + l * np.eye(J.shape[1])
    r = y - f0
    n = np.matmul(J.T, r)
    dp = np.linalg.solve(N, n)
    return dp


# Indicate where the normal equation matrix is calculated and how it is
# regularised [2 points]. Also indicate exactly where the residuals are calculated
# and explain how [2 points].

# TASK5
# I created a function that calculates the coefficient vector that best fits the
# training data. To do that, I initialised the parameter vector of coefficients
# with zeros. I setup an iterative procedure that alternates linearization and
# parameter update. The function takse the degree of the polynomial, the training
# data features, and the training data targets as input and return the best fitting
# polynomial coefficient vector as output.
def regression(deg, train_data, train_target):
    max_iter = 10
    p0 = np.zeros(num_coefficients_3(deg))
    for i in range(max_iter):
        f0, J = linearize(deg, train_data, p0)
        dp = calculate_update(train_target, f0, J)
        p0 += dp
    return p0


# Indicate the parameter vector and how it is updated [2 points]. How do you
# expect the parameter update and the residuals calculated in the previous task
# to evolve in the iterations? [2 points] How could you use this to determine the
# number of iterations required? [1 point]

# TASK6
# I setup two cross-validation procedures, one for the heat loads and one for
# cooling loads [1 point]. I calculated the difference between the predicted
# target and the actual target for the test set in each cross-validation
# fold [1 point] and outputed the mean of absolute differences across all folds
# for both the heating load estimation as well as the cooling load estimation
# [2 points]. Using this as a quality metric, I evaluated polynomial degrees
# ranging between 0 and 2 to determine the optimal degree for
# the model function for both the heating as well as the cooling loads [2 points].

def model_selection(df, data, target):
    lowest_degree_per_set = []
    lowest_degree = []
    for deg in range(1, 3):
        price_diff = []
        kfold = KFold(3, shuffle=True, random_state=1)
        for train_index, test_index in kfold.split(df):
            train_data = data[train_index]
            train_target = target[train_index]
            test_data = data[test_index]
            test_target = target[test_index]
            p0 = regression(deg, train_data, train_target)
            res = calculate_model_function(deg, test_data, p0)
            for x in range(len(res)):
                higher_price = max(res[x], test_target[x])
                lower_price = min(res[x], test_target[x])
                z = higher_price - lower_price
                price_diff.append(z)

        print()
        print("MEAN USING DEGREE OF ", deg)
        print(np.mean(price_diff))
        print('Difference between predicted target and actual target')
        print()
        print()

        lowest_degree.append(np.mean(price_diff))
    lowest_degree_per_set.append(lowest_degree.index(min(lowest_degree)) + 1)

    optimal_deg = max(set(lowest_degree_per_set), key=lowest_degree_per_set.count)
    print("Optimal Degree: ", optimal_deg)

    return lowest_degree_per_set


# TASK7
# Using the full dataset, I estimated the model parameters for both the heating
# loads as well as the cooling loads using the selected optimal model function
# as determined in task 6 [1 point]. I calculated the predicted heating and
# cooling loads using the estimated model parameters for the entire dataset
# [1 point]. I ploted the estimated loads against the true loads for both the
# heating and the cooling case [2 points]. I calculated and output the mean
# absolute difference between estimated heating/cooling loads and actual
# heating/cooling loads [2 points]
def visualization_of_results(df, data, target, deg_list):
    count = 0
    deg = deg_list[count]
    p0 = regression(deg, data, target)
    res = calculate_model_function(deg, data, p0)
    count = count + 1
    y = np.array(target)
    x = np.array(res)
    plt.plot(x, y, 'bo')
    plt.show()


def main():
    # TASK1
    # Here I read the data in from the csv file
    # and split it into features and targets
    # I calculate and output the minimum and maximum heating
    # and cooling loads of buildings present in the dataset
    df = pd.read_csv("energy_performance.csv")
    data = np.array(df[['Relative compactness', 'Surface area',
                        'Wall area', 'Roof area',
                        'Overall height', 'Orientation',
                        'Glazing area', 'Glazing area distribution']])

    targeth = np.array(df['Heating load'])
    targetc = np.array(df['Cooling load'])

    minheatingload = min(targeth)
    maxheatingload = max(targeth)

    mincoolingload = min(targetc)
    maxcoolingload = max(targetc)

    print('Heating Load')
    print('Min', minheatingload)
    print('Max', maxheatingload)
    print('')

    print('Cooling Load')
    print('Min', mincoolingload)
    print('Max', maxcoolingload)

    # TASK6
    lowest_degree_per_seth = model_selection(df, data, targeth)
    lowest_degree_per_setc = model_selection(df, data, targetc)

    # TASK7
    visualization_of_results(df, data, targeth, lowest_degree_per_seth)
    visualization_of_results(df, data, targetc, lowest_degree_per_setc)


main()

