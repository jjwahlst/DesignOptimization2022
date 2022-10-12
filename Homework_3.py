# Homework 3

import numpy as np
from scipy.optimize import minimize
import math
import bayes_opt
from bayes_opt import BayesianOptimization
import warnings


# Problem 1
a11 = 8.07131
a12 = 1730.63
a13 = 233.426

p1sat = 10 ** (a11 - a12/(20+a13))

a21 = 7.43155
a22 = 1554.679
a23 = 240.337

p2sat = 10 ** (a21 - a22/(20+a23))
# CHECK
# print('P1sat', p1sat, 'P2sat', p2sat)

def p(A12, A21, v1, v2):
    return v1 * math.exp(A12 * ((A21 * v2) / (A12 * v1 + A21 * v2)) ** 2) * p1sat + v2 * math.exp(
        A21 * ((A12 * v2) / (A12 * v1 + A21 * v2))) * p2sat


# Problem 2
def fun(x1, x2):
    return -((4 - 2.1 * x1 ** 2 + (x1 ** 4) / 3) * (1 ** 2) + x1 * x2 + (-4 + 4 * x2 ** 2) * x2 ** 2)


# Check function
# print('Test=', fun(1, 2))
# Do the optimization


search_space = {'x1': (-3, 3), 'x2': (-2, 2)}
optimizer = BayesianOptimization(f=fun, pbounds=search_space, random_state=1234, verbose=1)
optimizer.maximize(init_points=100, n_iter=15)
