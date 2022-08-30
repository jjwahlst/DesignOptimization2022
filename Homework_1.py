# initialize numpy and scipy

import numpy as np
from scipy.optimize import minimize

#test imports
# new_matrix = np.array([[1,2,3],[1,2,3],[1,2,3]])
# print(new_matrix)

# Problem 1

def fun(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    return (x1-x2)**2 +(x2+x3-2)**2 +(x4-1)**2 +(x5-1)**2

def ST1(x):
    return x1+3*x2
def st2(x):
    return x3+x4-2*x5
def ST3(x):
    return x2-x5


g1 = np.zeros(5)
g2 = np.array([1, 2, 3, 4, 5])
g3 = g1+5

b = (-10, 10)
bounds = (b, b, b, b, b)

sol_one = minimize(fun, g1, method='SLSQP', bounds=bounds)
sol_two = minimize(fun, g2, method='SLSQP', bounds=bounds)
sol_three = minimize(fun, g3, method='SLSQP', bounds=bounds)

print(sol_one.x)
print(sol_two.x)
print(sol_three.x)
