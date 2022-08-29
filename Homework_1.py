# initialize numpy and scipy

import numpy as np
from scipy.optimize import minimize

#test imports
# new_matrix = np.array([[1,2,3],[1,2,3],[1,2,3]])
# print(new_matrix)

# Problem 1

def fun(x):
    x1=x[0]
    x2=x[1]
    x3=x[2]
    x4=x[3]
    x5=x[4]
    return (x1-x2)**2 +(x2+x3-2)**2 +(x4-1)**2 +(x5-1)**2

def ST1(x):
    return x1+3*x2
def ST2(x):
    return x3+x4-2*x5
def ST3(x):
    return x2-x5


x0 =np.zeros(5)

b= (-10,10)
bounds=(b,b,b,b,b)

sol = minimize(fun, x0, method='SLSQP', bounds=bounds)

print(sol.x)