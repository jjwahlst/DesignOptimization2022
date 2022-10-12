

import numpy as np
from scipy.optimize import minimize
import math
import bayes_opt
from bayes_opt import BayesianOptimization
import warnings



# problem 2


def fun(x1, x2):

    return (4-2.1 * x1**2 + (x1**4)/3)*(1**2) + x1*x2 + (-4 + 4 * x2**2)*x2**2
# Check function


x0 = [1, 2]
print('Test=', fun(x0))


#check = [1, 1]
#fun(check)
#print(fun(check))


search_space = {'x1': (-3, 3), 'x2': (-2, 2)}

optimizer = BayesianOptimization(f=fun, pbounds=search_space, random_state = 1234, verbose=1)

optimizer.maximize(init_points=2, n_iter=10)

print(optimizer.maximize)
