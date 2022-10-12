

import numpy as np
from scipy.optimize import minimize
import math
import bayes_opt

# problem 2


def fun(x):
    x1 = x[0]
    x2 = x[1]
    return (4-2.1 * x1**2 + (x1**4)/3)*(x2**2) + x1*x2 + (-4 + 4 * x2**2)*x2**2
# Check function


check = [1, 1]
pbounds = 1;

print(fun(check))





