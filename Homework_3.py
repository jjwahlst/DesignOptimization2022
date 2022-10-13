# Homework 3

import numpy as np
from scipy.optimize import minimize
import math
import bayes_opt
from bayes_opt import BayesianOptimization
import torch
from torch.autograd import Variable
import warnings

# Problem 1


X1 = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
X2 = np.flip(x1, axis=1).copy()

a = np.array([[8.07131, 1730.63, 233.426], [7.43155, 1554.679, 240.337]])
T = 20

p1sat = 10 ** (a[0, 0] - a[0, 1] / (20 + a[0, 2]))
p2sat = 10 ** (a[1, 0] - a[1, 1] / (20 + a[1, 2]))
P = np.array([28.1, 34.4, 36.7, 36.9, 36.8, 36.7, 36.5, 35.4, 32.9, 27.7, 17.5])
P = torch.tensor(P, requires_grad=False, dtype=torch.float32)

A = Variable(torch.tensor([1.0, 1.0]), requires_grad=True)

x1 = torch.tensor(X1, requires_grad=False, dtype=torch.float32)
x2 = torch.tensor(X2, requires_grad=False, dtype=torch.float32)

a = 0.0001

for i in range(100):
    P_pred = x1 * torch.exp(A[0] * ((A[1] * x2) / (A[0] * x1 + A[1] * x2)) ** 2) * p1sat + x2 * torch.exp(
        A[1] * ((A[1] * x2) / (A[0] * x1 + A[1] * x2))) * p2sat
    loss = (P_pred - P) ** 2
    loss = loss.sum()

    loss.backward()

    with torch.no_grad():
        A -= a * A.grad()

        A.grad.zero()

print('Esimation A12, A21 is:', A)
print('FInal loss is:', loss.data.numpy())




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
