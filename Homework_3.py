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
X2 = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])

a = np.array([[8.07131, 1730.63, 233.426], [7.43155, 1554.679, 240.337]])
# print(a)


T = 20

p1sat = 10 ** (a[0, 0] - (a[0, 1] / (T + a[0, 2])))
p2sat = 10 ** (a[1, 0] - (a[1, 1] / (T + a[1, 2])))
print(p1sat)
print(p2sat)
P = np.array([28.1, 34.4, 36.7, 36.9, 36.8, 36.7, 36.5, 35.4, 32.9, 27.7, 17.5])
P = torch.tensor(P, requires_grad=False, dtype=torch.float32)

A = Variable(torch.tensor([1.0, 1.0]), requires_grad=True)

x1 = torch.tensor(X1, requires_grad=False, dtype=torch.float32)
x2 = torch.tensor(X2, requires_grad=False, dtype=torch.float32)
# print(x1)
# print(x2)


a = 0.0001

for i in range(100):
    P_pred = x1 * torch.exp(A[0] * ((A[1] * x2) / (A[0] * x1 + A[1] * x2)) ** 2) * p1sat + x2 * torch.exp(
        A[1] * ((A[0] * x1) / (A[0] * x1 + A[1] * x2)) ** 2) * p2sat
    loss = (P_pred - P) ** 2
    loss = loss.sum()
    loss.backward()
    with torch.no_grad():
        A -= a * A.grad
        A.grad.zero_()

print('Esimation A12, A21 is:', A)
print('FInal loss is:', loss.data.numpy())

import matplotlib.pyplot as plt

P_pred = P_pred.detach()
P = P.detach()
x1 = x1.detach()

plt.plot(x1, P_pred, label='Predicted Pressure')
plt.plot(x1, P, label='Actual Pressure')
plt.xlabel('x1')
plt.ylabel('Pressure')
plt.legend()
plt.title('Comparison between predicted pressure and actual pressure')
plt.show()


# CHECK
# print('P1sat', p1sat, 'P2sat', p2sat)


# Problem 2
def fun(x1, x2):
    return -((4 - 2.1 * x1 ** 2 + (x1 ** 4) / 3) * (1 ** 2) + x1 * x2 + (-4 + 4 * x2 ** 2) * x2 ** 2)


# Check function
# print('Test=', fun(1, 2))
# Do the optimization


search_space = {'x1': (-3, 3), 'x2': (-2, 2)}
optimizer = BayesianOptimization(f=fun, pbounds=search_space, random_state=1234, verbose=1)
optimizer.maximize(init_points=100, n_iter=15)

# All done
