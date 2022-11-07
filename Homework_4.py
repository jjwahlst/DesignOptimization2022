# Homework 4

import numpy as np
from scipy.optimize import minimize
import math
import bayes_opt
from bayes_opt import BayesianOptimization
import torch
from torch.autograd import Variable
import warnings
import torch as t
import torch.nn as nn
from torch import optim
from torch.nn import utils
import matplotlib.pyplot as plt


def fun(x):
    x1 = x[0]
    x2 = x[1]
    return (x1 + 1) ** 2 + (x2 - 2) ** 2

def ST1(x):
    return x1 - 2 <= 0
def ST2(x):
    return x2 - 1 <= 0
def ST3(x):
    return -x1 <= 0
def ST4(x):
    return -x2 <= 0





