# A program to find a, cloeset with f(x)
# f(x) = sin(x)
# g(x) = sin(ax)

import numpy as np
import sys


def func(x):
    return np.sin(x)


min_range = 0
max_range = np.pi
alpha = 0.000001
attempt = 100000
err = sys.float_info.epsilon

data_size = 10000
data = np.arange(min_range, max_range, (max_range - min_range)/data_size)
a_now = 2
a_pre = a_now


def loss_derivative(a):
    return np.sum(np.array([loss_derivative_function(a, x) if np.sign(loss_function(a, x)) == 1 else -loss_derivative_function(a, x) for x in data]))


def loss(a):
    return np.sum(np.array([np.abs(loss_function(a, x)) for x in data]))


def loss_derivative_function(a, x):
    return x * np.cos(a*x)


def loss_function(a, x):
    return np.sin(a*x) - np.sin(x)


for i in range(attempt):
    a_now = a_pre - alpha * loss_derivative(a_pre)
    a_pre = a_now
    y = loss(a_now)
    if np.abs(y) < err:
        break
    
    print("attempt {} : ({}, {})".format(i, a_now, y)) # print attempt contains current 'a' value and loss value

print("answer : ({}, {})".format(a_now, func(a_now)))