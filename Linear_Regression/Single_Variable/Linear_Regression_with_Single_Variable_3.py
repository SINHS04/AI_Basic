# A program to find a, cloeset with f(x)
# f(x) = sin(x)
# g(x) = sin(ax)

import numpy as np
import sys


def func(x):
    return np.sin(x)


min_range = 0
max_range = np.pi
data_range = max_range - min_range
alpha = 0.000001
attempt = 100000
err = sys.float_info.epsilon

data_size = 10000
data = np.array([func(x) for x in np.arange(min_range, max_range, data_range/data_size)])
a_now = 0.9999
a_pre = a_now


def loss_derivative(a):
    return np.sum(np.array([loss_derivative_f(a, x) if np.sign(loss_f(a, x)) == 1 else -loss_derivative_f(a, x) for x in range(data_size)])) / data_size


def loss(a):
    return np.sum(np.array([np.abs(loss_f(a, x)) for x in range(data_size)])) / data_size


def loss_derivative_f(a, x):
    return data_range * x * np.cos(a * data_range * x / data_size) / data_size


def loss_f(a, x):
    return np.sin(a * data_range * x / data_size) - data[x]


for i in range(attempt):
    a_now = a_pre - alpha * loss_derivative(a_pre)
    a_pre = a_now
    y = loss(a_now)
    if np.abs(y) < err:
        break
    
    print("attempt {} : ({}, {})".format(i, a_now, y)) # print attempt contains current 'a' value and loss value

print("answer : ({}, {})".format(a_now, func(a_now)))