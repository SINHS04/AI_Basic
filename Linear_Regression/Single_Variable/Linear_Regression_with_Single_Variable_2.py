# A program to find a, cloeset with f(x)
# f(x) = x**2
# g(x) = a * x**2

import numpy as np
import sys


def func(x):
    return x**2


def derivative(x):
    return 2 * x


min_range = -5
max_range = 5
alpha = 0.0000001
attempt = 100000
err = sys.float_info.epsilon

data_size = 10000
data = np.array([func(x) for x in np.arange(min_range, max_range, (max_range - min_range)/data_size)])
a_now = -15
a_pre = a_now


def loss_derivative(a):
    return np.sum(np.array([2 * (a - 1) * np.power(x, 4) for x in np.arange(min_range, max_range, (max_range - min_range)/data_size)]))


def loss(a):
    return np.sum(np.array([np.power(x*x*(1-a), 2) for x in np.arange(min_range, max_range, (max_range - min_range)/data_size)]))


for i in range(attempt):
    minus = alpha * loss_derivative(a_pre)
    a_now = a_pre - minus
    a_pre = a_now
    y = loss(a_now)
    if np.abs(y) < err:
        break
    
    print("attempt {} : ({}, {}, {})".format(i, a_now, y, minus))

print("answer : ({}, {})".format(a_now, func(a_now)))