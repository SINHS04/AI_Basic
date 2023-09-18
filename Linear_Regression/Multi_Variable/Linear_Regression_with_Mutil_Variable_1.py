# A program to find a and b, cloeset with f(x)
# f(x) = x**2 + 2
# g(x) = a * x**2 + b

import numpy as np
import sys


def func(x):
    return x**2 + 2


min_range = -2
max_range = 2
alpha = 0.0000001
attempt = 100000
err = sys.float_info.epsilon

data_size = 10000
x_data = np.arange(min_range, max_range, (max_range - min_range)/data_size)

a_now = 0
a_pre = a_now
b_now = 0
b_pre = b_now


def loss_derivative_a(a, b):
    return np.sum(np.array([2 * (x**2 * (a - 1) + b - 2) * x**2 for x in x_data]))


def loss_derivative_b(a, b):
    return np.sum(np.array([2 * (x**2 * (a - 1) + b - 2) for x in x_data]))


def loss(a, b):
    return np.sum(np.array([(x**2 * (a - 1) + 2 - b)**2 for x in x_data]))


for i in range(attempt):
    a_now = a_pre - alpha * loss_derivative_a(a_pre, b_pre)
    b_now = b_pre - alpha * loss_derivative_b(a_pre, b_pre)
    a_pre = a_now
    b_pre = b_now
    y = loss(a_now, b_now)
    if np.abs(y) < err:
        break
    
    print("attempt {} : (({}, {}), {})".format(i, a_now, b_now, y))

print("answer : ({}, {})".format(a_now, b_now))