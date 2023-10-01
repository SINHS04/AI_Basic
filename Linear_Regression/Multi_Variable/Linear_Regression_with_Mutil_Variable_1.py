# A program to find a and b, cloeset with f(x)
# f(x) = x**2 + 2
# g(x) = a * x**2 + b

import numpy as np
import sys


def func(x):
    return x**2 + 2


min_range = -2
max_range = 2
alpha = 0.000001
attempt = 10001
err = sys.float_info.epsilon

data_size = 10000
x_data = np.arange(min_range, max_range, (max_range - min_range) / data_size)

a = 5
b = 5


def loss_derivative(a, b):
    tmp = 2 * (x_data**2 * (a - 1) + b - 2)
    return np.sum(tmp * x_data**2), np.sum(tmp)


def loss(a, b):
    return np.sum((x_data**2 * (a - 1) + 2 - b) ** 2)


for i in range(attempt):
    gradient_a, gradient_b = loss_derivative(a, b)
    a = a - alpha * gradient_a
    b = b - alpha * gradient_b
    y = loss(a, b)
    if np.abs(y) < err:
        break

    if i % 200 == 0:
        print("attempt {} : (({}, {}), lose : {})".format(i, a, b, y))

print("answer : ({}, {})".format(a, b))
