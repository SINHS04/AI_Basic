# A program to find a closest x where f(x) equals to target
# f(x) = x * x

import numpy as np
import sys


def func(x):
    return x * x


def derivative(x):
    return 2 * x


alpha = 0.0000001
attempt = 100000
err = sys.float_info.epsilon

data_size = 10000
data = np.array([func(x) for x in np.arange(-2, 2, 4/data_size)])
print(data)
print(data[int(data_size/2)])
a_now = 0
a_pre = a_now


def loss_derivative(a):
    return np.sum(np.array([np.sqrt(2 * (a - 1) * np.power(x, 4)) for x in np.arange(-2, 2, 4/data_size)]))


def loss(a):
    return np.sum(np.array([np.sqrt(np.power(x*x*(1-a), 2)) for x in np.arange(-2, 2, 4/data_size)]))


for i in range(attempt):
    minus = alpha * loss_derivative(a_pre)
    a_now = a_pre - minus
    a_pre = a_now
    y = loss(a_now)
    if np.abs(y) < err:
        break
    
    print("attempt {} : ({}, {}, {})".format(i, a_now, y, minus))

print("answer : ({}, {})".format(a_now, func(a_now)))