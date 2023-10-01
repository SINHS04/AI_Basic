# A program to find a, cloeset with f(x)
# f(x) = sin(x)
# g(x) = sin(ax)

import numpy as np
import sys


def func(x):
    return np.sin(x)


min_range = 0
max_range = np.pi
lr = 0.001
attempt = 100000
err = sys.float_info.epsilon

data_size = 10000
x_data = np.arange(min_range, max_range, (max_range - min_range)/data_size)
y_data = func(np.arange(min_range, max_range, (max_range - min_range)/data_size))
a = 3


def loss_func(a):
    return np.sin(a * x_data) - y_data


def loss_func_derivate(a):
    return x_data * np.cos(a * x_data) * np.sign(loss_func(a))


def loss(a):
    return np.mean(np.abs(loss_func(a)))


def gradient(a):
    return np.mean(loss_func_derivate(a))


for i in range(attempt):
    a = a - lr * gradient(a)
    y = loss(a)
    if np.abs(y) < err:
        break
    
    print("attempt {} : ({}, {})".format(i, a, y)) # print attempt contains current 'a' value and loss value

print("answer : ({}, {})".format(a, func(a)))