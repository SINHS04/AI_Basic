# A program to find a closest x where f(x) equals to 2
# f(x) = x * x + 2

import numpy as np

alpha = 0.1
target = 2
attempt = 1000
err = 0.000001

x_pre = 8 # init value
x_now = x_pre

def func(x) :
    return x * x + 2

for i in range(attempt):
    x_now = x_pre - alpha * (2 * x_pre)
    if np.abs(func(x_now)-target) < err:
        break
    x_pre = x_now
    print("attemp {} : ({}, {})".format(i, x_now, func(x_now)))

print("answer : ({}, {})".format(x_now, func(x_now)))