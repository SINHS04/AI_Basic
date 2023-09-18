# A program to find a closest x where f(x) equals to target(0)
# f(x) = x * x

import numpy as np
import sys

alpha = 0.01
target = 0
attempt = 1000000
err = sys.float_info.epsilon

x_pre = 8 # init value
x_now = x_pre


def func(x):
    return x * x


def derivative(x):
    return 2 * x


for i in range(attempt):
    x_now = x_pre - alpha * derivative(x_pre)
    x_pre = x_now
    y = func(x_now)
    if np.abs(y - target) < err:
        break
    
    print("attempt {} : ({}, {})".format(i, x_now, y))

print("answer : ({}, {})".format(x_now, func(x_now)))
# alpha = 0.1,  answer : (0.0008507059173023465, 7.237005577332268e-07)
# alpha = 0.01, answer : (0.0009969965836718114, 9.940021878532632e-07)

# alpha = 0.1,   answer : (1.2141680576410815e-08, 1.4742040721959166e-16)
# alpha = 0.01,  answer : (1.4894821432797048e-08, 2.218557055149103e-16)
# alpha = 0.001, answer : (1.4888640838838194e-08, 2.2167162602792048e-16)

'''
 - Has big difference between alpha is 0.1 and 0.01
 - But not that big difference between alpha is 0.01 and 0.001
 - Compare with attempt, 0.01 is less than 1,000, but 0.001 is more than 10,000
 - It's important to make a best fit of alpha
'''