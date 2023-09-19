# Target f(x) = sin(x)
# g(x) = sin(ax)

import numpy as np


def func(x):
    return np.sin(x)


def loss(a, x_data, y_data):
    return np.mean(np.abs(np.sin(a * x_data) - y_data))


def loss_derivative(a, x_data, y_data):
    return np.mean(x_data * np.cos(a * x_data) * np.sign(np.sin(a * x_data) - y_data))


# Hyperparameters
min_range = 0
max_range = np.pi
data_size = 10000
alpha = 0.001
max_attempts = 100000
epsilon = np.finfo(float).eps

x_values = np.linspace(min_range, max_range, data_size)
y_values = func(x_values)

a_pre = 2
a_now = a_pre

for attempt in range(max_attempts):
    gradient = loss_derivative(a_pre, x_values, y_values)
    a_now = a_pre - alpha * gradient
    a_pre = a_now

    loss_value = loss(a_now, x_values, y_values)

    if loss_value < epsilon:
        break

    if attempt % 10 == 0:
        print(f"Attempt {attempt}: a = {a_now}, Loss = {loss_value}")

print(f"Final: a = {a_now}")