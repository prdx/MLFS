import numpy as np

def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))

def logistic_derivative(x):
    return logistic(x) * (1.0 - logistic(x))

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return (1.0 - (tanh(x)**2))

def relu(x):
    return np.maximum(x, 0, x)

def relu_derivative(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x

def softplus(x):
    return np.log(1.0 + np.exp(x))

def softplus_derivative(x):
    return 1.0 / (1.0 + np.exp(-x))
