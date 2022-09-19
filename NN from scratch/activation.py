from math import e

def sigmoid(x):
    return 1/(1 + e**(-x))


def d_sigmoid(y):
    return y * (1 - y)


def tanh(x):
    return (e**(2*x) - 1) / (e**(2*x) + 1)


def d_tanh(y):
    return 1 - y**2


def relu(x):
     return (x if x > 0 else 0)


def d_relu(y):
    return 1 if y > 0 else 0


def lrelu(x):
     return (x if x > 0 else 0.01 * x)


def d_lrelu(y):
     return 1 if y > 0 else 0.01 