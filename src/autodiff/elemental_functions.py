import numpy as np
from .tensor import Function


@Function
def add(x1, x2):
    z = x1 + x2
    z_adjoint = [np.ones_like(x1), np.ones_like(x2)]

    return z, z_adjoint


@Function
def subtract(x1, x2):
    z = x1 - x2
    z_adjoint = [np.ones_like(x1), -np.ones_like(x2)]

    return z, z_adjoint


@Function
def multiply(x1, x2):
    z = x1 * x2
    z_adjoint = [x2, x1]

    return z, z_adjoint


@Function
def pow(x, n=None):
    z = np.power(x, n)
    z_adjoint = [n * np.power(x, n-1)]

    return z, z_adjoint


@Function
def log(x):
    z = np.log(x)
    z_adjoint = [1 / x]

    return z, z_adjoint


@Function
def sin(x):
    z = np.sin(x)
    z_adjoint = [np.cos(x)]

    return z, z_adjoint


@Function
def cos(x):
    z = np.cos(x)
    z_adjoint = [np.sin(x)]

    return z, z_adjoint


@Function
def mm(x1, x2):
    z = np.dot(x1, x2)
    z_adjoint = [x2, x1]

    return z, z_adjoint


@Function
def exp(x):
    z = np.exp(x)
    z_adjoint = [z]

    return z, z_adjoint
