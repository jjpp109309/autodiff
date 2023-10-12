import numpy as np


# https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation
class Variable:

    def __init__(self, value, children=None):
        self.value = value
        self.children = children or []
        self.gradient = 0

    def grad(self, gradient=1):
        self.gradient += gradient

        for adjoint, child in self.children:
            child.grad(adjoint * gradient)

    def __add__(self, other):
        value = self.value + other.value
        children = [(1, self), (1, other)]

        return Variable(value, children)

    def __sub__(self, other):
        value = self.value - other.value
        children = [(1, self), (-1, other)]

        return Variable(value, children)

    def __mul__(self, other):
        value = self.value * other.value
        children = [(other.value, self), (self.value, other)]

        return Variable(value, children)

    def __repr__(self):
        return str(self.value)

    def __str__(self):
        return str(self.value)


class Tensor:

    def __init__(self, value, children=None):
        self.value = np.array(value)
        self.children = children or []
        self.gradient = 0

    def grad(self, gradient=1):
        self.gradient += gradient

        for adjoint, child in self.children:
            child.grad(adjoint * gradient)

    def __add__(self, other):
        value = self.value + other.value
        children = [
            (np.ones_like(self.value), self),
            (np.ones_like(other.value), other)
        ]

        return Tensor(value, children)

    def __sub__(self, other):
        value = self.value - other.value
        children = [
            (np.ones_like(self.value), self),
            (-np.ones_like(other.value), other)
        ]

        return Tensor(value, children)

    def __mul__(self, other):
        value = self.value * other.value
        children = [(other.value, self), (self.value, other)]

        return Tensor(value, children)

    def __neg__(self):
        value = -self.value
        children = [(-np.ones_like(self.value), self)]

        return Tensor(value, children)

    def __repr__(self):
        return str(self.value)

    def __str__(self):
        return str(self.value)


def sin(x: Tensor) -> Tensor:
    value = np.sin(x.value)
    children = [(np.cos(x.value), x)]

    return Tensor(value, children)


def cos(x: Tensor):
    value = np.cos(x.value)
    children = [(-np.sin(x.value), x)]

    return Tensor(value, children)


def exp(x: Tensor):
    value = np.exp(x.value)
    children = [(np.exp(x.value), x)]

    return Tensor(value, children)
