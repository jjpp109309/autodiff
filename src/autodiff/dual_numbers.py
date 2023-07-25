import numpy as np


class DualNumber():

    def __init__(u, x, y):
        u.x = x
        u.y = y

    def __repr__(u):
        return f'({u.x}, {u.y})'

    def __add__(u, v):
        x = u.x + v.x
        y = u.y + v.y

        return DualNumber(x, y)

    def __sub__(v, u):
        x = u.x - v.x
        y = u.y - v.y

        return DualNumber(x, y)

    def __mul__(v, u):
        x = u.x * v.x
        y = u.x * v.y + u.y * v.x

        return DualNumber(x, y)

    def __truediv__(v, u):
        x = u.x / v.y
        y = (u.x * v.y - u.y * v.x) / v.y ** 2

        return DualNumber(x, y)


def log(u: DualNumber) -> DualNumber:
    x = np.log(u.x)
    y = u.y / u.x

    return DualNumber(x, y)


def sin(u: DualNumber) -> DualNumber:
    x = np.sin(u.x)
    y = np.cos(u.x) * u.y

    return DualNumber(x, y)


def cos(u: DualNumber) -> DualNumber:
    x = np.cos(u.x)
    y = -np.sin(u.x) * u.y

    return DualNumber(x, y)


def exp(u: DualNumber) -> DualNumber:
    x = np.exp(u.x)
    y = np.exp(u.x) * u.y

    return DualNumber(x, y)
