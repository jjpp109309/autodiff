import numpy as np
class DualNumber():

    def __init__(self, x, y):
        self.x = x
        self.y = y

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
