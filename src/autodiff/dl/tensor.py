import numpy as np

from typing import List, Tuple


class Tensor:
    # based on: https://www.jmlr.org/papers/volume18/17-468/17-468.pdf

    def __init__(self, value):
        self.value = np.array(value)
        self.adjoints: List[Tuple[Tensor, float]] = None
        self.gradient = np.ones_like(self.value)

    def backward(self):
        # backprop
        for child, adjoint in self.adjoints:
            child.gradient += self.gradient * adjoint

            if child.adjoints:
                child.backward()

    def __repr__(self):
        return self.value.__repr__()


class Function:
    def __init__(self, func: callable):
        self.func = func

    def __call__(self, *args: Tensor) -> Tensor:

        value, adjoints = self.func(*[i.value for i in args])

        y = Tensor(value)
        y.adjoints = [(arg, adjoint) for arg, adjoint in zip(args, adjoints)]

        for arg in args:
            arg.gradient = 0

        return y
