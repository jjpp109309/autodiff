import numpy as np

from typing import List, Tuple


class Tensor:

    def __init__(self, value):
        self.value = np.array(value)
        self.adjoints: List[Tuple[Tensor, float]] = None
        self.gradient = np.ones_like(self.value)

    def grad(self):
        # backprop
        for child, adjoint in self.adjoints:
            child.gradient += self.gradient * adjoint

            if child.adjoints:
                child.grad()
