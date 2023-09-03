import numpy as np
from .elemental_functions import mm, add

from .tensor import Tensor


class Layer(object):

    def __init__(self):
        self.parameters = []

    def get_parameters(self):
        return self.parameters

    def forward(self, x):
        pass

    def __call__(self, x):
        return self.forward(x)


class Linear(Layer):

    def __init__(self, n_inputs, n_outputs):
        super.__init__()

        weights = np.random.normal(size=(n_outputs, n_inputs))
        weights *= np.sqrt(2 / n_inputs)
        self.weight = Tensor(weights)

        self.bias = Tensor(np.zeros(n_outputs))

    def forward(self, x):
        return add(mm(self.weight, x), self.bias)
