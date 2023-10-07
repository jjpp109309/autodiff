import numpy as np

from .tensor import Function, Tensor
from .elemental_functions import (
    multiply,
    exp,
    pow,
    add
)

@Function
def sigmoid(x: Tensor) -> Tensor:
    z = pow(add(), -1)

    return z, z_adjoint

