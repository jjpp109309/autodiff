import numpy as np

from typing import List
from collections import defaultdict
from enum import Enum, auto


class Operation(Enum):
    SUM: auto()
    NEG: auto()
    SUB: auto()


class Tensor:
    def __init__(self, data: np.ndarray, autograd=False,
                 creators: List = None,
                 creation_op: str = None, id: int = None):

        self.data = np.array(data)
        self.autograd = autograd
        self.creators = creators
        self.creation_op = creation_op
        self.id = id if id else np.random.randint(0, 1000000)

        self.grad = None
        self.children = defaultdict(lambda: 0)

        if creators:
            for c in creators:
                c.children[self.id] += 1

    def all_children_grads_accounted_for(self):

        accounted = True
        for id, count in self.children.items():
            if count:
                accounted = False
                break

        return accounted

    def backward(self, grad=None, grad_origin=None):

        if not grad and not grad_origin:
            grad = Tensor(np.ones_like(self.data))

        if self.autograd:
            if grad_origin:
                if self.children[grad_origin.id]:
                    self.children[grad_origin.id] -= 1
                else:
                    raise Exception("cannot backprop more than once")

            self.grad = self.grad + grad if self.grad else grad

            keep_propagating = (
                self.creators and (
                    self.all_children_grads_accounted_for()
                    or
                    not grad_origin
                )
            )

            if keep_propagating:
                match self.creation_op:
                    case Operation.SUM:
                        for creator in self.creators:
                            creator.backward(self.grad, self)

                    case Operation.NEG:
                        self.creators[0].backward(self.grad.__neg__())

    def __add__(self, other):

        autograd_on = self.autograd and other.autograd
        kwargs = {
            'autograd': True if autograd_on else False,
            'creators': [self, other] if autograd_on else None,
            'creation_op': Operation.SUM if autograd_on else None,
        }

        return Tensor(self.data + other.data, **kwargs)

    def __neg__(self):
        autograd_on = self.autograd
        kwargs = {
            'autograd': True if autograd_on else False,
            'creators': [self] if autograd_on else None,
            'creation_op': Operation.NEG if autograd_on else None,
        }

        return Tensor(-1 * self.data, **kwargs)

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())


class Function:
    def __init__(self, func: callable):
        self.func = func

    def __call__(self, *args: Tensor, **kwargs) -> Tensor:

        value, adjoints = self.func(*[i.value for i in args], **kwargs)

        y = Tensor(value)
        y.adjoints = [(arg, adjoint) for arg, adjoint in zip(args, adjoints)]

        for arg in args:
            arg.gradient = 0

        return y
