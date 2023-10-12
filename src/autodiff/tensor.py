import numpy as np

from typing import List
from collections import defaultdict
from enum import Enum


class Operation(Enum):
    ADD = 1
    NEG = 2
    SUB = 3
    MUL = 4
    EXPAND = 5
    TR = 6
    MM = 7
    SUM0 = 8
    SUM1 = 9
    SUM2 = 10
    SUM3 = 11
    SUM4 = 12
    SUM5 = 13
    EXPAND1 = 14
    EXPAND2 = 15
    EXPAND3 = 16
    EXPAND4 = 17
    EXPAND5 = 18
    TRANSPOSE = 19


class Tensor:
    # from: https://www.manning.com/books/grokking-deep-learning

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
                    case Operation.ADD:
                        for creator in self.creators:
                            creator.backward(self.grad, self)

                    case Operation.NEG:
                        self.creators[0].backward(self.grad.__neg__())

                    case Operation.SUB:
                        new = Tensor(self.grad.data)
                        self.creators[0].backward(new, self)
                        new = Tensor(self.grad.__neg__().data)
                        self.creators[1].backward(new, self)

                    case Operation.MUL:
                        new = self.grad * self.creators[1]
                        self.creators[0].backward(new, self)
                        new = self.grad * self.creators[0]
                        self.creators[1].backward(new, self)

                    case Operation.MM:
                        act = self.creators[0]
                        weights = self.creators[1]
                        new = self.grad.mm(weights.transpose())
                        act.backward(new)
                        new = self.grad.transpose().mm(act).transpose()
                        weights.backward(new)

                    case Operation.TRANSPOSE:
                        self.creators[0].backward(self.grad.transpose())

                    case Operation.SUM0:
                        dim = 0
                        ds = self.creators[0].data.shape[0]
                        self.creators[0].backward(self.grad.expand(dim, ds))

                    case Operation.SUM1:
                        dim = 1
                        ds = self.creators[0].data.shape[1]
                        self.creators[0].backward(self.grad.expand(dim, ds))

                    case Operation.SUM2:
                        dim = 2
                        ds = self.creators[0].data.shape[2]
                        self.creators[0].backward(self.grad.expand(dim, ds))

                    case Operation.SUM3:
                        dim = 3
                        ds = self.creators[0].data.shape[3]
                        self.creators[0].backward(self.grad.expand(dim, ds))

                    case Operation.SUM4:
                        dim = 4
                        ds = self.creators[0].data.shape[4]
                        self.creators[0].backward(self.grad.expand(dim, ds))

                    case Operation.SUM5:
                        dim = 5
                        ds = self.creators[0].data.shape[5]
                        self.creators[0].backward(self.grad.expand(dim, ds))

                    case Operation.EXPAND1:
                        dim = 1
                        self.creators[0].backward(self.grad.sum(dim))

                    case Operation.EXPAND2:
                        dim = 2
                        self.creators[0].backward(self.grad.sum(dim))

                    case Operation.EXPAND3:
                        dim = 3
                        self.creators[0].backward(self.grad.sum(dim))

                    case Operation.EXPAND4:
                        dim = 4
                        self.creators[0].backward(self.grad.sum(dim))

                    case Operation.EXPAND1:
                        dim = 5
                        self.creators[0].backward(self.grad.sum(dim))


    def __add__(self, other):

        autograd_on = self.autograd and other.autograd
        kwargs = {
            'autograd': True if autograd_on else False,
            'creators': [self, other] if autograd_on else None,
            'creation_op': Operation.ADD if autograd_on else None,
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

    def __sub__(self, other):
        autograd_on = self.autograd and other.autograd
        kwargs = {
            'autograd': True if autograd_on else False,
            'creators': [self, other] if autograd_on else None,
            'creation_op': Operation.SUB if autograd_on else None,
        }

        return Tensor(self.data - other.data, **kwargs)

    def __mul__(self, other):
        autograd_on = self.autograd and other.autograd
        kwargs = {
            'autograd': True if autograd_on else False,
            'creators': [self, other] if autograd_on else None,
            'creation_op': Operation.MUL if autograd_on else None,
        }

        return Tensor(self.data * other.data, **kwargs)

    def sum(self, dim):
        autograd_on = self.autograd
        kwargs = {
            'autograd': True if autograd_on else False,
            'creators': [self] if autograd_on else None,
            'creation_op': 'SUM{}'.format(dim) if autograd_on else None,
        }

        return Tensor(self.data.sum(dim), **kwargs)

    def expand(self, dim, copies):

        trans_cmd = list(range(len(self.data.shape)))
        trans_cmd.insert(dim, len(self.data.shape))
        new_shape = list(self.data.shape) + [copies]
        new_data = self.data.repeat(copies).reshape(new_shape)
        new_data = new_data.transpose(trans_cmd)

        autograd_on = self.autograd
        kwargs = {
            'autograd': True if autograd_on else False,
            'creators': [self] if autograd_on else None,
            'creation_op': 'EXPAND{}'.format(dim) if autograd_on else None,
        }

        return Tensor(new_data, **kwargs)

    def transpose(self):
        autograd_on = self.autograd
        kwargs = {
            'autograd': True if autograd_on else False,
            'creators': [self] if autograd_on else None,
            'creation_op': 'TRANSPOSE' if autograd_on else None,
        }

        return Tensor(self.data.T(), **kwargs)

    def mm(self, x):
        autograd_on = self.autograd
        kwargs = {
            'autograd': True if autograd_on else False,
            'creators': [self] if autograd_on else None,
            'creation_op': 'TRANSPOSE' if autograd_on else None,
        }

        return Tensor(self.data.dot(x.data), **kwargs)

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


# https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation
class Variable:

    def __init__(self, value):
        self.value = value
        self.children = []
        self.grad_value = []

    def grad(self):
        if not self.grad_value:
            self.grad_value = sum(w * var.grad() for w, var in self.children)

        return self.grad_value

    def __add__(self, other):
        z = Variable(self.value + other.value)

        self.children.append((1, z))
        self.other.append((1, z))

        return z

    def __mul__(self, other):
        z = Variable(self.value * other.value)

        self.children.append((other.value, z))
        other.children.append((self.value, z))

        return z

    def __repr__(self):
        return str(self.value)

    def __str__(self):
        return str(self.value)



