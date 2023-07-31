from typing import List, Tuple


class Variable:

    def __init__(self, value):
        self.value = value
        self.adjoints: List[Tuple[Variable, float]] = None
        self.gradient = 1

    def grad(self):
        # backprop
        for child, adjoint in self.adjoints:
            child.gradient += self.gradient * adjoint

            if child.adjoints:
                child.grad()


class Function:
    def __init__(self, func: callable):
        self.func = func

    def __call__(self, *args: Variable) -> Variable:

        value, adjoints = self.func(*[i.value for i in args])

        y = Variable(value)
        y.adjoints = [(arg, adjoint) for arg, adjoint in zip(args, adjoints)]

        for arg in args:
            arg.gradient = 0

        return y
