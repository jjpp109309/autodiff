from enum import Enum, auto


class Operation(Enum):
    SUM = auto()
    SUB = auto()

Operation = Enum('operation', ['SUM', 'SUB'])


x = Operation.SUM

match x:
    case Operation.SUM:
        print(3)
    case Operation.SUB:
        print(1)
