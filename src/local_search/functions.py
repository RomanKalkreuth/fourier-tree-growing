import math
from abc import ABC, abstractmethod

class Functions(ABC):
    @staticmethod
    @abstractmethod
    def arity(function):
        pass


class Mathematical(Functions):
    @staticmethod
    def add(x, y):
        return x + y

    @staticmethod
    def sub(x, y):
        return x - y

    @staticmethod
    def mul(x, y):
        return x * y

    @staticmethod
    def div(x, y):
        if y == 0:
            return 1.0
        return x / y

    @staticmethod
    def sin(x):
        return math.sin(x)

    @staticmethod
    def cos(x):
        return math.cos(x)

    @staticmethod
    def exp(x):
        return math.exp(x)

    @staticmethod
    def log(x):
        return math.log(abs(x))

    @staticmethod
    def arity(func):
        match func.__name__:
            case "add":
                return 2
            case "sub":
                return 2
            case "mul":
                return 2
            case "div":
                return 2
            case "sin":
                return 1
            case "cos":
                return 1
            case "exp":
                return 1
            case "log":
                return 1
