# Copyright (C) 2023 -
# Roman Kalkreuth (Roman.Kalkreuth@lip6.fr)
# Computer Lab of Paris 6, Sorbonne Université (Paris, France)

__author__ = 'Roman Kalkreuth'
__copyright__ = 'Copyright (C) 2023, Roman Kalkreuth'
__version__ = '1.0'
__email__  = 'Roman.Kalkreuth@lip6.fr'

import math

def add(x, y):
    return x + y

def sub(x, y):
    return x - y

def mul(x, y):
    return x * y


def div(x, y):
    if y == 0:
        return 1.0
    return x / y

def sin(x):
    return math.sin(x)

def cos(x):
    return math.cos(x)

def exp(x):
    return math.exp(x)
def log(x):
    return math.log(abs(x))

