import math
import numpy as np
from math import sqrt

def absolute_error(actual, prediction):
    """

    """
    return np.sum(np.abs(np.subtract(actual, prediction)))


def mean_squared_error(actual, prediction):
    """

    """
    return np.square(np.subtract(actual, prediction)).mean()


def root_mean_squared_error(actual, prediction):
    """

    """
    return sqrt(mean_squared_error(actual, prediction))

def calculate_fitness(actual, prediction, method="abs"):
    """

    """
    match method:
        case "abs":
            return absolute_error(actual, prediction)
        case "mse":
            return mean_squared_error(actual, prediction)
        case "rmse":
            return root_mean_squared_error(actual, prediction)

def is_better(fitness1, fitness2, minimizing_fitness=True, strict=True):
    if minimizing_fitness:
        if strict:
            if fitness1 < fitness2: return True
        else:
            if fitness1 <= fitness2: return True
    else:
        if strict:
            if fitness1 > fitness2: return True
        else:
            if fitness1 >= fitness2: return True
    return False

def is_ideal(fitness, ideal_fitness, minimizing_fitness=True):
    if minimizing_fitness:
        return fitness <= ideal_fitness
    else:
        return fitness >= ideal_fitness
