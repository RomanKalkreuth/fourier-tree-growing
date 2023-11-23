# Copyright (C) 2023 -
# Roman Kalkreuth (Roman.Kalkreuth@lip6.fr)
# Computer Lab of Paris 6, Sorbonne Université (Paris, France)

import numpy as np
from math import sqrt

__author__ = 'Roman Kalkreuth'
__copyright__ = 'Copyright (C) 2023, Roman Kalkreuth'
__version__ = '1.0'
__email__ = 'Roman.Kalkreuth@lip6.fr'


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


def calculate_fitness(actual, prediction, metric="abs"):
    """

    """
    match metric:
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


def best_fitness(individuals, minimizing_fitness="True"):
    fitnesses = [i[1] for i in individuals]
    fitnesses.sort()
    if minimizing_fitness:
        return fitnesses[0]
    else:
        return fitnesses(len(fitnesses) - 1)
