import numpy as np
from math import sqrt

def absolute_error(actual, prediction):
    return np.sum(np.abs(np.subtract(actual, prediction)))


def mean_squared_error(actual, prediction):
    return np.square(np.subtract(actual, prediction)).mean()


def root_mean_squared_error(actual, prediction):
    return sqrt(mean_squared_error(actual, prediction))