import math
import numpy as np


def random_samples_float(min, max, n):
    """

    """
    assert min < max
    return (max - min) * np.random.random_sample(n) + min


def random_samples_int(min, max, n):
    """

    """
    assert min < max
    return np.random.randint(min, max, n)


def evenly_spaced_grid(start, stop, step):
    """

    """
    n = math.floor((abs(start) + abs(stop)) / step)
    return np.linspace(start, stop, n)
