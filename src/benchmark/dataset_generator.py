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

def generate_function_values(function, data_points):
    """

    """
    num_instances = len(data_points)
    function_values = np.empty(num_instances)

    for index, data_points in enumerate(data_points):
        function_values[index] = function(data_points)

    return function_values

def concatenate_data(data_points, function_values):
    """

    """
    return np.stack((data_points, function_values), axis=1)