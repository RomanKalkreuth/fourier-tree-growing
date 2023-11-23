import math
import numpy as np


def random_samples_float(min, max, n, dim=1):
    assert min < max
    samples = []

    for i in range(0, dim):
        sample = (max - min) * np.random.random_sample(n) + min
        samples.append(sample)

    return np.stack(samples, axis=1)

def random_samples_int(min, max, n, dim=1):
    assert min < max
    samples = []

    for i in range(0, dim):
        sample = np.random.randint(min, max, n)
        samples.append(sample)

    return np.stack(samples, axis=1)


def evenly_spaced_grid(start, stop, step):
    n = math.floor((abs(start) + abs(stop)) / step)
    return np.linspace(start, stop, n)


def generate_function_values(function, data_points: np.array) -> np.array:
    num_instances = len(data_points)
    function_values = np.empty(num_instances)
    dim = data_points.shape[0]

    for i, dp in enumerate(data_points):
        if dim == 1:
            function_values[i] = function(dp[0])
        else:
            function_values[i] = function(dp)

    return function_values


def stack_data(arr1, arr2):
    return np.column_stack((arr1, arr2))
