import numpy as np
from math import sqrt


def absolute_error(actual, prediction):
    return np.sum(np.abs(np.subtract(actual, prediction)))


def mean_squared_error(actual, prediction):
    return np.square(np.subtract(actual, prediction)).mean()


def root_mean_squared_error(actual, prediction):
    return sqrt(mean_squared_error(actual, prediction))


def is_better(cost1, cost2, minimizing=True, strict=True):
    if minimizing:
        if strict:
            if cost1 < cost2: return True
        else:
            if cost1 <= cost2: return True
    else:
        if strict:
            if cost1 > cost2: return True
        else:
            if cost1 >= cost2: return True
    return False


def is_ideal(cost, ideal_cost, minimizing=True):
    if minimizing:
        return cost <= ideal_cost
    else:
        return cost >= ideal_cost


def evaluate(tree, X, y, f_eval):
    cost = 0.0
    dim = len(X)
    for i, x in enumerate(X):
        y_pred = tree.evaluate(x)
        loss = f_eval(y_pred, y[i])
        cost += loss
    return cost


def evaluate_sample(sample, X, y, f_eval):
    costs = []
    for tree in sample:
        cost = evaluate(tree, X, y, f_eval)
        costs.append(cost)
    return costs
