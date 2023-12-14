from parse_tree import ParseTree
import variation
import config
import util
import evaluation
import sampling
from transition_matrix import TransitionMatrix

import matplotlib.pyplot as plt
import numpy as np

import src.benchmark.symbolic_regression.dataset_generator as generator
import src.benchmark.symbolic_regression.benchmark_functions as functions


def hill_climbing(n_iter, X, y, f_eval, ideal, minimizing=True, strict=True):
    best = ParseTree
    best.init_tree(min_depth=config.MIN_INIT_TREE_DEPTH, max_depth=config.MAX_INIT_TREE_DEPTH)

    best_cost = evaluation.evaluate(best, X, y, f_eval)

    for i in range(0, n_iter):
        candidate = variate(best)
        cost = evaluation.evaluate(candidate, X, y, f_eval)

        if evaluation.is_better(cost, best_cost, minimizing=minimizing, strict=strict):
            best = candidate

        if evaluation.is_ideal(best_cost, ideal, minimizing=minimizing):
            break


def variate(tree):
    variation.single_node_variation(tree)
    return tree


def run():
    MAX_DEPTH = 8
    MIN_DEPTH = 2
    N = 10000

    objective_function = functions.koza1
    X = generator.random_samples_float(-1.0, 1.0, 20, dim=1)
    y = generator.generate_function_values(objective_function, X)

    matrix = TransitionMatrix(config.NUM_FUNCTIONS)

    d = 0
    for i in range(0, N):
        tree = ParseTree()
        tree.init_tree(min_depth=config.MIN_INIT_TREE_DEPTH, max_depth=config.MAX_INIT_TREE_DEPTH)

        v1 = evaluation.evaluate(tree, X, y, evaluation.absolute_error)
        transition = variation.single_node_variation(tree)
        v2 = evaluation.evaluate(tree, X, y, evaluation.absolute_error)

        d = round(v2 - v1, 2)

        if transition[0] in config.FUNCTIONS:
            matrix.update(transition[0], transition[1], d)

    fig, ax = plt.subplots()
    im = ax.imshow(matrix.matrix)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Color bar", rotation=-90, va="bottom")

    print(matrix)

    plt.show()

    #sample = sampling.sample_uniform(MIN_DEPTH,MAX_DEPTH, N)
    #costs = evaluation.evaluate_sample(sample, X, y, evaluation.absolute_error)
    #depths = [tree.depth() for tree in sample]

    #dist = {}

    #for i in range(0, 0):
    #    depth = depths[i]

    #    if depth in dist:
    #        dist[depth] += costs[i]
    #    else:
    #        dist[depth] = costs[i]

    #plt.bar(dist.keys(), dist.values())
    #plt.xlabel("Tree depth")
    #plt.ylabel("Cumulative Cost")
    #plt.yscale("log")
    #plt.show()

run()
