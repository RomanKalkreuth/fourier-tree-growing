from parse_tree import ParseTree
import variation
import config
import util
import evaluation

import src.benchmark.symbolic_regression.dataset_generator as generator
import src.benchmark.symbolic_regression.benchmark_functions as functions

def hill_climbing(n_iter, X, y, f_eval, ideal, minimizing=True, strict=True):
    best = ParseTree
    best.init_tree(min_depth=config.MIN_INIT_TREE_DEPTH, max_depth=config.MAX_INIT_TREE_DEPTH)

    best_cost = evaluate(best, X, y, f_eval)

    for i in range(0, n_iter):
        candidate = variate(best)
        cost = evaluate(candidate, X, y, f_eval)

        if is_better(cost, best_cost, minimizing=minimizing, strict=strict):
            best = candidate

        if is_ideal(best_cost, ideal, minimizing=minimizing):
            break


def variate(tree):
    variation.single_node_variation(tree)
    return tree


def evaluate(tree, X, y, f_eval):
    cost = 0.0
    for i, x in enumerate(X):
        y_pred = tree.evaluate(x)
        cost += f_eval(y_pred, y[i])
    return cost


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

def run():

    objective_function = functions.koza1
    X = generator.random_samples_float(-1.0, 1.0, 20, dim=1)
    y = generator.generate_function_values(objective_function, X)

    tree = ParseTree()
    tree.init_tree(min_depth=config.MIN_INIT_TREE_DEPTH, max_depth=config.MAX_INIT_TREE_DEPTH)

    v = evaluate(tree, X, y, evaluation.absolute_error)
    print(v)

    variation.single_node_variation(tree)

    v = evaluate(tree, X, y, evaluation.absolute_error)
    print(v)



run()
