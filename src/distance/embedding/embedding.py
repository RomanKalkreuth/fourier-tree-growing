import sys
sys.path.insert(0, '../../gp/gpsimple')

from gp_simple import GPSimple
from gp_tree import GPNode
import gp_problem as problem
import gp_config as config

import src.conversion.tree_conversion as convert
import src.benchmark.symbolic_regression.dataset_generator as generator
import src.benchmark.symbolic_regression.benchmark_functions as benchmarks

def embed_tree_list(tree_list):
    vocab = set(tree_list)
    symbol_to_ix = {word: i for i, word in enumerate(vocab)}
    print(symbol_to_ix)


def init():
    MIN_INIT_TREE_DEPTH = 2
    MAX_INIT_TREE_DEPTH = 6

    objective_function = benchmarks.koza1
    X_train = generator.random_samples_float(-1.0, 1.0, 20, dim=1)
    y_train = generator.generate_function_values(objective_function, X_train)
    regression_problem = problem.RegressionProblem(X_train, y_train)

    GPSimple.init(problem=regression_problem, algorithm=None)

    tree = GPNode()
    tree.init_tree(min_depth=MIN_INIT_TREE_DEPTH, max_depth=MAX_INIT_TREE_DEPTH)

    tree_list = convert.tree_to_list(tree)
    convert.symbols_to_string(tree_list, config.FUNCTIONS)

    embed_tree_list(tree_list)

init()