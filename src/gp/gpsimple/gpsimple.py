# Copyright (C) 2023 -
# Roman Kalkreuth (Roman.Kalkreuth@lip6.fr)
# Computer Lab of Paris 6, Sorbonne Université (Paris, France)

from random import seed

from apted import APTED, Config

from gp_tree import GPNode
import gp_config as config
import gp_problem as problem
import gp_util as util
import src.benchmark.symbolic_regression.dataset_generator as generator
import src.benchmark.symbolic_regression.functions as functions
import src.distance.distance as distance

__author__ = 'Roman Kalkreuth'
__copyright__ = 'Copyright (C) 2023, Roman Kalkreuth'
__version__ = '1.0'
__email__ = 'Roman.Kalkreuth@lip6.fr'

seed()

function = functions.koza1

X_train = generator.random_samples_float(-1.0, 1.0, 20)
y_train = generator.generate_function_values(function, X_train)

regression_problem = problem.RegressionProblem(X_train, y_train)

# algorithm.one_plus_lambda(num_generations=config.MAX_GENERATIONS, lmbda=config.LAMBDA,
#                          ideal_fitness=config.IDEAL_FITNESS, problem=regression_problem)


tree1 = GPNode()
tree1.init(config.MIN_TREE_DEPTH, config.MAX_TREE_DEPTH)

tree2 = GPNode()
tree2.init(config.MIN_TREE_DEPTH, config.MAX_TREE_DEPTH)

node_list, adj_list = util.transform_list_format(tree1)
tree1.print_tree()
print()
print(adj_list)
print(node_list)
expression1 = util.generate_bracket_notation(tree1)
expression2 = util.generate_bracket_notation(tree2)
