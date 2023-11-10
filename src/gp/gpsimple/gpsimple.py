# Copyright (C) 2023 -
# Roman Kalkreuth (Roman.Kalkreuth@lip6.fr)
# Computer Lab of Paris 6, Sorbonne Université (Paris, France)

from random import seed

from apted import APTED, Config

from gp_tree import GPNode
import gp_config as config
import gp_problem as problem
import gp_util as util
import gp_algorithm as algorithm
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

#ea = algorithm.canonical_ea(max_generations=10000, population_size=100, mutation_rate=0.05,
#                            crossover_rate=0.9, tournament_size=2, stopping_criteria=0.01,
#                            problem=regression_problem, num_elites=2)

#algorithm.evolve(ea)

tree1 = GPNode()
tree1.init(config.MIN_INIT_TREE_DEPTH, config.MAX_INIT_TREE_DEPTH)

#tree2 = GPNode()
#tree2.init(config.MIN_INIT_TREE_DEPTH, config.MAX_INIT_TREE_DEPTH)

tree1.print_tree()
print()

size = tree1.size()
depth = tree1.depth()
print(size)
print(depth)

node_list, adj_list = util.convert_list_format(tree1)

print(node_list)
print(adj_list)

#tree2.print_tree()

#ted = distance.tree_edit_distance(tree1, tree2)
#print(ted)
