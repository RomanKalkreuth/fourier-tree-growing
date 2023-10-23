# Copyright (C) 2023 -
# Roman Kalkreuth (Roman.Kalkreuth@lip6.fr)
# Computer Lab of Paris 6, Sorbonne Université (Paris, France)

from random import seed
from gp_tree import GPNode
import gp_config as config
import gp_util as util
import gp_problem as problem
import gp_algorithm as algorithm
import src.benchmark.dataset_generator as generator
import src.benchmark.functions as functions
import src.distance.distance as distance

__author__ = 'Roman Kalkreuth'
__copyright__ = 'Copyright (C) 2023, Roman Kalkreuth'
__version__ = '1.0'
__email__  = 'Roman.Kalkreuth@lip6.fr'

seed()

function = functions.koza1

X_train = generator.random_samples_float(-1.0, 1.0, 20)
y_train = generator.generate_function_values(function, X_train)

regression_problem = problem.RegressionProblem(X_train, y_train)

algorithm.one_plus_lambda(num_generations=config.MAX_GENERATIONS, lmbda=config.LAMBDA,
                          ideal_fitness=config.IDEAL_FITNESS, problem=regression_problem)



#tree1 = GPNode()
#tree1.init(config.MIN_DEPTH, config.MAX_DEPTH)
#expression1 = util.generate_expression(tree1)
#tree2 = GPNode()
#tree2.init(config.MIN_DEPTH, config.MAX_DEPTH)
#expression2 = util.generate_expression(tree2)
#distance = distance.levenshtein_distance(tree1,tree2)
#print(expression1)
#print(expression2)
#print(distance)




