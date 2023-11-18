# Copyright (C) 2023 -
# Roman Kalkreuth (Roman.Kalkreuth@lip6.fr)
# Computer Lab of Paris 6, Sorbonne Université (Paris, France)

import random
import numpy as np

from gp_tree import GPNode
import gp_config as config
import gp_problem as problem
import gp_util as util
import gp_stats as stats
import gp_init as init
import gp_algorithm as algorithm
import src.benchmark.symbolic_regression.dataset_generator as generator
import src.benchmark.symbolic_regression.functions as functions
import src.distance.distance as distance


__author__ = 'Roman Kalkreuth'
__copyright__ = 'Copyright (C) 2023, Roman Kalkreuth'
__version__ = '1.0'
__email__ = 'Roman.Kalkreuth@lip6.fr'

random.seed()
np.random.seed()

function = functions.koza2

X_train = generator.random_samples_float(-1.0, 1.0, 20)
y_train = generator.generate_function_values(function, X_train)

regression_problem = problem.RegressionProblem(X_train, y_train)


# algorithm.one_plus_lambda(num_generations=config.MAX_GENERATIONS, lmbda=config.LAMBDA,
#                          ideal_fitness=config.IDEAL_FITNESS, problem=regression_problem)

#ea = algorithm.canonical_ea(max_generations=10000, population_size=100, mutation_rate=0.05,
#                            crossover_rate=0.9, tournament_size=2, stopping_criteria=0.01,
#                            problem=regression_problem, num_elites=2)

parameters = init.init_parameters(algorithm.canonical_ea)

#ea = algorithm.canonical_ea(problem=regression_problem, parameters=parameters)

results = algorithm.evolve(algorithm=algorithm.canonical_ea,
                 problem=regression_problem,
                 parameters=parameters,
                 num_jobs=100,
                 minimalistic_output=True)

stats.write_results_to_csv(results)