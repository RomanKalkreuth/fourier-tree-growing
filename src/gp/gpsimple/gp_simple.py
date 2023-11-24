# Copyright (C) 2023 -
# Roman Kalkreuth (Roman.Kalkreuth@lip6.fr)
# Computer Lab of Paris 6, Sorbonne Université (Paris, France)

import random
import numpy as np

from typing import Callable

from gp_problem import GPProblem
from gp_config import GPConfig, Hyperparameters
import gp_init

__author__ = 'Roman Kalkreuth'
__copyright__ = 'Copyright (C) 2023, Roman Kalkreuth'
__version__ = '1.0'
__email__ = 'Roman.Kalkreuth@lip6.fr'

random.seed()
np.random.seed()


class GPSimple:
    config: GPConfig
    hyperparameters: Hyperparameters
    problem: GPProblem
    algorithm: Callable

    @staticmethod
    def init(problem, algorithm):
        GPSimple.config = gp_init.init_config()
        GPSimple.problem = problem
        GPSimple.algorithm = algorithm

        if algorithm is not None:
            GPSimple.hyperparameters = gp_init.init_hyperparameters(algorithm)


    @staticmethod
    def run():
        return GPSimple.evolve(algorithm=GPSimple.algorithm,
                               problem=GPSimple.problem,
                               hyperparameters=GPSimple.hyperparameters,
                               num_jobs=GPSimple.config.num_jobs,
                               minimalistic_output=GPSimple.config.minimalistic_output)

    @staticmethod
    def evolve(algorithm, hyperparameters, problem, num_jobs=1,
               silent_evolver=False, minimalistic_output=False, config=None):

        result = []

        if config is not None:
            silent_algorithm = config.silent_algorithm
            silent_evolver = config.silent_evolver
            num_jobs = config.num_jobs
            max_generations = config.max_generations
            minimalistic_output = config.minimalistic_output
            stopping_criteria = config.stopping_criteria
            minimizing_fitness = config.minimizing_fitness

        for job in range(num_jobs):
            best_fitness, num_evaluations = algorithm(max_generations=max_generations,
                                                      problem=problem,
                                                      silent=silent_algorithm,
                                                      minimalistic_output=minimalistic_output,
                                                      minimizing_fitness=minimizing_fitness,
                                                      stopping_criteria=stopping_criteria,
                                                      hyperparameters=hyperparameters)
            result.append((best_fitness, num_evaluations))
            if not silent_evolver:
                if not minimalistic_output:
                    print("Job #" + str(job) + " - Evaluations: " + str(num_evaluations) +
                      " - Best Fitness: " + str(best_fitness))
                else:
                    print(str(num_evaluations) + ";" + str(best_fitness))
        return result

