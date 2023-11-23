# Copyright (C) 2023 -
# Roman Kalkreuth (Roman.Kalkreuth@lip6.fr)
# Computer Lab of Paris 6, Sorbonne Université (Paris, France)

import random
import numpy as np

from dataclasses import dataclass
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
           silent=False, minimalistic_output=False, config=None):

        result = []

        if config is not None:
            silent = config.silent
            num_jobs = config.num_jobs
            minimalistic_output = config.minimalistic_output

        for job in range(num_jobs):
            best_fitness, num_evaluations = algorithm(hyperparameters=hyperparameters, problem=problem)
            result.append((best_fitness, num_evaluations))
            if not silent:
                if not minimalistic_output:
                    print("Job #" + str(job) + " - Evaluations: " + str(num_evaluations) +
                      " - Best Fitness: " + str(best_fitness))
                else:
                    print(str(num_evaluations) + ";" + str(best_fitness))
        return result

