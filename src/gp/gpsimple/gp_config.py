# Copyright (C) 2023 -
# Roman Kalkreuth (Roman.Kalkreuth@lip6.fr)
# Computer Lab of Paris 6, Sorbonne Université (Paris, France)

from dataclasses import dataclass

import gp_functions as gpf
import gp_util as util


__author__ = 'Roman Kalkreuth'
__copyright__ = 'Copyright (C) 2023, Roman Kalkreuth'
__version__ = '1.0'
__email__  = 'Roman.Kalkreuth@lip6.fr'

FUNCTIONS = [gpf.add, gpf.sub, gpf.mul, gpf.div]
TERMINALS = ['x']
VARIABLES = util.get_variables_from_terminals(TERMINALS)

MU = 1
LAMBDA = 1

MINIMIZING_FITNESS = True
STOPPING_CRITERIA = 0.01
FITNESS_METRIC = "abs"

MIN_INIT_TREE_DEPTH = 2
MAX_INIT_TREE_DEPTH = 3
SUBTREE_DEPTH = 6

MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.7

TOURNAMENT_SIZE = 2

NUM_ELITES = 2

POPULATION_SIZE = 100

MAX_GENERATIONS = 10000
NUM_JOBS = 100

NUM_FUNCTIONS = len(FUNCTIONS)
NUM_TERMINALS = len(TERMINALS)
NUM_VARIABLES = len(VARIABLES)

@dataclass
class GPConfig:
    num_jobs: int
    max_generations: int

    mu: int
    lmbda: int

    population_size: int
    num_elites: int

    mutation_rate: float
    crossover_rate: float

    tournament_size: int

    min_init_tree_depth: int
    max_init_tree_depth: int
    subtree_depth: int

    functions: list
    terminals: list
    variables: list

    num_functions: int
    num_terminals: int
    num_variables: int










