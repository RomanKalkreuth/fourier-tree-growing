# Copyright (C) 2023 -
# Roman Kalkreuth (Roman.Kalkreuth@lip6.fr)
# Computer Lab of Paris 6, Sorbonne Université (Paris, France)

from dataclasses import dataclass

from gp_functions import Functions, Mathematical
import gp_util as util

__author__ = 'Roman Kalkreuth'
__copyright__ = 'Copyright (C) 2023, Roman Kalkreuth'
__version__ = '1.0'
__email__ = 'Roman.Kalkreuth@lip6.fr'


FUNCTIONS = [Mathematical.add, Mathematical.sub, Mathematical.mul, Mathematical.div]
TERMINALS = ['x', 'y']
VARIABLES = util.get_variables_from_terminals(TERMINALS)

FUNCTION_CLASS = Mathematical

MIN_INIT_TREE_DEPTH = 2
MAX_INIT_TREE_DEPTH = 6
SUBTREE_DEPTH = 6

MAX_GENERATIONS = 2000
NUM_JOBS = 100
SILENT = True
MINIMALISTIC_OUTPUT = False

MINIMIZING_FITNESS = True
STOPPING_CRITERIA = 0.01
FITNESS_METRIC = "abs"

MU = 1
LAMBDA = 1

POPULATION_SIZE = 1000
NUM_ELITES = 1

MUTATION_RATE = 0.01
CROSSOVER_RATE = 0.9

TOURNAMENT_SIZE = 10

NUM_FUNCTIONS = len(FUNCTIONS)
NUM_TERMINALS = len(TERMINALS)
NUM_VARIABLES = len(VARIABLES)


@dataclass
class GPConfig:
    num_jobs: int
    max_generations: int

    functions: list
    terminals: list
    variables: list

    function_class: Functions

    num_functions: int
    num_terminals: int
    num_variables: int

    fitness_metric: str
    stopping_criteria: float
    minimizing_fitness: bool

    silent: bool
    minimalistic_output:bool

    #def validate(self):
    # TODO: Implement validation routine

@dataclass
class Hyperparameters:

    tree_init_depth = tuple
    subtree_depth: int

    mu: int
    lmbda: int

    population_size: int
    num_elites: int

    tournament_size: int

    crossover_rate: float
    mutation_rate: float

    #def validate(self):
    # TODO: Implement validation routine



