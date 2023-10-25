# Copyright (C) 2023 -
# Roman Kalkreuth (Roman.Kalkreuth@lip6.fr)
# Computer Lab of Paris 6, Sorbonne Université (Paris, France)

import gp_functions as gpf
import gp_util as util

__author__ = 'Roman Kalkreuth'
__copyright__ = 'Copyright (C) 2023, Roman Kalkreuth'
__version__ = '1.0'
__email__  = 'Roman.Kalkreuth@lip6.fr'

FUNCTIONS = [gpf.add, gpf.sub, gpf.mul, gpf.div]
TERMINALS = ['x', 0, 1]
VARIABLES = util.get_variables_from_terminals(TERMINALS)

MU = 1
LAMBDA = 1

MINIMIZING_FITNESS = True
IDEAL_FITNESS = 0.01
FITNESS_METRIC = "abs"

MIN_TREE_DEPTH = 2
MAX_TREE_DEPTH = 3
SUBTREE_DEPTH = 1

MUTATION_RATE = 0.1
MAX_GENERATIONS = 10000

NUM_FUNCTIONS = len(FUNCTIONS)
NUM_TERMINALS = len(TERMINALS)
NUM_VARIABLES = len(VARIABLES)


