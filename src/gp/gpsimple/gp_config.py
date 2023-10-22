import gp_functions as gpf
import gp_util as util

FUNCTIONS = [gpf.add, gpf.sub, gpf.mul, gpf.div]
TERMINALS = ['x', 0, 1]
VARIABLES = util.get_variables_from_terminals(TERMINALS)

MU = 1
LAMBDA = 1

MINIMIZING_FITNESS = True
IDEAL_FITNESS = 0.01
FITNESS_CALCULATION_METHOD = "abs"

MIN_DEPTH = 2
MAX_DEPTH = 3
SUBTREE_DEPTH = 2

MUTATION_RATE = 0.1

NUM_FUNCTIONS = len(FUNCTIONS)
NUM_TERMINALS = len(TERMINALS)

NUM_GENERATIONS = 1000

