import gp_functions as gpf


FUNCTIONS = [gpf.add, gpf.sub, gpf.mul, gpf.div]
TERMINALS = ['x', 0, 1]
VARIABLES = ['x']

MIN_DEPTH = 2
MAX_DEPTH = 6
SUBTREE_DEPTH = 2

MUTATION_RATE = 0.1

NUM_FUNCTIONS = len(FUNCTIONS)
NUM_TERMINALS = len(TERMINALS)

NUM_GENERATIONS = 1000
