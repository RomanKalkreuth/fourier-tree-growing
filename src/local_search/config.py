from functions import Mathematical
import util

FUNCTIONS = [Mathematical.add, Mathematical.sub, Mathematical.mul, Mathematical.div, Mathematical.sin,
             Mathematical.cos, Mathematical.log, Mathematical.sqrt]
TERMINALS = ['x', 1.0]
VARIABLES = util.get_variables_from_terminals(TERMINALS)

FUNCTION_CLASS = Mathematical

MIN_INIT_TREE_DEPTH = 2
MAX_INIT_TREE_DEPTH = 4

NUM_FUNCTIONS = len(FUNCTIONS)
NUM_TERMINALS = len(TERMINALS)
NUM_VARIABLES = len(VARIABLES)

