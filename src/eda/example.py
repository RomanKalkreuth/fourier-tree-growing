import sys

sys.path.insert(0, '../gp/gpsimple')

import copy

from gp_simple import GPSimple
from gp_tree import GPNode
import gp_config as config
import gp_problem as problem
import gp_fitness as fitness
import gp_util as util

import src.conversion.tree_conversion as convert

import src.benchmark.symbolic_regression.dataset_generator as generator
import src.benchmark.symbolic_regression.benchmark_functions as benchmarks


# Define constants for the minimum and maximum tree depth
MIN_INIT_TREE_DEPTH = 2
MAX_INIT_TREE_DEPTH = 4

# The function set is stored in the gp_config.py file. It contains the functions which
# are called during evaluation
print("Function set:")
print(config.FUNCTIONS)
print()

# For the conversion back to a dynamic binary tree structure,
# we need the functions names to distinguish between functions and terminals
functions = []
for f in config.FUNCTIONS:
    functions.append(f.__name__)

print("Function names:")
print(functions)
print()

# Besides the function nodes, a GP parse tree has terminals which represent the leaves of the tree.
# The terminals are stored in the terminal set
print("Terminal set:")
print(config.TERMINALS)
print()

# Choose a simple symbolic regression benchmark
objective_function = benchmarks.nguyen9

# Generate the training dataset
X_train = generator.random_samples_float(-1.0, 1.0, 20, dim=2)
y_train = generator.generate_function_values(objective_function, X_train)

# Set up the GP regression problem with the training set
regression_problem = problem.RegressionProblem(X_train, y_train)

# Init the GPSimple system with our regression problem but without an algorithm
GPSimple.init(problem=regression_problem, algorithm=None)

# Instantiate and initialize a random GP parse tree from the function and terminal set
tree = GPNode()
tree.init_tree(min_depth=MIN_INIT_TREE_DEPTH, max_depth=MAX_INIT_TREE_DEPTH)

# Predict the function values with our randomly generate tree
prediction = regression_problem.evaluate(tree)

# Store the real function values
actual = regression_problem.y_train


# Print the tree vertically
print("Random parse tree:")
tree.print_tree()
print()


# Calculate the fitness which is defined by the absolute distance between predicted and
# actual values
fitness_val = fitness.calculate_fitness(actual, prediction, metric="abs")

print("Fitness value (distance to optimum):")
print(fitness_val)

print()
# Convert the tree to a list
print("List format:")
tree_list = convert.tree_to_list(tree)

# Print the tree as list
print(tree_list)

# Convert the list where symbols are strings
tree_list_str = copy.copy(tree_list)
convert.symbols_to_string(tree_list_str, config.FUNCTIONS)

print(tree_list_str)

convert.symbols_to_type(tree_list_str, config.FUNCTIONS, config.TERMINALS)
# tree_list = tree_list_str
# print(tree_list)

print()

structure_validation = convert.validate_structure(tree_list, config.FUNCTIONS)
print("Structure validation: " + str(structure_validation))

tree = convert.list_to_tree(tree_list, config.FUNCTIONS)

print()
print("Re-converted tree from list format:")
tree.print_tree()

print()

err = util.validate_tree(tree, config.FUNCTIONS, config.TERMINALS)
print("Error of the tree: " + str(err))
