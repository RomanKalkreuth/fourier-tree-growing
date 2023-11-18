import sys
sys.path.insert(0, '../gp/gpsimple')

import copy

from gp_tree import GPNode
import gp_config as config
import gp_problem as problem
import gp_fitness as fitness
import tree_convert as convert

import src.benchmark.symbolic_regression.dataset_generator as generator
import src.benchmark.symbolic_regression.functions as benchmarks

# Define constants for the minimum and maximum tree depth
MIN_INIT_TREE_DEPTH = 2
MAX_INIT_TREE_DEPTH = 3

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


# Instantiate and initialize a random GP parse tree from the function and terminal set
tree = GPNode()
tree.init(min_depth=MIN_INIT_TREE_DEPTH, max_depth=MAX_INIT_TREE_DEPTH)


# Print the tree vertically
print("Random parse tree:")
tree.print_tree()
print()

# Choose a simple symbolic regression benchmark
benchmark = benchmarks.koza1

# Generate the training dataset
X_train = generator.random_samples_float(-1.0, 1.0, 20)
y_train = generator.generate_function_values(benchmark, X_train)

# Set up the GP regression problem with the training set
regression_problem = problem.RegressionProblem(X_train, y_train)

# Predict the function values with our randomly generate tree
prediction = regression_problem.evaluate(tree)

# Store the real function values
actual = regression_problem.y_train

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
#tree_list = tree_list_str
#print(tree_list)

print()

structure_validation = convert.validate_structure(tree_list, config.FUNCTIONS)
print("Structure validation: " + str(structure_validation))

tree = convert.list_to_tree(tree_list, config.FUNCTIONS)

print()
print("Re-converted tree from list format:")
tree.print_tree()

