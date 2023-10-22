from gp_tree import GPNode
import gp_config as config
import gp_util as util
import gp_problem as problem
from random import seed
import src.benchmark.dataset_generator as generator
import src.benchmark.functions as functions
import src.distance.distance as distance

seed()

function = functions.koza1

data_points = generator.random_samples_float(-1.0, 1.0, 20)
function_values = generator.generate_function_values(function, data_points)
training_set = generator.concatenate_data(data_points, function_values)

problem = problem.RegressionProblem(training_set)

tree1 = GPNode()
tree1.init(config.MIN_DEPTH, config.MAX_DEPTH)
expression1 = util.generate_expression(tree1)

tree2 = GPNode()
tree2.init(config.MIN_DEPTH, config.MAX_DEPTH)
expression2 = util.generate_expression(tree2)

distance = distance.levenshtein_distance(tree1,tree2)

print(expression1)
print(expression2)
print(distance)

#tree.mutate()
#tree.print_tree()
#eval = tree.evaluate(training_set[0])



