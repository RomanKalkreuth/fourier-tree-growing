from gp_problem import RegressionProblem
import gp_algorithm
from gp_simple import GPSimple
from gp_tree import GPNode

import gp_fitness as fitness

import src.benchmark.symbolic_regression.dataset_generator as generator
import src.benchmark.symbolic_regression.benchmark_functions as functions

import src.distance.distance as distance

objective_function = functions.nguyen9
X_train = generator.random_samples_float(-1.0, 1.0, 20, dim=2)
y_train = generator.generate_function_values(objective_function, X_train)

regression_problem = RegressionProblem(X_train, y_train)
ea = gp_algorithm.one_plus_lambda_ea

actual = regression_problem.y_train

GPSimple.init(regression_problem, ea)

t1 = GPNode()
t1.init_tree(2, 6)

t1.print_tree()

#t2 = GPNode()
#t2.init_tree(2, 3)

t2 = t1.clone()
t2.mutate(mutation_rate=0.2, subtree_depth=2)

t2.print_tree()

std,syd = distance.decomposed_distance(t1,t2)
print(std)
print(syd)

p1 = regression_problem.evaluate(t1)
f1 = fitness.calculate_fitness(actual, p1, metric="abs")

p2 = regression_problem.evaluate(t2)
f2 = fitness.calculate_fitness(actual, p2, metric="abs")

print()

print(f1)
print(f2)
