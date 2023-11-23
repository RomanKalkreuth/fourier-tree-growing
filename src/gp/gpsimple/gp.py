from gp_problem import RegressionProblem
import gp_algorithm
from gp_simple import GPSimple

import src.benchmark.symbolic_regression.dataset_generator as generator
import src.benchmark.symbolic_regression.benchmark_functions as functions

objective_function = functions.nguyen9
X_train = generator.random_samples_float(-1.0, 1.0, 20, dim=2)
y_train = generator.generate_function_values(objective_function, X_train)

regression_problem = RegressionProblem(X_train, y_train)
ea = gp_algorithm.one_plus_lambda_ea

GPSimple.init(regression_problem, ea)
GPSimple.run()

