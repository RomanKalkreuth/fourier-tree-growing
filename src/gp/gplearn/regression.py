from gplearn.genetic import SymbolicRegressor

import sys
import random

sys.path.insert(0, '../../benchmark/symbolic_regression')

import src.benchmark.symbolic_regression.dataset_generator as generator
import src.benchmark.symbolic_regression.benchmark_functions as functions
import src.evaluation.evaluation as evaluation
import src.algorithm.algorithm as algorithm


def gplearn_regressor(X_train, y_train, population_size=1000,
                      generations=100, stopping_criteria=0.01,
                      p_crossover=0.9, p_subtree_mutation=0.1,
                      p_hoist_mutation=0.00, p_point_mutation=0.0,
                      max_samples=0.9, verbose=1,
                      parsimony_coefficient=0.01, random_state=42):

    est_gp = SymbolicRegressor(population_size=population_size,
                               generations=generations, stopping_criteria=stopping_criteria,
                               p_crossover=p_crossover, p_subtree_mutation=p_subtree_mutation,
                               p_hoist_mutation=p_hoist_mutation, p_point_mutation=p_point_mutation,
                               max_samples=max_samples, verbose=verbose,
                               parsimony_coefficient=parsimony_coefficient, random_state=random_state)

    est_gp.fit(X_train, y_train)

DEGREE = 6

objective_function = functions.polynomial
X = generator.random_samples_float(-1.0, 1.0, 20, dim=1)
y = generator.generate_polynomial_values(X, degree=DEGREE)
f_eval = evaluation.absolute_error

random.seed()

gplearn_regressor(X,y)




