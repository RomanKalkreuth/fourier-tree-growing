import sys

sys.path.insert(0, '../gp/gpsimple')
sys.path.insert(0, '../distance')
sys.path.insert(0, '../eda')

import random
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity
import seaborn as sns

import distance as distance
from gp_tree import GPNode
import gp_problem as problem
import gp_fitness as fitness
import gp_mutation as mutation

import src.benchmark.symbolic_regression.dataset_generator as generator
import src.benchmark.symbolic_regression.benchmark_functions as benchmarks

random.seed()
np.random.seed()

MIN_INIT_TREE_DEPTH = 2
MAX_INIT_TREE_DEPTH = 7
SUBTREE_DEPTH = 4
MUTATION_RATE = 0.1
NUM_SAMPLES = 20000

# Choose a simple symbolic regression benchmark
benchmark = benchmarks.nguyen8

# Generate the training dataset
X_train = generator.random_samples_float(0, 4, 20)
y_train = generator.generate_function_values(benchmark, X_train)

regression_problem = problem.RegressionProblem(X_train, y_train)
actual = regression_problem.y_train

fitnesses = np.empty(NUM_SAMPLES)
tree = GPNode()

for i in range(NUM_SAMPLES):
    tree.init_tree(min_depth=MIN_INIT_TREE_DEPTH, max_depth=MAX_INIT_TREE_DEPTH)

    pred = regression_problem.evaluate(tree)
    fit_val = fitness.calculate_fitness(actual, pred, metric="abs")

    fitnesses[i] = fit_val

print(fitnesses)

X = fitnesses[:, np.newaxis]
X_plot = np.linspace(0, 100, 1000)[:, np.newaxis]
kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(X)
log_dens = kde.score_samples(X_plot)
plt.plot(
    X_plot[:, 0],
    np.exp(log_dens)
)
plt.xlabel("Objective function value")
plt.ylabel("Density")
plt.title("Nguyen-8")
plt.show()
