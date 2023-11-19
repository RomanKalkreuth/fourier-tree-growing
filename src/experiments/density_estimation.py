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
import src.benchmark.symbolic_regression.functions as benchmarks

random.seed()
np.random.seed()

MIN_INIT_TREE_DEPTH = 2
MAX_INIT_TREE_DEPTH = 4
SUBTREE_DEPTH = 4
MUTATION_RATE = 0.1
NUM_SAMPLES = 100000

# Choose a simple symbolic regression benchmark
benchmark = benchmarks.koza2

# Generate the training dataset
X_train = generator.random_samples_float(-1.0, 1.0, 20)
y_train = generator.generate_function_values(benchmark, X_train)

regression_problem = problem.RegressionProblem(X_train, y_train)
actual = regression_problem.y_train

fitnesses = np.empty(NUM_SAMPLES)
tree = GPNode()

for i in range(NUM_SAMPLES):
    tree.init(min_depth=MIN_INIT_TREE_DEPTH, max_depth=MAX_INIT_TREE_DEPTH)

    pred = regression_problem.evaluate(tree)
    fit_val = fitness.calculate_fitness(actual, pred, metric="abs")

    fitnesses[i] = fit_val

print(fitnesses)

#plot = px.histogram(x=fitnesses)
#plot.show()

#plt.hist(fitnesses, bins="auto")
#plt.xlim(0, 1000)
#plt.show()

X = fitnesses.reshape(-1, 1)
kde = KernelDensity(kernel='gaussian', bandwidth=3).fit(X)
s = np.linspace(0,10000)
e = kde.score_samples(s.reshape(-1,1))
plt.plot(s, e)
plt.show()

#kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(fitnesses)
#sns.kdeplot(data=fitnesses, shade=True, bw=0.5)
#plt.xlim(0, 100000)
#plt.show()

