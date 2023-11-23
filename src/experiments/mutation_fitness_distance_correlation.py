import sys

sys.path.insert(0, '../gp/gpsimple')
sys.path.insert(0, '../distance')
sys.path.insert(0, '../eda')

import random
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
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
MAX_INIT_TREE_DEPTH = 6
SUBTREE_DEPTH = 4
MAX_NODE_MUTATIONS = 32
MUTATION_RATE = 0.05
NUM_SAMPLES = 1000
STEP = 0.01
N = int(1.0 / STEP) + 1

# Choose a simple symbolic regression benchmark
benchmark = benchmarks.nguyen8

# Generate the training dataset
X_train = generator.random_samples_float(0, 4, 20)
y_train = generator.generate_function_values(benchmark, X_train)

regression_problem = problem.RegressionProblem(X_train, y_train)
actual = regression_problem.y_train

pd = []
fd = []

t1 = GPNode()
t1.init_tree(min_depth=MIN_INIT_TREE_DEPTH, max_depth=MAX_INIT_TREE_DEPTH)

p1 = regression_problem.evaluate(t1)
f1 = fitness.calculate_fitness(actual, p1, metric="abs")

for i in range(NUM_SAMPLES):
    t2 = t1.clone()
    t2.mutate(mutation_rate=MUTATION_RATE, subtree_depth=SUBTREE_DEPTH)

    # n = random.randint(1, MAX_NODE_MUTATIONS)
    # mutation.node_mutation(t2, n)

    p2 = regression_problem.evaluate(t2)
    f2 = fitness.calculate_fitness(actual, p2, metric="abs")

    # ld = distance.levenshtein_distance(t1, t2)
    ted = distance.tree_edit_distance(t1, t2)

    if ted > 0:
        fd.append(int(abs(f1 - f2)))
        pd.append(ted)

pd_arr = np.array(pd)
fd_arr = np.array(fd)

# print(pd_arr)
# cv = np.cov(pd_arr, fd_arr)
cr = np.corrcoef(pd_arr, fd_arr)
print(cr)

plt.scatter(pd_arr, fd_arr)
plt.ylim(0, 100)
plt.xlabel("Distance")
plt.ylabel("Fitness")
plt.title("Nguyen-7")
plt.show()

#plot = px.scatter(x=pd_arr, y=fd_arr)
#plot.update_layout(yaxis_range=[0,1000])
#plot.show()

# plt.hist(pd_arr, bins=15)
# plt.xlim(0, 1000)
# plt.show()

# plt.hist2d(fd_arr, pd_arr, bins=(300, 300), cmap=plt.cm.jet)
# plt.xlim(0, 1000)
# plt.show()

# fig = px.density_heatmap(x=fd_arr, y=pd_arr, marginal_x="histogram", marginal_y="histogram")
# fig.show()
