import sys
import random
from operator import itemgetter

import numpy as np

sys.path.insert(0, '../representation')
sys.path.insert(0, '../analysis')
sys.path.insert(0,'../benchmark/symbolic_regression')

from src.representation.parse_tree import ParseTree
import src.evaluation.evaluation as evaluation
import src.analysis.analysis as analysis

import src.benchmark.symbolic_regression.dataset_generator as generator
import src.benchmark.symbolic_regression.benchmark_functions as functions

random.seed()

MIN_INIT_TREE_DEPTH = 2
MAX_INIT_TREE_DEPTH = 6
ITERATIONS = 1000
MUTATION_RATE = 0.1
LAMBDA = 2
INSTANCES = 1
MAX_DEG = 10

objective_function = functions.nguyen4
X = generator.random_samples_float(-1.0, 1.0, 20, dim=1)
y = generator.generate_polynomial_values(X, 8)
f_eval = evaluation.absolute_error

sequences = []
candidates = []

counts = np.zeros(INSTANCES)
depths = np.zeros(LAMBDA)


for instance in range(0, INSTANCES):
    parent = ParseTree()
    parent.init_tree(min_depth=MIN_INIT_TREE_DEPTH, max_depth=MAX_INIT_TREE_DEPTH)
    best = evaluation.evaluate(parent, X, y, f_eval)
    seq1 = [float(parent.evaluate(x)) for x in X]
    deg1 = analysis.polynomial_degree_fit(X.flatten(), seq1, MAX_DEG)

    count = 0
    for iter in range(0, ITERATIONS):

        sequences.clear()
        candidates.clear()

        sequences.append(seq1[:])

        for i in range(0, LAMBDA):
            candidate = parent.clone()
            candidate.variate(mutation_rate=MUTATION_RATE)

            cost = evaluation.evaluate(candidate, X, y, f_eval)
            seq = [float(candidate.evaluate(x)) for x in X]
            deg = analysis.polynomial_degree_fit(X.flatten(), seq, MAX_DEG)

            candidates.append((candidate, cost, seq[:], deg))
            sequences.append(seq[:])

        candidates = sorted(candidates, key=itemgetter(1))
        best_cost_gen = candidates[0][1]
        best_cand = candidates[0][0]


        if evaluation.is_better(best_cost_gen, best, minimizing=True, strict=True):
            deg = candidates[0][3]
            if deg1 != deg:
                linear_dependent = False
                count += 1
            else:
                linear_dependent = True

            print(f'generation {iter} liner dependency with parent:', linear_dependent, file=sys.stderr)

            best = best_cost_gen
            parent = best_cand

    counts[instance] = count

print(np.mean(counts))
