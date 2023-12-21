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
import src.variation.variation as variation

import src.benchmark.symbolic_regression.dataset_generator as generator
import src.benchmark.symbolic_regression.benchmark_functions as functions

random.seed()

MIN_INIT_TREE_DEPTH = 2
MAX_INIT_TREE_DEPTH = 6
MAX_SUBTREE_DEPTH = 6
GENERATIONS = 100
IDEAL_COST = 10e-2
MUTATION_RATE = 0.1
LAMBDA = 20
INSTANCES = 2

objective_function = functions.nguyen4
X = generator.random_samples_float(-1.0, 1.0, 20, dim=1)
y = generator.generate_function_values(objective_function, X)
f_eval = evaluation.absolute_error

sequences = []
candidates = []

counts = np.zeros(INSTANCES)
depths = np.zeros(LAMBDA)

for instance in range(0, INSTANCES):
    parent = ParseTree()
    parent.init_tree(min_depth=MIN_INIT_TREE_DEPTH, max_depth=MAX_INIT_TREE_DEPTH)
    best_cost = evaluation.evaluate(parent, X, y, f_eval)
    seq1 = [float(parent.evaluate(x)) for x in X]

    count = 0
    for generation in range(0, GENERATIONS):

        sequences.clear()
        candidates.clear()

        sequences.append(seq1[:])

        for i in range(0, LAMBDA):
            candidate = parent.clone()
            variation.probabilistic_subtree_mutation(tree=candidate, mutation_rate=MUTATION_RATE,
                                                     max_depth=MAX_SUBTREE_DEPTH)

            cost = evaluation.evaluate(candidate, X, y, f_eval)
            seq = [float(candidate.evaluate(x)) for x in X]

            candidates.append((candidate, cost, seq[:]))
            sequences.append(seq[:])

        candidates = sorted(candidates, key=itemgetter(1))
        best_cost_gen = candidates[0][1]
        best_cand = candidates[0][0]

        if evaluation.is_better(best_cost_gen, best_cost, minimizing=True, strict=True):
            linear_dependency = analysis.linear_dependency([seq1, candidates[0][2]])

            if not linear_dependency:
                print(f'generation {generation} liner dependency with parent:', linear_dependency, file=sys.stderr)
            best_cost = best_cost_gen
            parent = best_cand
            v1 = candidates[0][2]
            if linear_dependency:
                count += 1

        if evaluation.is_ideal(best_cost, ideal_cost=IDEAL_COST):
            print(f'Ideal fitness reached in generation {generation}', file=sys.stderr)
            break

    counts[instance] = count

print(np.mean(counts))
