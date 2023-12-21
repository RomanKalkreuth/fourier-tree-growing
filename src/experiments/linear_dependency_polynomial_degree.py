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
MAX_SUBTREE_DEPTH = 3
IDEAL_COST = 10e-2
GENERATIONS = 10000
MUTATION_RATE = 0.1
LAMBDA = 8
INSTANCES = 1
MAX_DEG = 20

objective_function = functions.koza1
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
    best_cost = evaluation.evaluate(parent, X, y, f_eval)
    count = 0

    for generation in range(0, GENERATIONS):

        seq1 = [float(parent.evaluate(x)) for x in X]
        #deg1 = analysis.polynomial_degree_fit(X.flatten(), seq1, MAX_DEG)
        x_lin, y_lin = analysis.function_values(parent, -10, 10, 100)
        deg1 = analysis.polynomial_degree_fit(x_lin, y_lin, MAX_DEG)
        depth1 = parent.depth()

        sequences.clear()
        candidates.clear()

        sequences.append(seq1[:])

        for i in range(0, LAMBDA):
            candidate = parent.clone()
            #variation.probabilistic_subtree_mutation(tree=candidate, mutation_rate=MUTATION_RATE,
            #                                         max_depth=SUBTREE_DEPTH)

            variation.uniform_subtree_mutation(tree=candidate, max_depth=MAX_SUBTREE_DEPTH)

            cost = evaluation.evaluate(candidate, X, y, f_eval)
            seq = [float(candidate.evaluate(x)) for x in X]

            x_lin, y_lin = analysis.function_values(candidate, -10, 10, 100)
            deg = analysis.polynomial_degree_fit(x_lin, y_lin, MAX_DEG)

            #deg = analysis.polynomial_degree_fit(X.flatten(), seq, MAX_DEG)

            candidates.append((candidate, cost, seq[:], deg))
            sequences.append(seq[:])

        candidates = sorted(candidates, key=itemgetter(1))
        best_cost_gen = candidates[0][1]
        best_candidate = candidates[0][0]

        if evaluation.is_better(best_cost_gen, best_cost, minimizing=True, strict=True):
            deg = candidates[0][3]
            depth = best_candidate.depth()

            if deg1 != deg:
                linear_dependency = False
                count += 1
            else:
                linear_dependency = True

            best_cost = best_cost_gen
            parent = best_candidate

            print(f'Generation {generation}, best cost: {best_cost}, linear dependency with parent: {linear_dependency}, polynomial degree of parent: {deg1}, '
                  f'polynomial degree of offspring: {deg}, depth of parent: {depth1}, depth of offspring: {depth}', file=sys.stderr)

            #print(f'Expression: {parent}', file=sys.stderr)

            if evaluation.is_ideal(best_cost, ideal_cost=IDEAL_COST):
                print(f'Ideal fitness reached in generation {generation}', file=sys.stderr)
                print(f'Expression: {parent}', file=sys.stderr)
                break

    counts[instance] = count

print(np.mean(counts))
