import sys
sys.path.insert(0, '../representation')
sys.path.insert(0, '../evaluation')
sys.path.insert(0, '../variation')
sys.path.insert(0, '../analysis')

import src.representation.parse_tree as ParseTree
import src.evaluation.evaluation as evaluation
import src.variation.variation as variation
import src.analysis.analysis as analysis

def hill_climbing(n_iter, X, y, f_eval, ideal, minimizing=True, strict=True):
    best = ParseTree
    best_cost = evaluation.evaluate(best, X, y, f_eval)

    for i in range(0, n_iter):
        candidate = best.clone()
        cost = evaluation.evaluate(candidate, X, y, f_eval)

        if evaluation.is_better(cost, best_cost, minimizing=minimizing, strict=strict):
            best = candidate

        if evaluation.is_ideal(best_cost, ideal, minimizing=minimizing):
            break

def one_plus_lambda(max_generations, lmbda, X, y, f_eval,
                    tree_init_depth=(2, 6),
                    max_subtree_depth=3,
                    mutation_rate=0.1,
                    stopping_criteria=0.01,
                    minimizing=True,
                    max_deg=30) :

    parent = ParseTree()
    parent.init_tree(min_depth=tree_init_depth[0], max_depth=tree_init_depth[1])
    best_cost = evaluation.evaluate(parent, X, y, f_eval)

    sequences = []
    candidates = []

    for gen in range(0, max_generations):
        seq1 = [float(parent.evaluate(x)) for x in X]

        for i in range(0, lmbda):
            candidate = parent.clone()
            variation.uniform_subtree_mutation(tree=candidate, max_depth=max_subtree_depth)
            cost = evaluation.evaluate(candidate, X, y, f_eval)
            seq = [float(candidate.evaluate(x)) for x in X]
            x_lin, y_lin = analysis.function_values(candidate, -10, 10, 100)
            deg = analysis.polynomial_degree_fit(x_lin, y_lin, max_deg)
            candidates.append((candidate, cost, seq[:], deg))

#def mu_plus_lambda(n_iter, mu, lbmda, X, y, f_eval, ideal, minimizing=True, strict=True):

