import sys
sys.path.insert(0, '../representation')
sys.path.insert(0, '../evaluation')

import src.representation.parse_tree as ParseTree
import src.evaluation.evaluation as evaluation

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

#def one_plus_lambda(n_iter, lbmda, X, y, f_eval, ideal, minimizing=True, strict=True):
#def mu_plus_lambda(n_iter, mu, lbmda, X, y, f_eval, ideal, minimizing=True, strict=True):