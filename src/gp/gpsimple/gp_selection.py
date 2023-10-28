import random as random
import gp_fitness as fitness
import gp_config as config

# Copyright (C) 2023 -
# Roman Kalkreuth (Roman.Kalkreuth@lip6.fr)
# Computer Lab of Paris 6, Sorbonne Université (Paris, France)

__author__ = 'Roman Kalkreuth'
__copyright__ = 'Copyright (C) 2023, Roman Kalkreuth'
__version__ = '1.0'
__email__  = 'Roman.Kalkreuth@lip6.fr'

def deterministic_tournament_selection(population,k):
    """

    """
    n = len(population)
    tournament = []

    for i in range(k):
        rand_index = random.randint(0, n)
        rand_ind = population[rand_index]
        tournament.append(rand_ind)

    best_ind, best_fit_val = tournament[0]
    best = ()

    for ind, fit_val in tournament:
        if fitness.is_better(fit_val, best_fit_val, config.MINIMIZING_FITNESS):
            best = (ind, fit_val)
            best_fit_val = fit_val

    return best




