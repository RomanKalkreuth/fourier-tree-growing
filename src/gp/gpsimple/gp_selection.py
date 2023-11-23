import random as random
import gp_fitness as fitness
import gp_config as config

# Copyright (C) 2023 -
# Roman Kalkreuth (Roman.Kalkreuth@lip6.fr)
# Computer Lab of Paris 6, Sorbonne Université (Paris, France)

__author__ = 'Roman Kalkreuth'
__copyright__ = 'Copyright (C) 2023, Roman Kalkreuth'
__version__ = '1.0'
__email__ = 'Roman.Kalkreuth@lip6.fr'


def tournament_selection(population, k=2, minimizing_fitness=True):
    n = len(population)
    tournament = []

    for i in range(k):
        rand_index = random.randint(0, n - 1)
        rand_ind = population[rand_index]
        tournament.append(rand_ind)

    tournament.sort(key=lambda tup: tup[1], reverse=not minimizing_fitness)

    return tournament[0]
