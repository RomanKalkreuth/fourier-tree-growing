import random as random

def tournament_selection(population, k=2, minimizing_fitness=True):
    n = len(population)
    tournament = []

    for i in range(k):
        rand_index = random.randint(0, n - 1)
        rand_ind = population[rand_index]
        tournament.append(rand_ind)

    tournament.sort(key=lambda tup: tup[1], reverse=not minimizing_fitness)

    return tournament[0]