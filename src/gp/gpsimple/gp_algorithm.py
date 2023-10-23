# Copyright (C) 2023 -
# Roman Kalkreuth (Roman.Kalkreuth@lip6.fr)
# Computer Lab of Paris 6, Sorbonne Université (Paris, France)

import gp_fitness as fitness
from gp_tree import GPNode

__author__ = 'Roman Kalkreuth'
__copyright__ = 'Copyright (C) 2023, Roman Kalkreuth'
__version__ = '1.0'
__email__  = 'Roman.Kalkreuth@lip6.fr'

def one_plus_lambda(num_generations, lmbda, ideal_fitness, problem, fitness_metric="abs", minimizing_fitness=True,
                    strict_selection=True):
    """

    """
    parent = GPNode(init_tree=True)
    num_fitness_eval = 0

    prediction = problem.evaluate(parent)
    actual = problem.y_train
    best_fitness = fitness.calculate_fitness(actual, prediction, metric=fitness_metric)

    for gen in range(num_generations):
        for i in range(lmbda):
            offspring = parent
            offspring.mutate()

            prediction = problem.evaluate(offspring)
            offspring_fitness = fitness.calculate_fitness(actual, prediction, metric=fitness_metric)

            num_fitness_eval += 1
            print("Generation #" + str(gen) + " - Best Fitness: " + str(best_fitness))

            if fitness.is_better(offspring_fitness, best_fitness, minimizing_fitness=minimizing_fitness,
                                 strict=strict_selection):
                best_fitness = offspring_fitness
                parent = offspring

                if fitness.is_ideal(best_fitness, ideal_fitness, minimizing_fitness=minimizing_fitness):
                    break

    return best_fitness, num_fitness_eval
