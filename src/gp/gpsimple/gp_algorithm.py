# Copyright (C) 2023 -
# Roman Kalkreuth (Roman.Kalkreuth@lip6.fr)
# Computer Lab of Paris 6, Sorbonne Université (Paris, France)

import gp_fitness as fitness
import gp_selection as selection
import gp_crossover as crossover
from gp_tree import GPNode

__author__ = 'Roman Kalkreuth'
__copyright__ = 'Copyright (C) 2023, Roman Kalkreuth'
__version__ = '1.0'
__email__ = 'Roman.Kalkreuth@lip6.fr'


def one_plus_lambda_ea(max_generations=10000000,
                       lmbda=1,
                       tree_init_depth=(2,6),
                       subtree_depth=3,
                       mutation_rate=0.01,
                       stopping_criteria=0.01,
                       problem=None,
                       fitness_metric="abs",
                       minimizing_fitness=True,
                       strict_selection=True,
                       silent=False,
                       hyperparameters=None):

    if hyperparameters is not None:
        lmbda = hyperparameters['lambda']
        mutation_rate = hyperparameters['mutation_rate']
        tree_init_depth = hyperparameters['tree_init_depth']
        subtree_depth = hyperparameters['subtree_depth']
        stopping_criteria = hyperparameters['stopping_criteria']

    parent = GPNode()
    parent.init_tree(min_depth=tree_init_depth[0], max_depth=tree_init_depth[1])
    num_evaluations = 0

    prediction = problem.evaluate(parent)
    actual = problem.y_train
    best_fitness = fitness.calculate_fitness(actual, prediction, metric=fitness_metric)


    #TODO Fix bug related with the determination of the best offspring by using sorted
    for gen in range(max_generations):
        for i in range(lmbda):
            offspring = parent
            offspring.mutate(mutation_rate=mutation_rate, subtree_depth=subtree_depth)

            prediction = problem.evaluate(offspring)
            offspring_fitness = fitness.calculate_fitness(actual, prediction, metric=fitness_metric)

            num_evaluations += 1
            if not silent:
                print("Generation #" + str(gen) + " - Best Fitness: " + str(best_fitness))

            if fitness.is_better(offspring_fitness, best_fitness, minimizing_fitness=minimizing_fitness,
                                 strict=strict_selection):
                best_fitness = offspring_fitness
                parent = offspring

                if fitness.is_ideal(best_fitness, stopping_criteria, minimizing_fitness=minimizing_fitness):
                    if not silent:
                        print("Ideal fitness reached in generation #" + str(gen))
                    break

    return best_fitness, num_evaluations


def canonical_ea(max_generations=100,
                 population_size=500,
                 tree_init_depth=(2,6),
                 subtree_depth=3,
                 crossover_rate=0.9,
                 mutation_rate=0.01,
                 tournament_size=7,
                 stopping_criteria=0.01,
                 num_elites=1,
                 problem=None,
                 fitness_metric="abs",
                 minimizing_fitness=True,
                 silent=False,
                 hyperparameters=None):
    if hyperparameters is not None:
        max_generations = hyperparameters['max_generations']
        population_size = hyperparameters['population_size']
        tree_init_depth = hyperparameters['tree_init_depth']
        subtree_depth = hyperparameters['subtree_depth']
        crossover_rate = hyperparameters['crossover_rate']
        mutation_rate = hyperparameters['mutation_rate']
        tournament_size = hyperparameters['tournament_size']
        stopping_criteria = hyperparameters['stopping_criteria']
        num_elites = hyperparameters['num_elites']

    population = init_population(population_size, tree_init_depth, problem, fitness_metric)
    num_offspring = population_size - num_elites
    num_evaluations = 0

    for gen in range(max_generations):
        sort_individuals(population, minimizing_fitness=minimizing_fitness)
        best_fitness = population[0][1]
        elites = population[0:num_elites]
        offspring = []

        if fitness.is_ideal(best_fitness, ideal_fitness=stopping_criteria):
            if not silent:
                print("Ideal fitness reached in generation #" + str(gen))
            break

        if not silent:
            print("Generation #" + str(gen) + " - Best Fitness: " + str(best_fitness))

        for i in range(num_offspring):
            parent1 = selection.tournament_selection(population, tournament_size)
            parent2 = selection.tournament_selection(population, tournament_size)

            ptree1, ptree2 = parent1[0], parent2[0]
            otree = breed_offspring(ptree1, ptree2, crossover_rate, mutation_rate, subtree_depth)

            offspring.append((otree, None))

        evaluate_individuals(offspring, problem, fitness_metric=fitness_metric)
        num_evaluations += num_offspring

        population = elites + offspring

    return best_fitness, num_evaluations


def sort_individuals(population, minimizing_fitness=True):
    population.sort(key=lambda tup: tup[1], reverse=not minimizing_fitness)
    return population


def breed_offspring(tree1, tree2, crossover_rate, mutation_rate, subtree_depth):
    otree1, otree2 = crossover.subtree_crossover(tree1, tree2, crossover_rate=crossover_rate)
    otree1.mutate(mutation_rate=mutation_rate, subtree_depth=subtree_depth)
    return otree1


def evaluate_individuals(individuals, problem, fitness_metric):
    actual = problem.y_train
    for index, individual in enumerate(individuals):
        prediction = problem.evaluate(individual[0])
        fitness_val = fitness.calculate_fitness(actual, prediction, metric=fitness_metric)
        individuals[index] = (individual[0], fitness_val)

def init_population(population_size, tree_init_depth, problem, fitness_metric):
    actual = problem.y_train
    population = []
    for i in range(population_size):
        individual = GPNode()
        individual.init_tree(min_depth=tree_init_depth[0], max_depth=tree_init_depth[1])
        prediction = problem.evaluate(individual)
        fitness_val = fitness.calculate_fitness(actual, prediction, metric=fitness_metric)
        population.append((individual, fitness_val))
    return population
