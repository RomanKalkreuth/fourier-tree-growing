import random
import sys
from operator import itemgetter

sys.path.insert(0, '../representation')
sys.path.insert(0, '../evaluation')
sys.path.insert(0, '../variation')
sys.path.insert(0, '../analysis')

from src.representation.parse_tree import ParseTree
import src.evaluation.evaluation as evaluation
import src.variation.variation as variation
import src.selection.selection as selection


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


def one_plus_lambda(max_evaluations, lmbda, X, y, f_eval,
                    tree_init_depth=(2, 6),
                    max_subtree_depth=3,
                    stopping_criteria=0.01,
                    minimizing=True):
    num_fitness_evals = 0
    parent = ParseTree()
    parent.init_tree(min_depth=tree_init_depth[0], max_depth=tree_init_depth[1])

    best_cost = evaluation.evaluate(parent, X, y, f_eval)

    offsprings = []

    max_generations = max_evaluations // lmbda

    for gen in range(0, max_generations):
        for i in range(0, lmbda):
            offspring = parent.clone()
            variation.uniform_subtree_mutation(tree=offspring, max_depth=max_subtree_depth)
            cost = evaluation.evaluate(offspring, X, y, f_eval)
            offsprings.append((offspring, cost))
            num_fitness_evals += 1

        offsprings = sorted(offsprings, key=itemgetter(1), reverse=not minimizing)
        best_cost_gen = offsprings[0][1]
        best_offspring = offsprings[0][0]

        if evaluation.is_better(best_cost_gen, best_cost, minimizing=minimizing, strict=True):
            best_cost = best_cost_gen
            parent = best_offspring

        if evaluation.is_ideal(best_cost, ideal_cost=stopping_criteria):
            break

    return num_fitness_evals


def mu_plus_lambda(max_evaluations, mu, lmbda, X, y, f_eval,
                   crossover_rate=0.9,
                   mutation_rate=0.1,
                   tree_init_depth=(2, 6),
                   max_subtree_depth=3,
                   stopping_criteria=0.01,
                   minimizing=True):
    num_fitness_evals = 0
    parents = []
    offsprings = []

    best_cost = None

    max_generations = max_evaluations // lmbda

    for i in range(0, mu):
        parent = ParseTree()
        parent.init_tree(min_depth=tree_init_depth[0], max_depth=tree_init_depth[1])

        cost = evaluation.evaluate(parent, X, y, f_eval)
        num_fitness_evals += 1

        parents.append((parent, cost))

        if best_cost is None:
            best_cost = cost
        elif evaluation.is_better(cost, best_cost, minimizing=minimizing, strict=True):
            best_cost = cost

    for gen in range(0, max_generations):
        for i in range(0, lmbda):
            parent1 = parents[random.randint(0, mu - 1)]
            parent2 = parents[random.randint(0, mu - 1)]
            ptree1, ptree2 = parent1[0], parent2[0]

            otree = breed_offspring(ptree1, ptree2, crossover_rate=crossover_rate,
                                    max_subtree_depth=max_subtree_depth,
                                    mutation_type='probabilistic',
                                    mutation_rate=mutation_rate)

            cost = evaluation.evaluate(otree, X, y, f_eval)
            num_fitness_evals += 1

            offsprings.append((otree, cost))

        offsprings = sorted(offsprings, key=itemgetter(1), reverse=not minimizing)
        parents = offsprings[0:mu]

        best_cost_gen = offsprings[0][1]

        if evaluation.is_better(best_cost_gen, best_cost, minimizing=minimizing, strict=True):
            best_cost = best_cost_gen

        if evaluation.is_ideal(best_cost, ideal_cost=stopping_criteria):
            break

    return num_fitness_evals


def canonical_ea(max_evaluations, X, y, f_eval,
                 population_size=500,
                 tree_init_depth=(2, 6),
                 max_subtree_depth=3,
                 crossover_rate=0.9,
                 mutation_rate=0.01,
                 tournament_size=7,
                 stopping_criteria=0.01,
                 num_elites=1,
                 minimizing=True):
    num_evaluations = 0
    num_offspring = population_size - num_elites

    population = init_population(population_size, tree_init_depth, X, y, f_eval)
    num_evaluations += population_size

    max_generations = max_evaluations // num_offspring

    for gen in range(0, max_generations):
        population = sorted(population, key=itemgetter(1), reverse=not minimizing)
        best_cost = population[0][1]
        elites = population[0:num_elites]
        offsprings = []

        for i in range(num_offspring):
            parent1 = selection.tournament_selection(population, tournament_size)
            parent2 = selection.tournament_selection(population, tournament_size)

            ptree1, ptree2 = parent1[0], parent2[0]

            otree = breed_offspring(ptree1, ptree2, crossover_rate=crossover_rate,
                                    max_subtree_depth=max_subtree_depth,
                                    mutation_type='probabilistic',
                                    mutation_rate=mutation_rate)

            offsprings.append((otree, None))

        evaluate_individuals(offsprings, X, y, f_eval)
        num_evaluations += num_offspring

        population = elites + offsprings

        if evaluation.is_ideal(best_cost, ideal_cost=stopping_criteria):
            break

    return num_evaluations


def init_population(population_size, tree_init_depth, X, y, f_eval):
    population = []
    for i in range(population_size):
        individual = ParseTree()
        individual.init_tree(min_depth=tree_init_depth[0], max_depth=tree_init_depth[1])
        cost = evaluation.evaluate(individual, X, y, f_eval)
        population.append((individual, cost))
    return population


def breed_offspring(tree1, tree2, crossover_rate, max_subtree_depth, mutation_type='probabilistic', mutation_rate=0.1):
    otree1, otree2 = variation.subtree_crossover(tree1, tree2, crossover_rate=crossover_rate)
    if mutation_type == 'probabilistic':
        variation.probabilistic_subtree_mutation(tree=otree1, mutation_rate=mutation_rate, max_depth=max_subtree_depth)
    elif mutation_type == 'uniform':
        variation.uniform_subtree_mutation(tree=otree1, max_depth=max_subtree_depth)
    else:
        raise RuntimeError("Unknown mutation type selected")
    return otree1


def evaluate_individuals(individuals, X, y, f_eval):
    for index, individual in enumerate(individuals):
        cost = evaluation.evaluate(individual[0], X, y, f_eval)
        individuals[index] = (individual[0], cost)
