import gp_config as config
import gp_fitness as fitness
from gp_tree import ParseTree

def one_plus_lambda(num_generations, lmbda, dataset, strict=True):
    parent = ParseTree()
    best_fitness = fitness.absolute_error(parent, dataset)
    fitnesses = []

    for gen in range(num_generations):
        for i in range(lmbda):
            offspring = parent
            offspring.mutate()
            offspring_fitness = fitness.absolute_error(offspring, dataset)

            if fitness.is_better(offspring_fitness, best_fitness, minimizing_fitness=config.MINIMIZING_FITNESS,
                                 strict=strict):
                best_fitness = offspring_fitness
                parent = offspring

