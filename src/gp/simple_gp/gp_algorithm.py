import gp_config as config
import gp_evaluation as evaluation
from gp_tree import ParseTree

def one_plus_lambda(num_generations, lmbda, dataset, strict=True):
    parent = ParseTree()
    best_fitness = evaluation.evaluate(parent, dataset)
    fitnesses = []

    for gen in range(num_generations):
        for i in range(lmbda):
            offspring = parent
            offspring.mutate()
            offspring_fitness = evaluation.evaluate(offspring, dataset)

            if evaluation.is_better(offspring_fitness, best_fitness, config.MINIMIZING_FITNESS):
                best_fitness = offspring_fitness
                parent = offspring

