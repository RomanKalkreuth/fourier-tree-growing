import gp_config as config
from gp_tree import ParseTree

def one_plus_lambda(num_generations, lmbda, dataset, strict_selection=True):
    parent = ParseTree()
    best = evaluate(parent, dataset)
    fitnesses = []

    for gen in range(num_generations):
        for i in range(lmbda):
            offspring = parent
            offspring.mutate()
            fitness = evaluate(offspring, dataset)

            if fitness < best:
                best = fitness
                parent = offspring

def evaluate(self, individual, dataset):
    return sum([abs(individual.evaluate(ds[0]) - ds[1]) for ds in dataset])