import gp_config as config
from gp_tree import ParseTree

def one_plus_lambda(num_generations, lmbda, dataset, strict_selection=True):
    parent = ParseTree()
    best_fitness = evaluate_individual(parent)
    fitnesses = []

    for gen in range(num_generations):
        for i in range(lmbda):
            offspring = parent
            offspring.mutate()
            fitness = evaluate_individual(parent)

def evaluate_individual(ind, dataset):
    for i

def create_variables_dict(data):
    dict = {}
    for term in config.TERMINALS
        if term == isinstance(term, str):

