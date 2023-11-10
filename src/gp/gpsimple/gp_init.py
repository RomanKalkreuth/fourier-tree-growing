from gp_tree import GPNode
import gp_fitness as fitness
import gp_config as config

def init_population(size, problem, fitness_metric):
    actual = problem.y_train
    population = []
    for i in range(size):
        individual = GPNode(init_tree=True)
        prediction = problem.eval(individual)
        fitness_val = fitness.calculate_fitness(actual, prediction, metric=fitness_metric)
        population.append((individual, fitness_val))
    return population

def init_parameters(algorithm):
    parameters = {}

    parameters['max_generations'] = config.MAX_GENERATIONS
    parameters['stopping_criteria'] = config.STOPPING_CRITERIA
    parameters['metric'] = config.FITNESS_METRIC
    parameters['minimizing_fitness'] = config.MINIMIZING_FITNESS
    parameters['init_depth'] = (config.MIN_INIT_TREE_DEPTH, config.MAX_INIT_TREE_DEPTH)
    parameters['subtree_depth'] = config.SUBTREE_DEPTH

    parameters['functions'] = config.FUNCTIONS
    parameters['terminals'] = config.TERMINALS

    match algorithm.__name__:
        case "one_plus_lambda_ea":
            parameters['lambda'] = config.LAMBDA
            parameters['mutation_rate'] = config.MUTATION_RATE
        case "canonical_ea":
            parameters['population_size'] = config.POPULATION_SIZE
            parameters['num_elites'] = config.NUM_ELITES
            parameters['mutation_rate'] = config.MUTATION_RATE
            parameters['crossover_rate'] = config.CROSSOVER_RATE
            parameters['tournament_size'] = config.TOURNAMENT_SIZE

    return parameters