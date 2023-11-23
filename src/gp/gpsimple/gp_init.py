import gp_config as config


def init_config():
    gpc = config.GPConfig(num_jobs=config.NUM_JOBS,
                          max_generations=config.MAX_GENERATIONS,
                          functions=config.FUNCTIONS,
                          terminals=config.TERMINALS,
                          variables=config.VARIABLES,
                          function_class=config.FUNCTION_CLASS,
                          num_functions=len(config.FUNCTIONS),
                          num_terminals=len(config.TERMINALS),
                          num_variables=len(config.VARIABLES),
                          minimizing_fitness=config.MINIMIZING_FITNESS,
                          stopping_criteria=config.STOPPING_CRITERIA,
                          fitness_metric=config.FITNESS_METRIC,
                          minimalistic_output=config.MINIMALISTIC_OUTPUT,
                          silent=config.SILENT
                          )
    return gpc


def init_hyperparameters(algorithm):
    hp = {}

    hp['max_generations'] = config.MAX_GENERATIONS
    hp['stopping_criteria'] = config.STOPPING_CRITERIA
    hp['fitness_metric'] = config.FITNESS_METRIC
    hp['minimizing_fitness'] = config.MINIMIZING_FITNESS
    hp['tree_init_depth'] = (config.MIN_INIT_TREE_DEPTH, config.MAX_INIT_TREE_DEPTH)
    hp['subtree_depth'] = config.SUBTREE_DEPTH
    hp['silent'] = config.SILENT

    hp['functions'] = config.FUNCTIONS
    hp['terminals'] = config.TERMINALS

    match algorithm.__name__:
        case "one_plus_lambda_ea":
            hp['lambda'] = config.LAMBDA
            hp['mutation_rate'] = config.MUTATION_RATE
        case "canonical_ea":
            hp['population_size'] = config.POPULATION_SIZE
            hp['num_elites'] = config.NUM_ELITES
            hp['mutation_rate'] = config.MUTATION_RATE
            hp['crossover_rate'] = config.CROSSOVER_RATE
            hp['tournament_size'] = config.TOURNAMENT_SIZE
    return hp

