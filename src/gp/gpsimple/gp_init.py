
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
                          silent_algorithm=config.SILENT_ALGORITHM,
                          silent_evolver=config.SILENT_EVOLVER
                          )
    return gpc


def init_hyperparameters(algorithm):
    hp = {}
    hp['tree_init_depth'] = (config.MIN_INIT_TREE_DEPTH, config.MAX_INIT_TREE_DEPTH)
    hp['subtree_depth'] = config.SUBTREE_DEPTH

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

def validate_hyperparameters(hp, algorithm):

    assert isinstance(hp['tree_init_depth'], tuple)

    assert isinstance(hp['tree_init_depth'][0], int)
    assert isinstance(hp['tree_init_depth'][1], int)

    assert hp['tree_init_depth'][0] > 0
    assert hp['tree_init_depth'][1] > 0

    assert hp['tree_init_depth'][0] < hp['tree_init_depth'][1]

    assert isinstance(hp['subtree_depth'], int)
    assert isinstance(hp['subtree_depth'], int)

    match algorithm.__name__:
        case "one_plus_lambda_ea":
            assert isinstance(hp['lambda'], int)
            assert hp['lambda'] > 0

            assert isinstance(hp['mutation_rate'], float)
            assert 0.0 <= hp['mutation_rate'] <= 1.0
        case "canonical_ea":

            assert isinstance(hp['population_size'], int)
            assert hp['population_size'] > 0

            assert isinstance(hp['num_elites'], int)
            assert hp['num_elites'] > 0

            assert isinstance(hp['mutation_rate'], float)
            assert 0.0 <= hp['mutation_rate'] <= 1.0

            assert isinstance(hp['crossover_rate'], float)
            assert 0.0 <= hp['crossover_rate'] <= 1.0

            assert isinstance(hp['tournament_size'], int)
            assert hp['tournament_size'] > 0
