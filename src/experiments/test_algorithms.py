import pytest
import importlib
import myrandom
import numpy as np
import filecmp
import src.benchmark.symbolic_regression.benchmark_functions as benchmarks
from src.functions.functions import Mathematical
import src.constants.constants as constants
import os
import large_scale_poly_2 as lsp2
import src.representation.parse_tree as parse_tree

ftgexpr = importlib.import_module("ftg-expr")


def generate_test_data():
    timestamp = "01-02-2024"
    MIN_INIT_TREE_DEPTH = 2
    MAX_INIT_TREE_DEPTH = 6
    MAX_SUBTREE_DEPTH = 3
    MU = 1024
    LAMBDA = 500
    IDEAL_COST = 1e-8
    NUM_EVALUATIONS = 100
    MUTATION_RATE = 0.1
    CROSSOVER_RATE = 0.9
    TOURNAMENT_SIZE = 2
    POPULATION_SIZE = 500
    NUM_ELITES = 2
    BENCHMARS1D = {
        'koza1': benchmarks.koza1,
        'koza2': benchmarks.koza2,
        'koza3': benchmarks.koza3,
        'nguyen3': benchmarks.nguyen3,
        'nguyen4': benchmarks.nguyen4,
        'nguyen5': benchmarks.nguyen5,
        'nguyen6': benchmarks.nguyen6,
        'nguyen7': benchmarks.nguyen7,
        'nguyen8': benchmarks.nguyen8,
    }
    parse_tree.set_functions(
        [Mathematical.add, Mathematical.mul, Mathematical.sub, Mathematical.sin, Mathematical.log,
         Mathematical.cos, Mathematical.div])
    for alg in ["ftg", "one-plus-lambda", "canonical-ea"]:
        for lambda_val in [1, 10]:
            if alg == "canonical-ea" and lambda_val == 1:
                continue
            if alg == "ftg" and lambda_val == 10:
                continue
            for benchmark in ['koza1', 'koza2', 'koza3', 'nguyen3', 'nguyen4', 'nguyen5', 'nguyen6', 'nguyen7',
                              'nguyen8']:
                for constant in ["1", "none", "koza-erc"]:
                    if constant == 'none':
                        parse_tree.set_terminals(['x'])
                    elif constant == '1':
                        parse_tree.set_terminals(['x', 1])
                    elif constant == 'koza-erc':
                        parse_tree.set_terminals(['x', constants.koza_erc])
                    for instance in range(5):
                        dirname = f"testdata/conventional-benchmarks/A-{alg}_L{lambda_val}_B-{benchmark}_C-{constant}_{timestamp}"
                        os.makedirs(dirname, exist_ok=True)
                        F = BENCHMARS1D[benchmark]
                        myrandom.set_random_seed(instance)
                        if benchmark.startswith('nguyen') and int(benchmark.lstrip('nguyen')) <= 6:
                            X = [myrandom.RND.uniform(-1, 1) for _ in range(20)]
                        elif benchmark == 'nguyen7':
                            X = [myrandom.RND.uniform(0, 2) for _ in range(20)]
                        elif benchmark == 'nguyen8':
                            X = [myrandom.RND.uniform(0, 4) for _ in range(20)]
                        else:
                            X = [myrandom.RND.uniform(-1, 1) for _ in range(20)]
                        sr_instance = ftgexpr.SR(X, F)
                        ftgexpr.set_log_file(f'{dirname}/run-{instance}')
                        match alg:
                            case 'one-plus-lambda':
                                ftgexpr.one_plus_lambda(max_evaluations=NUM_EVALUATIONS,
                                                        lmbda=lambda_val,
                                                        sr_instance=sr_instance,
                                                        tree_init_depth=(MIN_INIT_TREE_DEPTH, MAX_INIT_TREE_DEPTH),
                                                        max_subtree_depth=MAX_SUBTREE_DEPTH,
                                                        stopping_criteria=IDEAL_COST)
                            case 'mu-plus-lambda':
                                ftgexpr.mu_plus_lambda(max_evaluations=NUM_EVALUATIONS,
                                                       mu=MU,
                                                       lmbda=LAMBDA,
                                                       sr_instance=sr_instance,
                                                       crossover_rate=CROSSOVER_RATE,
                                                       mutation_rate=MUTATION_RATE,
                                                       tree_init_depth=(MIN_INIT_TREE_DEPTH, MAX_INIT_TREE_DEPTH),
                                                       max_subtree_depth=MAX_SUBTREE_DEPTH,
                                                       stopping_criteria=IDEAL_COST)
                            case 'canonical-ea':
                                ftgexpr.canonical_ea(max_evaluations=NUM_EVALUATIONS,
                                                     sr_instance=sr_instance,
                                                     population_size=POPULATION_SIZE,
                                                     tree_init_depth=(MIN_INIT_TREE_DEPTH, MAX_INIT_TREE_DEPTH),
                                                     max_subtree_depth=MAX_SUBTREE_DEPTH,
                                                     crossover_rate=CROSSOVER_RATE,
                                                     mutation_rate=MUTATION_RATE,
                                                     tournament_size=TOURNAMENT_SIZE,
                                                     stopping_criteria=IDEAL_COST,
                                                     num_elites=NUM_ELITES)
                            case 'ftg':
                                ftgexpr.ftg(max_evaluations=NUM_EVALUATIONS,
                                            sr_instance=sr_instance,
                                            stopping_criteria=IDEAL_COST)
                        ftgexpr.LOG_FILE.close()
                        # assert filecmp.cmp(f'{dirname}/run-{instance}', 'tmp1.csv')


def generate_lsp_test_data():
    timestamp = "01-02-2024"
    MIN_INIT_TREE_DEPTH = 2
    MAX_INIT_TREE_DEPTH = 6
    MAX_SUBTREE_DEPTH = 3
    MU = 1024
    LAMBDA = 500
    IDEAL_COST = 0
    NUM_EVALUATIONS = 100
    MUTATION_RATE = 0.1
    CROSSOVER_RATE = 0.9
    TOURNAMENT_SIZE = 2
    POPULATION_SIZE = 500
    NUM_ELITES = 2
    parse_tree.set_functions([Mathematical.add, Mathematical.mul, Mathematical.sub])
    for alg in ["ftg", "one-plus-lambda", "canonical-ea"]:
        for lambda_val in [1, 10]:
            if alg == "canonical-ea" and lambda_val == 1:
                continue
            if alg == "ftg" and lambda_val == 10:
                continue
            for degree in [10, 100]:
                for constant in ["1", "none", "koza-erc"]:
                    if constant == 'none':
                        parse_tree.set_terminals(['x'])
                    elif constant == '1':
                        parse_tree.set_terminals(['x', 1])
                    elif constant == 'koza-erc':
                        parse_tree.set_terminals(['x', constants.koza_erc])
                    for instance in range(5):
                        dirname = f"testdata/lsp/A-{alg}_L{lambda_val}_D{degree}_C-{constant}_{timestamp}"
                        os.makedirs(dirname, exist_ok=True)
                        myrandom.set_random_seed(instance)
                        F = lsp2.Poly([i for i in range(degree + 1)], [1 for _ in range(degree + 1)])
                        sr_instance = lsp2.SR(F)
                        lsp2.set_log_file(f'{dirname}/run-{instance}')
                        match alg:
                            case 'one-plus-lambda':
                                lsp2.one_plus_lambda(max_evaluations=NUM_EVALUATIONS,
                                                     lmbda=lambda_val,
                                                     F=F,
                                                     tree_init_depth=(MIN_INIT_TREE_DEPTH, MAX_INIT_TREE_DEPTH),
                                                     max_subtree_depth=MAX_SUBTREE_DEPTH,
                                                     stopping_criteria=IDEAL_COST)
                            case 'canonical-ea':
                                lsp2.canonical_ea(max_evaluations=NUM_EVALUATIONS,
                                                  F=F,
                                                  population_size=POPULATION_SIZE,
                                                  tree_init_depth=(MIN_INIT_TREE_DEPTH, MAX_INIT_TREE_DEPTH),
                                                  max_subtree_depth=MAX_SUBTREE_DEPTH,
                                                  crossover_rate=CROSSOVER_RATE,
                                                  mutation_rate=MUTATION_RATE,
                                                  tournament_size=TOURNAMENT_SIZE,
                                                  stopping_criteria=IDEAL_COST,
                                                  num_elites=NUM_ELITES)
                            case 'ftg':
                                lsp2.ftg(max_evaluations=NUM_EVALUATIONS,
                                         sr_instance=sr_instance,
                                         stopping_criteria=IDEAL_COST)
                        lsp2.LOG_FILE.close()
                        # assert filecmp.cmp(f'{dirname}/run-{instance}', 'tmp1.csv')


def check_alg_conventional_benchmarks(alg):
    timestamp = "01-02-2024"
    MIN_INIT_TREE_DEPTH = 2
    MAX_INIT_TREE_DEPTH = 6
    MAX_SUBTREE_DEPTH = 3
    MU = 1024
    LAMBDA = 500
    IDEAL_COST = 1e-8
    NUM_EVALUATIONS = 100
    MUTATION_RATE = 0.1
    CROSSOVER_RATE = 0.9
    TOURNAMENT_SIZE = 2
    POPULATION_SIZE = 500
    NUM_ELITES = 2
    BENCHMARS1D = {
        'koza1': benchmarks.koza1,
        'koza2': benchmarks.koza2,
        'koza3': benchmarks.koza3,
        'nguyen3': benchmarks.nguyen3,
        'nguyen4': benchmarks.nguyen4,
        'nguyen5': benchmarks.nguyen5,
        'nguyen6': benchmarks.nguyen6,
        'nguyen7': benchmarks.nguyen7,
        'nguyen8': benchmarks.nguyen8,
    }
    parse_tree.set_functions(
        [Mathematical.add, Mathematical.mul, Mathematical.sub, Mathematical.sin, Mathematical.log,
         Mathematical.cos, Mathematical.div])
    for lambda_val in [1, 10]:
        if alg == "canonical-ea" and lambda_val == 1:
            continue
        if alg == "ftg" and lambda_val == 10:
            continue
        for benchmark in ['koza1', 'koza2', 'koza3', 'nguyen3', 'nguyen4', 'nguyen5', 'nguyen6', 'nguyen7',
                          'nguyen8']:
            for constant in ["1", "none", "koza-erc"]:
                if constant == 'none':
                    parse_tree.set_terminals(['x'])
                elif constant == '1':
                    parse_tree.set_terminals(['x', 1])
                elif constant == 'koza-erc':
                    parse_tree.set_terminals(['x', constants.koza_erc])
                for instance in range(5):
                    dirname = f"testdata/conventional-benchmarks/A-{alg}_L{lambda_val}_B-{benchmark}_C-{constant}_{timestamp}"
                    F = BENCHMARS1D[benchmark]
                    myrandom.set_random_seed(instance)
                    if benchmark.startswith('nguyen') and int(benchmark.lstrip('nguyen')) <= 6:
                        X = [myrandom.RND.uniform(-1, 1) for _ in range(20)]
                    elif benchmark == 'nguyen7':
                        X = [myrandom.RND.uniform(0, 2) for _ in range(20)]
                    elif benchmark == 'nguyen8':
                        X = [myrandom.RND.uniform(0, 4) for _ in range(20)]
                    else:
                        X = [myrandom.RND.uniform(-1, 1) for _ in range(20)]
                    sr_instance = ftgexpr.SR(X, F)
                    originial_log_file = f'{dirname}/run-{instance}'
                    ftgexpr.set_log_file('tmp.csv')
                    match alg:
                        case 'one-plus-lambda':
                            ftgexpr.one_plus_lambda(max_evaluations=NUM_EVALUATIONS,
                                                    lmbda=lambda_val,
                                                    sr_instance=sr_instance,
                                                    tree_init_depth=(MIN_INIT_TREE_DEPTH, MAX_INIT_TREE_DEPTH),
                                                    max_subtree_depth=MAX_SUBTREE_DEPTH,
                                                    stopping_criteria=IDEAL_COST)
                        case 'canonical-ea':
                            ftgexpr.canonical_ea(max_evaluations=NUM_EVALUATIONS,
                                                 sr_instance=sr_instance,
                                                 population_size=POPULATION_SIZE,
                                                 tree_init_depth=(MIN_INIT_TREE_DEPTH, MAX_INIT_TREE_DEPTH),
                                                 max_subtree_depth=MAX_SUBTREE_DEPTH,
                                                 crossover_rate=CROSSOVER_RATE,
                                                 mutation_rate=MUTATION_RATE,
                                                 tournament_size=TOURNAMENT_SIZE,
                                                 stopping_criteria=IDEAL_COST,
                                                 num_elites=NUM_ELITES)
                        case 'ftg':
                            ftgexpr.ftg(max_evaluations=NUM_EVALUATIONS,
                                        sr_instance=sr_instance,
                                        stopping_criteria=IDEAL_COST)
                    ftgexpr.LOG_FILE.close()
                    assert filecmp.cmp(originial_log_file, 'tmp.csv')


def check_alg_lsp(alg):
    timestamp = "01-02-2024"
    MIN_INIT_TREE_DEPTH = 2
    MAX_INIT_TREE_DEPTH = 6
    MAX_SUBTREE_DEPTH = 3
    MU = 1024
    LAMBDA = 500
    IDEAL_COST = 0
    NUM_EVALUATIONS = 100
    MUTATION_RATE = 0.1
    CROSSOVER_RATE = 0.9
    TOURNAMENT_SIZE = 2
    POPULATION_SIZE = 500
    NUM_ELITES = 2
    parse_tree.set_functions([Mathematical.add, Mathematical.mul, Mathematical.sub])
    for lambda_val in [1, 10]:
        if alg == "canonical-ea" and lambda_val == 1:
            continue
        if alg == "ftg" and lambda_val == 10:
            continue
        for degree in [10, 100]:
            for constant in ["1", "none", "koza-erc"]:
                if constant == 'none':
                    parse_tree.set_terminals(['x'])
                elif constant == '1':
                    parse_tree.set_terminals(['x', 1])
                elif constant == 'koza-erc':
                    parse_tree.set_terminals(['x', constants.koza_erc])
                for instance in range(5):
                    dirname = f"testdata/lsp/A-{alg}_L{lambda_val}_D{degree}_C-{constant}_{timestamp}"
                    myrandom.set_random_seed(instance)
                    F = lsp2.Poly([i for i in range(degree + 1)], [1 for _ in range(degree + 1)])
                    sr_instance = lsp2.SR(F)
                    lsp2.set_log_file('tmp.csv')
                    match alg:
                        case 'one-plus-lambda':
                            lsp2.one_plus_lambda(max_evaluations=NUM_EVALUATIONS,
                                                 lmbda=lambda_val,
                                                 F=F,
                                                 tree_init_depth=(MIN_INIT_TREE_DEPTH, MAX_INIT_TREE_DEPTH),
                                                 max_subtree_depth=MAX_SUBTREE_DEPTH,
                                                 stopping_criteria=IDEAL_COST)
                        case 'canonical-ea':
                            lsp2.canonical_ea(max_evaluations=NUM_EVALUATIONS,
                                              F=F,
                                              population_size=POPULATION_SIZE,
                                              tree_init_depth=(MIN_INIT_TREE_DEPTH, MAX_INIT_TREE_DEPTH),
                                              max_subtree_depth=MAX_SUBTREE_DEPTH,
                                              crossover_rate=CROSSOVER_RATE,
                                              mutation_rate=MUTATION_RATE,
                                              tournament_size=TOURNAMENT_SIZE,
                                              stopping_criteria=IDEAL_COST,
                                              num_elites=NUM_ELITES)
                        case 'ftg':
                            lsp2.ftg(max_evaluations=NUM_EVALUATIONS,
                                     sr_instance=sr_instance,
                                     stopping_criteria=IDEAL_COST)
                    lsp2.LOG_FILE.close()
                    assert filecmp.cmp(f'{dirname}/run-{instance}', 'tmp.csv')


def test_ftg_on_conventional():
    check_alg_conventional_benchmarks('ftg')


def test_one_plus_lambda_on_conventional():
    check_alg_conventional_benchmarks('one-plus-lambda')


def test_canonical_ea_on_conventional():
    check_alg_conventional_benchmarks('canonical-ea')


def test_ftg_on_lsp():
    check_alg_lsp('ftg')


def test_one_plus_lambda_on_lsp():
    check_alg_lsp('one-plus-lambda')


def test_canonical_ea_on_lsp():
    check_alg_lsp('canonical-ea')


# def test_gen_lsp():
#     generate_test_data()
#     generate_lsp_test_data()
