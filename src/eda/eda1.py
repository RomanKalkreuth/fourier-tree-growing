# @author Kirill Antonov

import sys
sys.path.insert(0, '../gp/gpsimple')

import copy
from gp_tree import GPNode
import gp_config as config
import gp_problem as problem
import gp_fitness as fitness
import gp_util as util
import tree_convert as convert

import src.benchmark.symbolic_regression.dataset_generator as generator
import src.benchmark.symbolic_regression.functions as benchmarks

import math
import numpy as np

from myrandom import RandomEngine
import mylogger



def objfunction(tree_list_str):
    tree_list = copy.copy(tree_list_str)
    convert.symbols_to_type(tree_list, config.FUNCTIONS, config.TERMINALS)
    tree = convert.list_to_tree(tree_list, config.FUNCTIONS)
    err = util.validate_tree(tree, config.FUNCTIONS, config.TERMINALS)
    X_train = generator.random_samples_float(-1.0, 1.0, 20)
    y_train = generator.generate_function_values(benchmark, X_train)
    regression_problem = problem.RegressionProblem(X_train, y_train)
    prediction = regression_problem.evaluate(tree)
    actual = regression_problem.y_train
    fitness_val = fitness.calculate_fitness(actual, prediction, metric="abs")
    return fitness_val + math.exp(err) - 1



def D_upd(D, d, mu_, to_id, MINDEPTH, p):
    for i in range(len(D[d])):
        cnt = np.zeros(len(D[d][i]), dtype=int)
        for j in range(mu_):
            if 2**(d+MINDEPTH)-1 != len(p[j]):
                continue
            x, value = p[j]
            cnt[function_to_id[x[i]]] += 1
        if np.sum(cnt) == 0:
            return
        for j in range(len(D[d][i])):
            D[d][i][j] = cnt[j] / mu_


def gp_umda(f, is_term, mu_, lambda_, MINDEPTH, MAXDEPTH, FNS, TNS):
    D_dep = [1/(MAXDEPTH - MINDEPTH + 1) for i in range(MINDEPTH, MAXDEPTH + 1)]
    D_fns = [[[1/len(FNS) for j in range(len(FNS))] for i in range(2**(d-1)-1)] for d in range(MINDEPTH, MAXDEPTH + 1)]
    D_tns = [[[1/len(TNS) for j in range(len(TNS))] for i in range(2**(d-1))] for d in range(MINDEPTH, MAXDEPTH + 1)]
    found_min = float("inf")
    found_argmin = None
    spent_budget = 0
    while not is_term(spent_budget, found_min):
        p = []
        for i in range(lambda_):
            d = RandomEngine.sample_discrete_dist(D_dep)
            fl = [None] * (2**(MINDEPTH+d-1)-1)
            tl = [None] * (2**(MINDEPTH+d-1))
            for i in range(len(fl)):
                fl[i] = FNS[RandomEngine.sample_discrete_dist(D_fns[d][i])]
            for i in range(len(tl)):
                tl[i] = TNS[RandomEngine.sample_discrete_dist(D_tns[d][i])]
            x = fl + tl
            value_x = f(x)
            spent_budget += 1
            p.append((x, value_x))
        p.sort(key=lambda xy: xy[1])
        if p[0][1] < found_min:
            found_argmin, found_min = copy.copy(p[0])
        cnt = np.zeros(len(D_dep), dtype=int)
        for i in range(mu_):
            d = 0
            x, value = p[i]
            while 2**d - 1 != len(x):
                d +=1
            cnt[d-MINDEPTH] += 1
        for i in range(len(D_dep)):
            D_dep[i] = cnt[i] / mu_
        for d in range(0, MAXDEPTH - MINDEPTH + 1):
            D_upd(D_fns, d, mu_, function_to_id, MINDEPTH, p)
            D_upd(D_tns, d, mu_, terminal_to_id, MINDEPTH, p)
    return found_argmin, found_min


def gp_rs(f, is_term, MINDEPTH, MAXDEPTH, FNS, TNS):
    argmin_, min_ = None, float("inf")
    spent_budget = 0
    while not is_term(spent_budget, min_):
        d = RandomEngine.sample_int_uniform(MINDEPTH, MAXDEPTH + 1)
        fns = np.random.choice(FNS, 2**(d-1)-1)
        tns = np.random.choice(TNS, 2**(d-1))
        x = fns.tolist() + tns.tolist()
        value_x = f(x)
        spent_budget += 1
        if value_x < min_:
            argmin_, min_ = x, value_x
    return argmin_, min_


def add_logger(f, ndim: int, root_name: str, alg_name: str, alg_info: str, f_name: str):
    wrapped_f = mylogger.MyObjectiveFunctionWrapper(f, dim=ndim, fname=f_name)
    global logger
    logger = mylogger.MyLogger(root=root_name,
                               folder_name="everyeval",
                               algorithm_name=alg_name,
                               algorithm_info=alg_info,
                               isLogArg=True)
    logger_best = mylogger.MyLogger(root=root_name,
                                    folder_name="bestsofar",
                                    algorithm_name=alg_name,
                                    algorithm_info=alg_info,
                                    logStrategy=mylogger.LoggingBestSoFar,
                                    isLogArg=True)
    wrapped_f.attach_logger(logger)
    wrapped_f.attach_logger(logger_best)
    return wrapped_f


def experiment():
    functions = []
    function_to_id = {}
    for f in config.FUNCTIONS:
        function_to_id[f.__name__] = len(functions)
        functions.append(f.__name__)

    terminals = []
    terminal_to_id = {}
    for t in config.TERMINALS:
        terminal_to_id[str(t)] = len(terminals)
        terminals.append(str(t))

    global benchmark
    for benchmark in [benchmarks.koza1, benchmarks.koza2, benchmarks.koza3]:
        for i in range(10):
            print(f'Run {i} ...')
            profiled_objfunction = add_logger(objfunction, 1, f'gp-rs-exp-{benchmark.__name__}', 'rs-umda', 'rs for GP simple impl', benchmark.__name__)
            gp_umda(profiled_objfunction, lambda spent, value: spent >= 10000 or value == 0, 100, 1000, 2, 7, functions, terminals)
            gp_rs(profiled_objfunction, lambda spent, value: spent >= 10000 or value == 0, 2, 7, functions, terminals)


experiment()
