import sys
import os
import myrandom
import argparse
import src.representation.parse_tree as parse_tree
from src.representation.parse_tree import ParseTree, FUNCTIONS, VARIABLES, TERMINALS
from src.functions.functions import Mathematical
import numpy as np
import src.variation.variation as variation
import src.selection.selection as selection
import src.constants.constants as constants
from operator import itemgetter
import src.benchmark.symbolic_regression.benchmark_functions as benchmarks
import util.util as util


EPS = 1e-3


def set_log_file(logfile):
    global LOG_FILE
    LOG_FILE = open(logfile, 'w')


def mylog(gen, y, fy, num_evals):
    print(f'gen:{gen}', f'num_evals:{num_evals}', f'depth:{y.depth()}', f'size:{y.size()}', f'loss:{fy:.15f}', sep=', ', file=LOG_FILE)


def mylog_ftg(gen, y, fy, num_evals, cond, num_retries):
    print(f'gen:{gen}', f'num_evals:{num_evals}', f'depth:{y.depth()}', f'size:{y.size()}', f'cond:{cond:.3f}', f'loss:{fy:.15f}', f'retries:{num_retries}', sep=', ', file=LOG_FILE)

def eprintf(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class SR:
    def __init__(self, X, F):
        self.X = X
        self.F = F
        self.num_evals = 0

    def eval_on_dataset(self, f):
        self.num_evals += 1
        return np.array([self._eval_obj(f, x) for x in self.X])

    def _eval_obj(self, obj, x):
        if type(obj) == parse_tree.ParseTree:
            ans = obj.evaluate(x)
        else:
            ans = obj(x)
        return ans

    def loss(self, f):
        self.num_evals += 1
        loss = 0
        for i in range(len(self.X)):
            loss += (self.F(self.X[i]) - self._eval_obj(f, self.X[i]))**2
        return loss

    def inner_q_space(self, v1, v2):
        self.num_evals += 1
        prod = 0
        for x in self.X:
            prod += self._eval_obj(v1, x) * self._eval_obj(v2, x)
        return prod


class SRGrammMatrix:
    def __init__(self, sr_instance):
        self.elements = []
        self.py_matrix = []
        self.sr_instance = sr_instance

    def add(self, element):
        self.elements.append(element)
        for i in range(len(self.py_matrix)):
            self.py_matrix[i].append(self.sr_instance.inner_q_space(self.elements[i], element))
        new_row = []
        for e in self.elements:
            new_row.append(self.sr_instance.inner_q_space(e, element))
        self.py_matrix.append(new_row)

    def pop(self):
        self.elements.pop()
        self.py_matrix.pop()
        for p in self.py_matrix:
            p.pop()

    def get_inverse(self):
        G = np.array(self.py_matrix)
        # print('cond number:', np.linalg.cond(G))
        # return np.linalg.inv(G)
        u, s, v = np.linalg.svd(G)
        Ginv = np.dot(v.transpose(), np.dot(np.diag(s ** -1), u.transpose()))
        if not np.allclose(np.dot(G, Ginv), np.identity(len(G)), atol=1e-4):
            return None
        return Ginv

    def get_cond(self):
        G = np.array(self.py_matrix)
        return np.linalg.cond(G)


def treegen():
    v = parse_tree.ParseTree()
    v.init_tree(1, 10)
    return v


def create_tree(alpha, v):
    if len(v) == 0:
        return None
    tm = parse_tree.ParseTree()
    tm.symbol = Mathematical.mul
    tm.left = parse_tree.ParseTree()
    tm.left.symbol = alpha[0]
    tm.right = v[0]
    if len(v) == 1:
        # eprintf('composed:', util.generate_symbolic_expression(tm,parse_tree.FUNCTIONS))
        return tm
    left = tm
    cnt = 1
    while cnt < len(v):
        ta = parse_tree.ParseTree()
        ta.symbol = Mathematical.add
        ta.left = left
        tm = parse_tree.ParseTree()
        tm.symbol = Mathematical.mul
        tm.left = parse_tree.ParseTree()
        tm.left.symbol = alpha[cnt]
        tm.right = v[cnt]
        ta.right = tm
        left = ta
        cnt += 1
    # eprintf('composed:', util.generate_symbolic_expression(left,parse_tree.FUNCTIONS))
    return left


def ftg(max_evaluations, sr_instance, stopping_criteria=0):
    G = SRGrammMatrix(sr_instance)
    v1 = parse_tree.ParseTree()
    v1.symbol = 1
    G.add(v1)
    fc = [sr_instance.inner_q_space(sr_instance.F, v1)]
    Ginv = G.get_inverse()
    alpha1 = np.dot(Ginv, np.array(fc))
    F_hat = create_tree(alpha1, [v1])
    cnt_tree = 1
    vs = [v1]
    prvloss = float("inf")
    num_retries = 0
    while True:
        loss = sr_instance.loss(F_hat)
        print(f'trees {cnt_tree}, loss {loss:.10f}')
        mylog_ftg(cnt_tree, F_hat, loss, sr_instance.num_evals, G.get_cond(), num_retries)
        num_retries = 0
        if loss > prvloss:
            print('termination: numerical errors', file=LOG_FILE)
            break
        if cnt_tree >= len(sr_instance.X):
            print('termination: spanned all the space', file=LOG_FILE)
            break
        if sr_instance.num_evals >= max_evaluations:
            print('termination: exhaust all the budget', file=LOG_FILE)
            break
        if loss <= stopping_criteria:
            print('termination: found good enough solution', file=LOG_FILE)
            break
        perp = np.array([sr_instance.F(x) - sr_instance._eval_obj(F_hat, x) for x in sr_instance.X])
        sr_instance.num_evals += 1
        Ginv = None
        while Ginv is None:
            num_retries += 1
            v = treegen()
            while np.abs(np.dot(perp, sr_instance.eval_on_dataset(v))) < EPS:
                v = treegen()
            # eprintf('generated:', util.generate_symbolic_expression(v, parse_tree.FUNCTIONS))
            cnt_tree += 1
            vs.append(v)
            G.add(v)
            Ginv = G.get_inverse()
            if Ginv is None:
                G.pop()
                vs.pop()
                cnt_tree -= 1
            if sr_instance.num_evals >= max_evaluations:
                mylog_ftg(cnt_tree, F_hat, loss, sr_instance.num_evals, G.get_cond(), num_retries)
                print('termination: exhaust all the budget before lnd function is found', file=LOG_FILE)
                return
        fc.append(sr_instance.inner_q_space(sr_instance.F, v))
        alpha = np.dot(Ginv, np.array(fc))
        F_hat = create_tree(alpha, vs)
    return F_hat


def one_plus_lambda(max_evaluations, lmbda, sr_instance,
                    tree_init_depth=(1, 1),
                    max_subtree_depth=6,
                    stopping_criteria=0,
                    minimizing=True):
    num_fitness_evals = 0
    parent = ParseTree()
    parent.init_tree(min_depth=tree_init_depth[0], max_depth=tree_init_depth[1])

    best_cost = sr_instance.loss(parent)
    print(f'gen {0}:', best_cost)

    mylog(0, parent, best_cost, sr_instance.num_evals)

    offsprings = []

    max_generations = max_evaluations // lmbda

    for gen in range(0, max_generations):
        for i in range(0, lmbda):
            offspring = parent.clone()
            variation.uniform_subtree_mutation(tree=offspring, max_depth=max_subtree_depth)
            cost = sr_instance.loss(offspring)
            offsprings.append((offspring, cost))
            num_fitness_evals += 1
            mylog(gen + 1, offspring, cost, sr_instance.num_evals)

        offsprings = sorted(offsprings, key=itemgetter(1), reverse=not minimizing)
        best_cost_gen = offsprings[0][1]
        best_offspring = offsprings[0][0]

        if best_cost_gen < best_cost:
            best_cost = best_cost_gen
            parent = best_offspring
            print(f'gen {gen + 1}:', best_cost)

        if best_cost <= stopping_criteria:
            break

    return num_fitness_evals


def mu_plus_lambda(max_evaluations, mu, lmbda, sr_instance,
                   crossover_rate=0.9,
                   mutation_rate=0.1,
                   tree_init_depth=(2, 6),
                   max_subtree_depth=3,
                   stopping_criteria=0,
                   minimizing=True):
    num_fitness_evals = 0
    parents = []
    offsprings = []

    best_cost = None

    max_generations = max_evaluations // lmbda

    for i in range(0, mu):
        parent = ParseTree()
        parent.init_tree(min_depth=tree_init_depth[0], max_depth=tree_init_depth[1])

        cost = sr_instance.loss(parent)
        num_fitness_evals += 1

        parents.append((parent, cost))
        mylog(0, parent, cost, sr_instance.num_evals)

        if best_cost is None:
            best_cost = cost
        elif cost < best_cost:
            best_cost = cost
    print(f'gen {0}:', best_cost)
    for gen in range(0, max_generations):
        for i in range(0, lmbda):
            parent1 = parents[myrandom.RND.randint(0, mu - 1)]
            parent2 = parents[myrandom.RND.randint(0, mu - 1)]
            ptree1, ptree2 = parent1[0], parent2[0]

            otree = breed_offspring(ptree1, ptree2, crossover_rate=crossover_rate,
                                    max_subtree_depth=max_subtree_depth,
                                    mutation_type='probabilistic',
                                    mutation_rate=mutation_rate)

            cost = sr_instance.loss(otree)
            mylog(gen + 1, otree, cost, sr_instance.num_evals)
            num_fitness_evals += 1

            offsprings.append((otree, cost))

        offsprings = sorted(offsprings, key=itemgetter(1), reverse=not minimizing)
        parents = offsprings[0:mu]

        best_cost_gen = offsprings[0][1]

        if best_cost_gen < best_cost:
            best_cost = best_cost_gen
            print(f'gen {gen + 1}:', best_cost)

        if best_cost <= stopping_criteria:
            break

    return num_fitness_evals


def canonical_ea(max_evaluations, sr_instance,
                 population_size=500,
                 tree_init_depth=(2, 6),
                 max_subtree_depth=3,
                 crossover_rate=0.9,
                 mutation_rate=0.01,
                 tournament_size=7,
                 stopping_criteria=0,
                 num_elites=1,
                 minimizing=True):
    num_evaluations = 0
    num_offspring = population_size - num_elites

    population = init_population(population_size, tree_init_depth, sr_instance)
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

        evaluate_individuals(gen, offsprings, sr_instance)
        num_evaluations += num_offspring

        population = elites + offsprings

        population = sorted(population, key=itemgetter(1), reverse=not minimizing)
        best_cost = population[0][1]
        print(f'gen {gen + 1}:', best_cost)
        if best_cost <= stopping_criteria:
            break

    return num_evaluations


def init_population(population_size, tree_init_depth, sr_instance):
    population = []
    for i in range(population_size):
        individual = ParseTree()
        individual.init_tree(min_depth=tree_init_depth[0], max_depth=tree_init_depth[1])
        cost = sr_instance.loss(individual)
        population.append((individual, cost))
        mylog(0, individual, cost, sr_instance.num_evals)
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


def evaluate_individuals(gen, individuals, sr_instance):
    for index, individual in enumerate(individuals):
        cost = sr_instance.loss(individual[0])
        individuals[index] = (individual[0], cost)
        mylog(gen + 1, individual[0], cost, sr_instance.num_evals)


def main():
    MIN_INIT_TREE_DEPTH = 2
    MAX_INIT_TREE_DEPTH = 6
    MAX_SUBTREE_DEPTH = 3

    MU = 1024
    LAMBDA = 500

    IDEAL_COST = 1e-8
    NUM_EVALUATIONS = 100000

    MUTATION_RATE = 0.1
    CROSSOVER_RATE = 0.9

    TOURNAMENT_SIZE = 2
    POPULATION_SIZE = 500
    NUM_ELITES = 2

    DEGREE = 6

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

    parser = argparse.ArgumentParser('Experiments on classical benchmarks')
    parser.add_argument("--evaluations", type=int, help="Number of evaluations")


    parser.add_argument("--mu", type=int, help="Number of parents")
    parser.add_argument("--lmbda", type=int, help="Number of offsprings")

    parser.add_argument("--crate", type=float, help="Crossover rate")
    parser.add_argument("--mrate", type=float, help="Mutation rate")

    parser.add_argument("--popsize", type=int, help="Population size")
    parser.add_argument("--tsize", type=int, help="Tournament size")
    parser.add_argument("--nelites", type=int, help="Number of elites")

    parser.add_argument("--cnt", type=int, help="Number of instances to run sequentially starting from $instance", default=1)

    required_named = parser.add_argument_group('Required Named Arguments')
    required_named.add_argument("--benchmark", type=str, help="Benchmark problem",
                                choices=BENCHMARS1D.keys())
    required_named.add_argument("--instance", type=int, help="Number of the run")
    required_named.add_argument("--dirname", type=str, help="Name of the directory with result")
    required_named.add_argument("--algorithm", type=str, help="Search algorithm",
                                choices=['one-plus-lambda', 'canonical-ea', 'ftg'])
    required_named.add_argument("--constant", type=str, help="Constants in terminals",
                                choices=['none', '1', 'koza-erc'])

    args = parser.parse_args()

    INSTANCE = args.instance
    F = BENCHMARS1D[args.benchmark]
    ALGORITHM = args.algorithm
    if args.evaluations:
        NUM_EVALUATIONS = args.evaluations

    parse_tree.set_functions([Mathematical.add, Mathematical.mul, Mathematical.sub, Mathematical.sin, Mathematical.log, Mathematical.cos, Mathematical.div])
    if args.constant == 'none':
        parse_tree.set_terminals(['x'])
    elif args.constant == '1':
        parse_tree.set_terminals(['x', 1])
    elif args.constant == 'koza-erc':
        parse_tree.set_terminals(['x', constants.koza_erc])

    match ALGORITHM:
        case 'one-plus-lambda':
            if args.lmbda:
                LAMBDA = args.lmbda
        case 'mu-plus-lambda':
            if args.lmbda:
                LAMBDA = args.lmbda
            if args.mu:
                MU = args.mu
            if args.crate:
                CROSSOVER_RATE = args.crate
        case 'canonical-ea':
            if args.crate:
                CROSSOVER_RATE = args.crate
            if args.mrate:
                MUTATION_RATE = args.mrate
            if args.popsize:
                POPULATION_SIZE = args.popsize
            if args.tsize:
                TOURNAMENT_SIZE = args.tsize
            if args.nelites:
                NUM_ELITES = args.nelites

    dirname = args.dirname
    os.makedirs(dirname, exist_ok=True)
    for i in range(INSTANCE, INSTANCE + args.cnt):
        myrandom.set_random_seed(i)
        if args.benchmark.startswith('nguyen') and int(args.benchmark.lstrip('nguyen')) <= 6:
            X = [myrandom.RND.uniform(-1, 1) for _ in range(20)]
        elif args.benchmark == 'nguyen7':
            X = [myrandom.RND.uniform(0, 2) for _ in range(20)]
        elif args.benchmark == 'nguyen8':
            X = [myrandom.RND.uniform(0, 4) for _ in range(20)]
        else:
            X = [myrandom.RND.uniform(-1, 1) for _ in range(20)]
        sr_instance = SR(X, F)
        log_file = f'{dirname}/run-{i}'
        print(f'Logging to {log_file}', flush=True)
        global LOG_FILE
        with open(log_file, 'w') as LOG_FILE:
            match ALGORITHM:
                case 'one-plus-lambda':
                    one_plus_lambda(max_evaluations=NUM_EVALUATIONS,
                                    lmbda=LAMBDA,
                                    sr_instance=sr_instance,
                                    tree_init_depth=(MIN_INIT_TREE_DEPTH, MAX_INIT_TREE_DEPTH),
                                    max_subtree_depth=MAX_SUBTREE_DEPTH,
                                    stopping_criteria=IDEAL_COST)
                case 'mu-plus-lambda':
                    mu_plus_lambda(max_evaluations=NUM_EVALUATIONS,
                                   mu=MU,
                                   lmbda=LAMBDA,
                                   sr_instance=sr_instance,
                                   crossover_rate=CROSSOVER_RATE,
                                   mutation_rate=MUTATION_RATE,
                                   tree_init_depth=(MIN_INIT_TREE_DEPTH, MAX_INIT_TREE_DEPTH),
                                   max_subtree_depth=MAX_SUBTREE_DEPTH,
                                   stopping_criteria=IDEAL_COST)
                case 'canonical-ea':
                    canonical_ea(max_evaluations=NUM_EVALUATIONS,
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
                    ftg(max_evaluations=NUM_EVALUATIONS,
                        sr_instance=sr_instance,
                        stopping_criteria=IDEAL_COST)


if __name__ == '__main__':
    main()
