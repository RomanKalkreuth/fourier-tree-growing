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

LOG_FILE = None


def set_log_file(logfile):
    global LOG_FILE
    LOG_FILE = open(logfile, 'w')


class Poly:
    def __init__(self, ds, cs):
        self.ds = ds[:]
        self.cs = cs[:]

    def additive_op(self, op, poly):
        degrees_left = self.ds
        constants_left = self.cs
        degrees_right = poly.ds
        constants_right = poly.cs
        degrees, constants = [], []
        it_left, it_right = 0, 0
        while it_left < len(degrees_left) and it_right < len(degrees_right):
            if degrees_left[it_left] < degrees_right[it_right]:
                degrees.append(degrees_left[it_left])
                constants.append(constants_left[it_left])
                it_left += 1
            elif degrees_left[it_left] > degrees_right[it_right]:
                degrees.append(degrees_right[it_right])
                constants.append(op(0, constants_right[it_right]))
                it_right += 1
            else:
                degrees.append(degrees_left[it_left])
                constants.append(op(constants_left[it_left], constants_right[it_right]))
                it_left += 1
                it_right += 1
        while it_left < len(degrees_left):
            degrees.append(degrees_left[it_left])
            constants.append(constants_left[it_left])
            it_left += 1
        while it_right < len(degrees_right):
            degrees.append(degrees_right[it_right])
            constants.append(op(0, constants_right[it_right]))
            it_right += 1
        self.ds = degrees
        self.cs = constants
        return self

    def add(self, poly):
        return self.additive_op(Mathematical.add, poly)

    def sub(self, poly):
        return self.additive_op(Mathematical.sub, poly)

    def mul(self, poly):
        degrees_left = self.ds
        constants_left = self.cs
        degrees_right = poly.ds
        constants_right = poly.cs
        degrees, constants = [], []
        ds = np.zeros(len(degrees_left) * len(degrees_right), dtype=int)
        cs = [0] * (len(degrees_left) * len(degrees_right))
        cnt = 0
        for i in range(len(degrees_left)):
            for j in range(len(degrees_right)):
                ds[cnt] = degrees_left[i] + degrees_right[j]
                cs[cnt] = constants_left[i] * constants_right[j]
                cnt += 1
        sorted_ids = np.argsort(ds)
        prv_degree = ds[sorted_ids[0]]
        c = 0
        for i in sorted_ids:
            if prv_degree != ds[i]:
                degrees.append(int(prv_degree))
                constants.append(c)
                c = 0
            c += cs[i]
            prv_degree = ds[i]
        degrees.append(int(prv_degree))
        constants.append(c)
        self.ds = degrees
        self.cs = constants
        return self

    # integration with limits -1, 1
    def integrate1(self):
        ans, rest = 0, 0
        for d, c in zip(self.ds, self.cs):
            if ~d & 1:
                # if type(c) == int:
                    # x = c // (d + 1)
                    # rest += (c - x * (d + 1)) / (d + 1)
                    # ans += x
                # else:
                ans += c / (d + 1)
        return 2 * ans + 2 * rest

    # integration with limits 0, 1
    def integrate2(self):
        ans, rest = 0, 0
        for d, c in zip(self.ds, self.cs):
            if type(c) == int:
                x = c // (d + 1)
                rest += (c - x * (d + 1)) / (d + 1)
                ans += x
            else:
                ans += c / (d + 1)
        return ans + rest


# TREE_CS_TYPE = int


def tree_to_poly(tree):
    if not tree:
        return Poly([], [])
    poly_left = tree_to_poly(tree.left)
    poly_right = tree_to_poly(tree.right)
    if tree.symbol in VARIABLES:
        return Poly([1], [1])
    elif type(tree.symbol) == int or type(tree.symbol) == float or type(tree.symbol) == np.float64:
        return Poly([0], [tree.symbol])
    elif tree.symbol == Mathematical.add or tree.symbol == Mathematical.sub:
        return poly_left.additive_op(tree.symbol, poly_right)
    elif tree.symbol == Mathematical.mul:
        return poly_left.mul(poly_right)
    raise ValueError(f'Symbol {tree.symbol} is not supported')


def poly_to_str(poly, no_zero=False):
    sorted_degrees, constants = poly.ds, poly.cs
    a = []
    started = False
    for i in range(len(sorted_degrees)):
        if started:
            s = f'{abs(constants[i])}*x^{sorted_degrees[i]}'
            if constants[i] < 0:
                a.append(f' - {s}')
            elif constants[i] > 0:
                a.append(f' + {s}')
            else:
                if not no_zero:
                    a.append(f' + {s}')
        else:
            if not no_zero or constants[i] != 0:
                s = f'{constants[i]}*x^{sorted_degrees[i]}'
                a.append(s)
                started = True
    if not started:
        return '0'
    return ''.join(a)


def loss(F, y_poly):
    F_copy = Poly(F.ds, F.cs)
    F_copy.sub(y_poly)
    F_copy.mul(F_copy)
    return F_copy.integrate1()


def mylog(gen, y, fy, y_poly):
    print(f'gen:{gen}', f'depth:{y.depth()}', f'size:{y.size()}', f'loss:{fy:.15f}', poly_to_str(y_poly, True), sep=', ', file=LOG_FILE)


EPS = 1e-3


def mylog_ftg(gen, y, fy, y_poly, num_evals, cond, num_retries):
    print(f'gen:{gen}', f'depth:{y.depth()}', f'size:{y.size()}', f'num_evals:{num_evals}', f'cond:{cond:.3f}', f'loss:{fy:.15f}', f'retries:{num_retries}', poly_to_str(y_poly, True), sep=', ', file=LOG_FILE)


def eprintf(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class SR:
    def __init__(self, F):
        self.F = F
        self.num_evals = 0

    def loss(self, f):
        self.num_evals += 1
        return loss(self.F, f)

    def inner_q_space(self, v1, v2):
        self.num_evals += 1
        v1_copy = Poly(v1.ds, v1.cs)
        v1_copy.mul(v2)
        return v1_copy.integrate1()


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
    v1_poly = tree_to_poly(v1)
    G.add(v1_poly)
    fc = [sr_instance.inner_q_space(sr_instance.F, v1_poly)]
    Ginv = G.get_inverse()
    alpha1 = np.dot(Ginv, np.array(fc))
    F_hat = create_tree(alpha1, [v1])
    cnt_tree = 1
    vs = [v1]
    loss = float("inf")
    num_retries = 0
    while True:
        F_hat_poly = tree_to_poly(F_hat)
        prvloss = loss
        loss = sr_instance.loss(F_hat_poly)
        print(f'trees {cnt_tree}, loss {loss:.10f}')
        num_retries = 0
        if loss > prvloss:
            print('termination: numerical errors', file=LOG_FILE)
            break
        mylog_ftg(cnt_tree, F_hat, loss, F_hat_poly, sr_instance.num_evals, G.get_cond(), num_retries)
        if sr_instance.num_evals >= max_evaluations:
            print('termination: exhaust all the budget', file=LOG_FILE)
            break
        if loss <= stopping_criteria:
            print('termination: found good enough solution', file=LOG_FILE)
            break
        # perp = np.array([sr_instance.F(x) - sr_instance._eval_obj(F_hat, x) for x in sr_instance.X])
        perp_poly = Poly(sr_instance.F.ds, sr_instance.F.cs)
        perp_poly.sub(F_hat_poly)
        Ginv = None
        while Ginv is None:
            num_retries += 1
            v = treegen()
            v_poly = tree_to_poly(v)
            while sr_instance.inner_q_space(perp_poly, v_poly) < EPS:
            # while np.abs(np.dot(perp, sr_instance.eval_on_dataset(v))) < EPS:
                v = treegen()
                v_poly = tree_to_poly(v)
            # eprintf('generated:', util.generate_symbolic_expression(v, parse_tree.FUNCTIONS))
            cnt_tree += 1
            vs.append(v)
            G.add(v_poly)
            Ginv = G.get_inverse()
            if Ginv is None:
                G.pop()
                vs.pop()
                cnt_tree -= 1
            if sr_instance.num_evals >= max_evaluations:
                mylog_ftg(cnt_tree, F_hat, loss, F_hat_poly, sr_instance.num_evals, G.get_cond(), num_retries)
                print('termination: exhaust all the budget before lnd function is found', file=LOG_FILE)
                return
        fc.append(sr_instance.inner_q_space(sr_instance.F, v_poly))
        alpha = np.dot(Ginv, np.array(fc))
        F_hat = create_tree(alpha, vs)
    return F_hat


def one_plus_lambda(max_evaluations, lmbda, F,
                    tree_init_depth=(1, 1),
                    max_subtree_depth=6,
                    stopping_criteria=0,
                    minimizing=True):
    num_fitness_evals = 0
    parent = ParseTree()
    parent.init_tree(min_depth=tree_init_depth[0], max_depth=tree_init_depth[1])

    best_cost = loss(F, tree_to_poly(parent))
    print(f'gen {0}:', best_cost)

    mylog(0, parent, best_cost, tree_to_poly(parent))

    offsprings = []

    max_generations = max_evaluations // lmbda

    for gen in range(0, max_generations):
        for i in range(0, lmbda):
            offspring = parent.clone()
            variation.uniform_subtree_mutation(tree=offspring, max_depth=max_subtree_depth)
            offspring_poly = tree_to_poly(offspring)
            cost = loss(F, offspring_poly)
            offsprings.append((offspring, cost))
            num_fitness_evals += 1
            mylog(gen + 1, offspring, cost, offspring_poly)

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


def mu_plus_lambda(max_evaluations, mu, lmbda, F,
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

        cost = loss(F, tree_to_poly(parent))
        num_fitness_evals += 1

        parents.append((parent, cost))
        mylog(0, parent, cost, tree_to_poly(parent))

        if best_cost is None:
            best_cost = cost
        elif cost < best_cost:
            best_cost = cost
    print(f'gen {0}:', best_cost)
    for gen in range(0, max_generations):
        for i in range(0, lmbda):
            parent1 = parents[myrandom.RND.randint(0, mu - 1)]
            parent2 = parents[myrandom.RND.randint.randint(0, mu - 1)]
            ptree1, ptree2 = parent1[0], parent2[0]

            otree = breed_offspring(ptree1, ptree2, crossover_rate=crossover_rate,
                                    max_subtree_depth=max_subtree_depth,
                                    mutation_type='probabilistic',
                                    mutation_rate=mutation_rate)

            otree_poly = tree_to_poly(otree)
            cost = loss(F, otree_poly)
            mylog(gen + 1, otree, cost, otree_poly)
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


def canonical_ea(max_evaluations, F,
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

    population = init_population(population_size, tree_init_depth, F)
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

        evaluate_individuals(gen, offsprings, F)
        num_evaluations += num_offspring

        population = elites + offsprings

        population = sorted(population, key=itemgetter(1), reverse=not minimizing)
        best_cost = population[0][1]
        print(f'gen {gen + 1}:', best_cost)
        if best_cost <= stopping_criteria:
            break

    return num_evaluations


def init_population(population_size, tree_init_depth, F):
    population = []
    for i in range(population_size):
        individual = ParseTree()
        individual.init_tree(min_depth=tree_init_depth[0], max_depth=tree_init_depth[1])
        poly = tree_to_poly(individual)
        cost = loss(F, poly)
        population.append((individual, cost))
        mylog(0, individual, cost, poly)
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


def evaluate_individuals(gen, individuals, F):
    for index, individual in enumerate(individuals):
        individual_poly = tree_to_poly(individual[0])
        cost = loss(F, individual_poly)
        individuals[index] = (individual[0], cost)
        mylog(gen + 1, individual[0], cost, individual_poly)


def main():
    MIN_INIT_TREE_DEPTH = 2
    MAX_INIT_TREE_DEPTH = 6
    MAX_SUBTREE_DEPTH = 3

    MU = 1024
    LAMBDA = 500

    IDEAL_COST = 0
    NUM_EVALUATIONS = 100000

    MUTATION_RATE = 0.1
    CROSSOVER_RATE = 0.9

    TOURNAMENT_SIZE = 2
    POPULATION_SIZE = 500
    NUM_ELITES = 2

    DEGREE = 6

    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluations", type=int, help="Number of evaluations")


    parser.add_argument("--mu", type=int, help="Number of parents")
    parser.add_argument("--lmbda", type=int, help="Number of offsprings")

    parser.add_argument("--crate", type=float, help="Crossover rate")
    parser.add_argument("--mrate", type=float, help="Mutation rate")

    parser.add_argument("--popsize", type=int, help="Population size")
    parser.add_argument("--tsize", type=int, help="Tournament size")
    parser.add_argument("--nelites", type=int, help="Number of elites")

    required_named = parser.add_argument_group('Required Named Arguments')
    required_named.add_argument("--degree", type=int, help="Polynomial degree")
    required_named.add_argument("--instance", type=int, help="Number of the run")
    required_named.add_argument("--dirname", type=str, help="Name of the directory with result")
    required_named.add_argument("--algorithm", type=str, help="Search algorithm",
                                choices=['ftg', 'one-plus-lambda', 'canonical-ea'])
    required_named.add_argument("--constant", type=str, help="Constants in terminals",
                                choices=['none', '1', 'koza-erc'])

    args = parser.parse_args()

    INSTANCE = args.instance
    myrandom.set_random_seed(INSTANCE)
    DEGREE = args.degree
    ALGORITHM = args.algorithm
    if args.evaluations:
        NUM_EVALUATIONS = args.evaluations

    parse_tree.set_functions([Mathematical.add, Mathematical.mul, Mathematical.sub])
    if args.constant == 'none':
        parse_tree.set_terminals(['x'])
    elif args.constant == '1':
        parse_tree.set_terminals(['x', 1])
    elif args.constant == 'koza-erc':
        # global TREE_CS_TYPE
        parse_tree.set_terminals(['x', constants.koza_erc])
        # TREE_CS_TYPE = float

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
    F = Poly([i for i in range(DEGREE + 1)], [1 for _ in range(DEGREE + 1)])
    sr_instance = SR(F)
    if args.constant == 'none':
        F = Poly([i for i in range(1, DEGREE + 1)], [1 for _ in range(1, DEGREE + 1)])
    # F = Poly([DEGREE], [1])
    dirname = args.dirname
    os.makedirs(dirname, exist_ok=True)
    log_file = f'{dirname}/run-{INSTANCE}'
    print(f'Logging to {log_file}', flush=True)
    global LOG_FILE
    with open(log_file, 'w') as LOG_FILE:
        print('Target function:', poly_to_str(F, True), file=LOG_FILE)
        match ALGORITHM:
            case 'one-plus-lambda':
                one_plus_lambda(max_evaluations=NUM_EVALUATIONS,
                                lmbda=LAMBDA,
                                F=F,
                                tree_init_depth=(MIN_INIT_TREE_DEPTH, MAX_INIT_TREE_DEPTH),
                                max_subtree_depth=MAX_SUBTREE_DEPTH,
                                stopping_criteria=IDEAL_COST)
            case 'mu-plus-lambda':
                mu_plus_lambda(max_evaluations=NUM_EVALUATIONS,
                               mu=MU,
                               lmbda=LAMBDA,
                               F=F,
                               crossover_rate=CROSSOVER_RATE,
                               mutation_rate=MUTATION_RATE,
                               tree_init_depth=(MIN_INIT_TREE_DEPTH, MAX_INIT_TREE_DEPTH),
                               max_subtree_depth=MAX_SUBTREE_DEPTH,
                               stopping_criteria=IDEAL_COST)
            case 'canonical-ea':
                canonical_ea(max_evaluations=NUM_EVALUATIONS,
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
                ftg(max_evaluations=NUM_EVALUATIONS,
                    sr_instance=sr_instance,
                    stopping_criteria=IDEAL_COST)
        print('Target function:', poly_to_str(F, True), file=LOG_FILE)


if __name__ == '__main__':
    main()
