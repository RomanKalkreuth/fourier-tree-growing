import argparse
import sys

sys.path.insert(0, '../benchmark/symbolic_regression')

import src.benchmark.symbolic_regression.dataset_generator as generator
import src.benchmark.symbolic_regression.benchmark_functions as functions
import src.evaluation.evaluation as evaluation
import src.algorithm.algorithm as algorithm
import random

INSTANCES = 100
ALGORITHM = algorithm.one_plus_lambda
EVALUATION_FUNCTION = evaluation.absolute_error

MIN_INIT_TREE_DEPTH = 2
MAX_INIT_TREE_DEPTH = 6
MAX_SUBTREE_DEPTH = 3

MU = 1024
LAMBDA = 1024

IDEAL_COST = 10e-2
NUM_EVALUATIONS = 1000000

MUTATION_RATE = 0.01
CROSSOVER_RATE = 0.9

TOURNAMENT_SIZE = 7
POPULATION_SIZE = 1024
NUM_ELITES = 2

DEGREE = 6

random.seed()


def print_configuarion():
    print("Instances: %d" % INSTANCES)
    print("Evaluations: %d" % NUM_EVALUATIONS)
    print("Algorithm: %s" % str(ALGORITHM))
    print("Degree: %s" % str(DEGREE))
    print("")
    match ALGORITHM:
        case algorithm.one_plus_lambda:
            print("Lambda: %d" % LAMBDA)
        case algorithm.canonical_ea:
            print("Population size: %d" % POPULATION_SIZE)
            print("Tournament size: %d" % TOURNAMENT_SIZE)
            print("Elites: %d" % NUM_ELITES)
            print("Crossover rate: %f" % CROSSOVER_RATE)
            print("Mutation rate: %f" % MUTATION_RATE)


def run(instances):

    X = generator.random_samples_float(-1.0, 1.0, 20, dim=1)
    y = generator.generate_polynomial_values(X, degree=DEGREE)

    for instance in range(0, instances):
        match ALGORITHM:
            case algorithm.one_plus_lambda:
                result = algorithm.one_plus_lambda(max_evaluations=NUM_EVALUATIONS,
                                                      lmbda=LAMBDA,
                                                      X=X, y=y,
                                                      f_eval=EVALUATION_FUNCTION,
                                                      tree_init_depth=(MIN_INIT_TREE_DEPTH, MAX_INIT_TREE_DEPTH),
                                                      max_subtree_depth=MAX_SUBTREE_DEPTH,
                                                      stopping_criteria=IDEAL_COST)
            case algorithm.mu_plus_lambda:
                result = algorithm.mu_plus_lambda(max_evaluations=NUM_EVALUATIONS,
                                                     mu=MU,
                                                     lmbda=LAMBDA,
                                                     X=X, y=y,
                                                     f_eval=EVALUATION_FUNCTION,
                                                     crossover_rate=CROSSOVER_RATE,
                                                     mutation_rate=MUTATION_RATE,
                                                     tree_init_depth=(MIN_INIT_TREE_DEPTH, MAX_INIT_TREE_DEPTH),
                                                     max_subtree_depth=MAX_SUBTREE_DEPTH,
                                                     stopping_criteria=IDEAL_COST)
            case algorithm.canonical_ea:
                result = algorithm.canonical_ea(max_evaluations=NUM_EVALUATIONS,
                                                   X=X, y=y,
                                                   f_eval=EVALUATION_FUNCTION,
                                                   population_size=POPULATION_SIZE,
                                                   tree_init_depth=(MIN_INIT_TREE_DEPTH, MAX_INIT_TREE_DEPTH),
                                                   max_subtree_depth=MAX_SUBTREE_DEPTH,
                                                   crossover_rate=CROSSOVER_RATE,
                                                   mutation_rate=MUTATION_RATE,
                                                   tournament_size=TOURNAMENT_SIZE,
                                                   stopping_criteria=IDEAL_COST,
                                                   num_elites=NUM_ELITES)
        print(result)


if __name__ == '__main__':
    algo = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=int, help="Number of instances")
    parser.add_argument("--evaluations", type=int, help="Number of evaluations")
    parser.add_argument("--algorithm", type=int, help="Search algorithm")

    parser.add_argument("--degree", type=int, help="Polynomial degree")

    parser.add_argument("--mu", type=int, help="Number of parents")
    parser.add_argument("--lmbda", type=int, help="Number of offsprings")

    parser.add_argument("--crate", type=float, help="Crossover rate")
    parser.add_argument("--mrate", type=float, help="Mutation rate")

    parser.add_argument("--popsize", type=int, help="Population size")
    parser.add_argument("--tsize", type=int, help="Tournament size")
    parser.add_argument("--nelites", type=int, help="Number of elites")

    args = parser.parse_args()

    if args.instances:
        INSTANCES = args.instances
    if args.evaluations:
        NUM_EVALUATIONS = args.evaluations
    if args.degree:
        DEGREE = args.degree

    match args.algorithm:
        case 0:
            ALGORITHM = algorithm.one_plus_lambda
            if args.lmbda:
                LAMBDA = args.lmbda
        case 1:
            ALGORITHM = algorithm.mu_plus_lambda
            if args.lmbda:
                LAMBDA = args.lmbda
            if args.mu:
                MU = args.mu
            if args.crate:
                CROSSOVER_RATE = args.crate

        case 2:
            ALGORITHM = algorithm.canonical_ea
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

    #print_configuarion()
    #print("")
    run(instances=INSTANCES)
