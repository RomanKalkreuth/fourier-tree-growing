import gp_config as config
import gp_fitness as fitness
from gp_tree import GPNode

def one_plus_lambda(num_generations, lmbda, ideal_fitness, problem, minimizing_fitness=True, strict=True):
    parent = GPNode()
    num_fitness_eval = 0

    prediction = problem.evaluate(parent)
    actual = problem.ground_truth
    best_fitness = fitness.calculate_fitness(actual, prediction, method="abs")

    for gen in range(num_generations):
        for i in range(lmbda):
            offspring = parent
            offspring.mutate()
            offspring_fitness = fitness.calculate_fitness()

            if fitness.is_better(offspring_fitness, best_fitness, minimizing_fitness=minimizing_fitness,
                                 strict=strict):
                best_fitness = offspring_fitness
                parent = offspring

                if fitness.is_ideal(fitness, ideal_fitness, minimizing_fitness=minimizing_fitness):
                    break

    return best_fitness, num_fitness_eval