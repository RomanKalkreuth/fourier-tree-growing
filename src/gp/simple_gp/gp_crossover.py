from random import random, randint


def subtree_crossover(tree1, tree2):
    """

    """

    crossover_point1 = randint(0, tree1.size())
    crossover_point2 = randint(0, tree2.size())

    subtree1 = tree1.subtree(crossover_point1)
    subtree2 = tree2.subtree(crossover_point2)

    tree1.replace(subtree2, crossover_point1)
    tree2.replace(subtree1, crossover_point2)

    return tree1, tree2
