# Copyright (C) 2023 -
# Roman Kalkreuth (Roman.Kalkreuth@lip6.fr)
# Computer Lab of Paris 6, Sorbonne Université (Paris, France)

from random import random, randint

__author__ = 'Roman Kalkreuth'
__copyright__ = 'Copyright (C) 2023, Roman Kalkreuth'
__version__ = '1.0'
__email__  = 'Roman.Kalkreuth@lip6.fr'

def subtree_crossover(tree1, tree2):
    """

    """

    crossover_point1 = randint(0, tree1.size())
    crossover_point2 = randint(0, tree2.size())

    subtree1 = tree1.subtree(crossover_point1)
    subtree2 = tree2.subtree(crossover_point2)

    tree1.replace_subtree(subtree2, crossover_point1)
    tree2.replace_subtree(subtree1, crossover_point2)

    return tree1, tree2
