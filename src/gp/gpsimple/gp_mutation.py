# Copyright (C) 2023 -
# Roman Kalkreuth (Roman.Kalkreuth@lip6.fr)
# Computer Lab of Paris 6, Sorbonne Université (Paris, France)

import gp_config as config
from random import random

__author__ = 'Roman Kalkreuth'
__copyright__ = 'Copyright (C) 2023, Roman Kalkreuth'
__version__ = '1.0'
__email__  = 'Roman.Kalkreuth@lip6.fr'

def subtree_mutation(tree, mutation_rate):
    """

    """
    if random() < mutation_rate:
        tree.random_tree(grow=True, max_depth=config.SUBTREE_DEPTH)
    elif tree.left:
        tree.left.mutate(mutation_rate=mutation_rate)
    elif tree.right:
        tree.right.mutate(mutation_rate=mutation_rate)