import gp_config as config
from random import random

def subtree_mutation(tree):
    """

    """
    if random() < config.MUTATION_RATE:
        tree.generate_random_tree(grow=True, max_depth=config.SUBTREE_DEPTH)
    elif tree.left:
        tree.left.mutate()
    elif tree.right:
        tree.right.mutate()