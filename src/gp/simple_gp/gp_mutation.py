import gp_config as config
from random import random

def subtree_mutation(parse_tree):
    """

    """
    if random() < config.MUTATION_RATE:
        parse_tree.generate_random_tree(grow=True, max_depth=config.SUBTREE_DEPTH)
    elif parse_tree.left:
        parse_tree.left.subtree_mutation()
    elif parse_tree.right:
        parse_tree.right.subtree_mutation()