# Copyright (C) 2023 -
# Roman Kalkreuth (Roman.Kalkreuth@lip6.fr)
# Computer Lab of Paris 6, Sorbonne Université (Paris, France)

from random import random

__author__ = 'Roman Kalkreuth'
__copyright__ = 'Copyright (C) 2023, Roman Kalkreuth'
__version__ = '1.0'
__email__ = 'Roman.Kalkreuth@lip6.fr'

import random
import queue
import gp_config as config


def subtree_mutation(tree: object, mutation_rate: float, max_depth: int = 6):
    """
    Standard subtree mutation operator which is commonly used in GP.

    TODO: Revise traversal (non-recursive) and consider left - right
        branch choice per chance
    """
    if random.random() < mutation_rate:
        tree.random_tree(grow=True, min_depth=1, max_depth=max_depth)
    elif tree.left is not None:
        subtree_mutation(tree=tree.left, mutation_rate=mutation_rate, max_depth=max_depth)
    elif tree.right is not None:
        subtree_mutation(tree=tree.right, mutation_rate=mutation_rate, max_depth=max_depth)


def node_mutation(tree: object, n: int):
    size = tree.size()
    if size < n:
        n = size

    for i in range(n):
        single_node_mutation(tree)


def single_node_mutation(tree: object):
    num_nodes = tree.size()
    rand_node = random.randint(0, num_nodes)

    q = queue.Queue()
    q.put(tree)

    count = 0

    while not q.empty():
        node = q.get()

        if count == rand_node:
            if node.symbol in config.FUNCTIONS:
                symbols = config.FUNCTIONS
            else:
                symbols = config.TERMINALS

            rand_symbol = random_symbol(symbols)
            while rand_symbol == node.symbol:
                rand_symbol = random_symbol(symbols)

            node.symbol = rand_symbol
            return

        count += 1

        if node.left is not None:
            q.put(node.left)
        if node.right is not None:
            q.put(node.right)


def random_symbol(symbols):
    return symbols[random.randint(0, len(symbols) - 1)]
