import random

def subtree_variation(tree: object, mutation_rate: float, max_depth: int = 6):
    if random.random() < mutation_rate:
        tree.random_tree(grow=True, min_depth=1, max_depth=max_depth)
    elif tree.left is not None:
        subtree_variation(tree=tree.left, mutation_rate=mutation_rate, max_depth=max_depth)
    elif tree.right is not None:
        subtree_variation(tree=tree.right, mutation_rate=mutation_rate, max_depth=max_depth)