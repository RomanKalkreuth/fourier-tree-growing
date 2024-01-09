import random
from src.representation.parse_tree import ParseTree

def probabilistic_subtree_mutation(tree: object, mutation_rate: float, max_depth: int = 6):
    if random.random() < mutation_rate:
        tree.random_tree(grow=True, min_depth=1, max_depth=max_depth)
    elif tree.left is not None:
        probabilistic_subtree_mutation(tree=tree.left, mutation_rate=mutation_rate, max_depth=max_depth)
    elif tree.right is not None:
        probabilistic_subtree_mutation(tree=tree.right, mutation_rate=mutation_rate, max_depth=max_depth)

def uniform_subtree_mutation(tree: object, max_depth: int = 6):
    mutation_point = random.randint(0, tree.size() - 1)
    subtree = ParseTree()
    subtree.init_tree(1, max_depth)
    tree.replace_subtree(subtree, mutation_point)


def subtree_crossover(ptree1, ptree2, crossover_rate):
    if random.random() < crossover_rate:
        if ptree1.size() <= 1 or ptree2.size() <= 1:
            return ptree1, ptree2

        crossover_point1 = random.randint(1, ptree1.size() - 1)
        crossover_point2 = random.randint(1, ptree2.size() - 1)

        otree1 = ptree1.clone()
        otree2 = ptree2.clone()

        subtree1 = otree1.subtree(crossover_point1)
        subtree2 = otree2.subtree(crossover_point2)

        otree1.replace_subtree(subtree2, crossover_point1)
        otree2.replace_subtree(subtree1, crossover_point2)

        return otree1, otree2
    else:
        return ptree1, ptree2

