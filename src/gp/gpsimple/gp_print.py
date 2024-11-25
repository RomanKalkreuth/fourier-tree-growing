# Copyright (C) 2023 -
# Roman Kalkreuth (Roman.Kalkreuth@lip6.fr)
# Computer Lab of Paris 6, Sorbonne Université (Paris, France)

__author__ = 'Roman Kalkreuth'
__copyright__ = 'Copyright (C) 2023, Roman Kalkreuth'
__version__ = '1.0'
__email__ = 'Roman.Kalkreuth@lip6.fr'

ELBOW = "└──"
PIPE = "│  "
TEE = "├──"
BLANK = "   "


def print_tree_vertically(tree, term="", left=False, right=False):

    if tree.get_symbol is None:
        return

    prefix = ""

    if right or tree.parent is None:
        prefix += ELBOW
    elif left and tree.parent.right is None:
        prefix += ELBOW
    elif left:
        prefix += TEE

    print("%s%s%s" % (term, prefix, tree.get_symbol()))

    appendix = (PIPE if left and tree.parent.right is not None else BLANK)

    if tree.left is not None:
        print_tree_vertically(tree.left, term + appendix, left=True, right=False)
    if tree.right is not None:
        print_tree_vertically(tree.right, term + appendix, left=False, right=True)


def print_population(population):
    for index, (ind, fitness) in enumerate(population):
        print("Individual #" + str(index) + " - Fitness: " + str(fitness))
