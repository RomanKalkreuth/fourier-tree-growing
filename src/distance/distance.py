from Levenshtein import distance
import edist.ted as ted
import src.gp.gpsimple.gp_util as util


def levenshtein_distance(t1, t2):
    expression1 = util.generate_symbolic_expression(t1)
    expression2 = util.generate_symbolic_expression(t2)
    return distance(expression1, expression2)


def tree_edit_distance(t1, t2):
    x_nodes, x_adj = util.convert_list_format(t1)
    y_nodes, y_adj = util.convert_list_format(t2)

    return ted.standard_ted(x_nodes, x_adj, y_nodes, y_adj)


def decomposed_distance(t1, t2):
    std = structural_distance(t1, t2)
    syd = syntactic_distance(t1, t2)

    return std, syd


def structural_distance(t1, t2, d=0):
    if t1 is None and t2 is None:
        return 0

    if type(t1) is not type(t2):
        d = 1

    l1 = None if t1 is None else t1.left
    l2 = None if t2 is None else t2.left

    left = structural_distance(l1, l2)

    r1 = None if t1 is None else t1.right
    r2 = None if t2 is None else t2.right

    right = structural_distance(r1, r2)

    return d + left + right


def syntactic_distance(t1, t2, d=0):
    if t1 is None or t2 is None:
        return 0

    if t1.symbol is not t2.symbol:
        d = 1

    left = syntactic_distance(t1.left, t2.left)
    right = syntactic_distance(t1.right, t2.right)

    return d + left + right
