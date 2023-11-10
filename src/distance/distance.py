from Levenshtein import distance
import edist.ted as ted
import src.gp.gpsimple.gp_util as util

def levenshtein_distance(tree1, tree2):
    expression1 = util.generate_symbolic_expression(tree1)
    expression2 = util.generate_symbolic_expression(tree2)
    return distance(expression1, expression2)


def tree_edit_distance(tree1, tree2):
    """

    """

    x_nodes, x_adj = util.convert_list_format(tree1)
    y_nodes, y_adj = util.convert_list_format(tree2)

    return ted.standard_ted(x_nodes, x_adj, y_nodes, y_adj)