from Levenshtein import distance
import edist
import src.gp.gpsimple.gp_util as util

def levenshtein_distance(tree1, tree2):
    expression1 = util.generate_expression(tree1)
    expression2 = util.generate_expression(tree2)
    return distance(expression1,expression2)

#def tree_edit_distance(tree1, tree2):

