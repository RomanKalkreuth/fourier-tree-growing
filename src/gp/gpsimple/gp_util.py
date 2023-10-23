# Copyright (C) 2023 -
# Roman Kalkreuth (Roman.Kalkreuth@lip6.fr)
# Computer Lab of Paris 6, Sorbonne Université (Paris, France)

import gp_config as config

__author__ = 'Roman Kalkreuth'
__copyright__ = 'Copyright (C) 2023, Roman Kalkreuth'
__version__ = '1.0'
__email__  = 'Roman.Kalkreuth@lip6.fr'

def get_variables_from_terminals(terminals):
    """

    """
    variables = []
    for terminal in terminals:
        if type(terminal) == str:
            variables.append(terminal)
    return variables

def generate_expression(tree, expression=""):
    """

    """
    symbol = tree.get_symbol()
    expression += symbol

    if tree.symbol in config.FUNCTIONS:
        expression += "("
        if tree.left is not None:
            expression += generate_expression(tree.left)
        if tree.right is not None:
            expression += ", "
            expression += generate_expression(tree.right)
        expression += ")"

    return expression
