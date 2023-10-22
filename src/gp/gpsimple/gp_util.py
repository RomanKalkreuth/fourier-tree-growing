import gp_config as config
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
