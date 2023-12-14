
ELBOW = "└──"
PIPE = "│  "
TEE = "├──"
BLANK = "   "

def get_variables_from_terminals(terminals):
    variables = []
    for terminal in terminals:
        if type(terminal) == str:
            variables.append(terminal)
    return variables

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

def generate_symbolic_expression(tree, functions, expression=""):
    symbol = tree.get_symbol()
    expression += symbol

    if tree.symbol in functions:
        expression += "("
        if tree.left is not None:
            expression += generate_symbolic_expression(tree.left, functions)
        if tree.right is not None:
            expression += ", "
            expression += generate_symbolic_expression(tree.right, functions)
        expression += ")"

    return expression