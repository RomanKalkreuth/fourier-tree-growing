ELBOW = "└──"
PIPE = "│  "
TEE = "├──"
BLANK = "   "


def print_vertically(tree, term="", left=False, right=False):
    """

    """

    if tree.get_symbol is None:
        return

    prefix = ""

    if right or tree.parent is None:
        prefix += ELBOW
    elif left:
        prefix += TEE

    print("%s%s%s" % (term, prefix, tree.get_symbol()))

    if tree.left is not None:
        print_vertically(tree.left, term + (PIPE if left else BLANK), left=True, right=False)
    if tree.right is not None:
        print_vertically(tree.right, term + (PIPE if left else BLANK), left=False, right=True)
