import gp_config as config
import gp_mutation as mutation
import util.tree_print as printer
from random import random, randint
import queue


class ParseTree:
    """

    """

    def __init__(self, data=None, left=None, right=None, parent=None):
        """

        """
        self.symbol = data
        self.left = left
        self.right = right
        self.parent = parent

    def get_symbol(self):
        """

        """
        if self.symbol in config.FUNCTIONS:
            return self.symbol.__name__
        else:
            return str(self.symbol)

    def evaluate(self, variables):
        """

        """
        if self.symbol in config.FUNCTIONS:
            return self.symbol(self.left.absolute_error(variables), self.right.absolute_error(variables))
        elif self.symbol in variables:
            return variables[self.symbol]
        else:
            return self.symbol

    def print_tree(self, level=0):
        printer.print_vertically(self)

    def init(self, min_depth, max_depth):
        """

        """
        depth = randint(min_depth, max_depth)

        if random() < 0.5:
            self.generate_random_tree(grow=True, max_depth=depth, depth=0)
        else:
            self.generate_random_tree(grow=False, max_depth=depth, depth=0)

    def generate_random_tree(self, grow, max_depth, depth=0):
        """

        """
        if depth >= max_depth:
            self.symbol = config.TERMINALS[randint(0, config.NUM_TERMINALS - 1)]
        else:
            if grow is True:
                if random() < 0.5 or depth < config.MIN_DEPTH:
                    self.symbol = config.FUNCTIONS[randint(0, config.NUM_FUNCTIONS - 1)]
                else:
                    self.symbol = config.TERMINALS[randint(0, config.NUM_TERMINALS - 1)]
            else:
                self.symbol = config.FUNCTIONS[randint(0, config.NUM_FUNCTIONS - 1)]

            if self.symbol in config.FUNCTIONS:
                self.left = ParseTree()
                self.left.parent = self.symbol
                self.left.generate_random_tree(grow, max_depth, depth=depth + 1)

                self.right = ParseTree()
                self.right.parent = self.symbol
                self.right.generate_random_tree(grow, max_depth, depth=depth + 1)

    def size(self, left=0, right=0):
        """

        """
        if self.symbol is None:
            return 0

        if self.left is not None:
            left = self.left.size()
        if self.right is not None:
            right = self.right.size()

        return 1 + left + right

    def height(self, left=0, right=0):
        """

        """
        if self.left is not None:
            left = self.left.height()
        if self.right is not None:
            right = self.right.height()

        return max(left + 1, right + 1)

    def mutate(self):
        """

        """
        mutation.subtree_mutation(self)

    def level_order_search(self, node):
        q = queue.Queue()
        q.put((self, 0))

        while not q.empty():
            item = q.get()
            subtree = item[0]
            count = item[1]

            if count == node:
                return subtree

            if subtree.left is not None:
                count += 1
                q.put((subtree.left, count))
            if subtree.right is not None:
                count += 1
                q.put((subtree.right, count))

    def subtree(self, node):
        """

        """
        subtree = self.level_order_search(node)
        subtree.parent=None
        return subtree


    def replace(self, replacement_tree, node):
        subtree = self.level_order_search(node)
        subtree.symbol = replacement_tree.symbol
        subtree.left = replacement_tree.left
        subtree.right = replacement_tree.right



class GPNode:
    """

    """

    def __init__(self, symbol=None, left=None, right=None, parent=None):
        """

        """
        self.symbol = symbol
        self.left = left
        self.right = right
        self.parent = parent

    def get_symbol(self):
        """

        """
        if self.symbol in config.FUNCTIONS:
            return self.symbol.__name__
        else:
            return str(self.symbol)
