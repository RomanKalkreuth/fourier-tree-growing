from random import random, randint
import numpy as np
import queue
import src.util as util
from src.functions.functions import Mathematical

FUNCTIONS = [Mathematical.add, Mathematical.sub, Mathematical.mul, Mathematical.div]
TERMINALS = ['x', 1.0]
VARIABLES = [terminal for terminal in TERMINALS if type(terminal) == str]

FUNCTION_CLASS = Mathematical

MIN_INIT_TREE_DEPTH = 2
MAX_INIT_TREE_DEPTH = 4

GROW = True

NUM_FUNCTIONS = len(FUNCTIONS)
NUM_TERMINALS = len(TERMINALS)
NUM_VARIABLES = len(VARIABLES)


class ParseTree:
    def __init__(self, symbol=None, left=None, right=None, parent=None):
        self.symbol = symbol
        self.left = left
        self.right = right
        self.parent = parent

    def init_tree(self, min_depth: int, max_depth: int, grow=True):
        rand_depth = randint(min_depth, max_depth)

        if random() < 0.5:
            self.random_tree(grow=grow, min_depth=min_depth, max_depth=rand_depth)
        else:
            self.random_tree(grow=False, min_depth=min_depth, max_depth=rand_depth)

    def random_tree(self, grow: bool, min_depth: int, max_depth: int, depth: int = 0):
        if depth >= max_depth - 1:
            self.symbol = TERMINALS[randint(0, NUM_TERMINALS - 1)]
        else:
            if grow is True:
                if random() < 0.5 or depth < min_depth:
                    self.symbol = FUNCTIONS[randint(0, NUM_FUNCTIONS - 1)]
                else:
                    self.symbol = TERMINALS[randint(0, NUM_TERMINALS - 1)]
            else:
                self.symbol = FUNCTIONS[randint(0, NUM_FUNCTIONS - 1)]

            if self.symbol in FUNCTIONS:

                arity = FUNCTION_CLASS.arity(self.symbol)

                self.left = ParseTree()
                self.left.parent = self
                self.left.random_tree(grow, min_depth, max_depth, depth=depth + 1)

                if arity > 1:
                    self.right = ParseTree()
                    self.right.parent = self
                    self.right.random_tree(grow, min_depth, max_depth, depth=depth + 1)
                else:
                    self.right = None

    def evaluate(self, input: np.array):
        if self.symbol in FUNCTIONS:
            arity = FUNCTION_CLASS.arity(self.symbol)
            if arity == 1:
                return self.symbol(self.left.evaluate(input))
            else:
                return self.symbol(self.left.evaluate(input), self.right.evaluate(input))
        elif self.symbol in VARIABLES:
            if NUM_VARIABLES > 1:
                input_index = VARIABLES.index(self.symbol)
                return input[input_index]
            else:
                return input
        else:
            return self.symbol

    def size(self, left: int = 0, right: int = 0) -> int:
        if self.symbol is None:
            return 0

        if self.left is not None:
            left = self.left.size()
        if self.right is not None:
            right = self.right.size()

        return 1 + left + right

    def depth(self, left: int = 0, right: int = 0):
        if self.left is not None:
            left = self.left.depth()
        if self.right is not None:
            right = self.right.depth()

        return max(left + 1, right + 1)

    def get_symbol(self) -> str:
        if self.symbol in FUNCTIONS:
            return self.symbol.__name__
        else:
            return str(self.symbol)

    def subtree_at(self, node_num: int):
        q = queue.Queue()
        q.put(self)
        count = 0

        while not q.empty():
            subtree = q.get()

            if count == node_num:
                return subtree

            count += 1

            if subtree.left is not None:
                q.put(subtree.left)
            if subtree.right is not None:
                q.put(subtree.right)

    def subtree(self, node_num: int):
        subtree = self.subtree_at(node_num)

        if subtree is None:
            raise RuntimeError('Subtree not found')

        subtree.parent = None

        return subtree.clone()

    def replace_subtree(self, replacement: object, node: int):
        subtree = self.subtree_at(node)
        subtree.symbol = replacement.symbol
        subtree.left = replacement.left
        subtree.right = replacement.right
        
    def variate(self, mutation_rate: float, subtree: object = None, max_depth: int = 6):
        if subtree is None:
            subtree = self
        if random() < mutation_rate:
            subtree.random_tree(grow=True, min_depth=1, max_depth=max_depth)
        elif self.left is not None:
            self.variate(subtree=subtree.left, mutation_rate=mutation_rate)
        elif self.right is not None:
            self.variate(subtree=subtree.right, mutation_rate=mutation_rate)

    def clone(self, root: object = None, tree_clone: object = None) -> object:
        if tree_clone is None:
            tree_clone = ParseTree(symbol=self.symbol)
        if root is None:
            root = self

        if root.left is not None:
            tree_clone.left = ParseTree(symbol=root.left.symbol, parent=root)
            self.clone(root=root.left, tree_clone=tree_clone.left)
        if root.right is not None:
            tree_clone.right = ParseTree(symbol=root.right.symbol, parent=root)
            self.clone(root=root.right, tree_clone=tree_clone.right)

        return tree_clone

    def __str__(self):
        return util.generate_symbolic_expression(self, FUNCTIONS)
