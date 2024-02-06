from random import random, randint
import config
import numpy as np
import util

class ParseTree:
    def __init__(self, symbol=None, left=None, right=None, parent=None):
        self.symbol = symbol
        self.left = left
        self.right = right
        self.parent = parent

    def init_tree(self, min_depth: int, max_depth: int, grow=True):
        rand_depth = myrandom.RND.randint(min_depth, max_depth)

        if random() < 0.5:
            self.random_tree(grow=grow, min_depth=min_depth, max_depth=rand_depth)
        else:
            self.random_tree(grow=False, min_depth=min_depth, max_depth=rand_depth)

    def random_tree(self, grow: bool, min_depth: int, max_depth: int, depth: int = 0):
        if depth >= max_depth - 1:
            self.symbol = config.TERMINALS[randint(0, config.NUM_TERMINALS - 1)]
        else:
            if grow is True:
                if random() < 0.5 or depth < min_depth:
                    self.symbol = config.FUNCTIONS[randint(0, config.NUM_FUNCTIONS - 1)]
                else:
                    self.symbol = config.TERMINALS[randint(0, config.NUM_TERMINALS - 1)]
            else:
                self.symbol = config.FUNCTIONS[randint(0, config.NUM_FUNCTIONS - 1)]

            if self.symbol in config.FUNCTIONS:

                arity = config.FUNCTION_CLASS.arity(self.symbol)

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
        if self.symbol in config.FUNCTIONS:
            arity = config.FUNCTION_CLASS.arity(self.symbol)
            if arity == 1:
                return self.symbol(self.left.evaluate(input))
            else:
                return self.symbol(self.left.evaluate(input), self.right.evaluate(input))
        elif self.symbol in config.VARIABLES:
            if config.NUM_VARIABLES > 1:
                input_index = config.VARIABLES.index(self.symbol)
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
            left = self.left.dep()
        if self.right is not None:
            right = self.right.dep()

        return max(left + 1, right + 1)

    def get_symbol(self) -> str:
        if self.symbol in config.FUNCTIONS:
            return self.symbol.__name__
        else:
            return str(self.symbol)

    def __str__(self):
        return util.generate_symbolic_expression(self)
