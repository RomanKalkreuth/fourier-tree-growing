import gp_config as config
from random import random, randint


class ParseTree:
    """

    """

    def __init__(self, data=None, left=None, right=None):
        """

        """
        self.data = data
        self.left = left
        self.right = right

    def annotation(self):
        """

        """
        if self.data in config.FUNCTIONS:
            return self.data.__name__
        else:
            return str(self.data)

    def evaluate(self, variables):
        """

        """
        if self.data in config.FUNCTIONS:
            return self.data(self.left.evaluate(variables), self.right.evaluate(variables))
        elif self.data in variables:
            return variables[self.data]
        else:
            return self.data

    def print_tree(self, term="", level=0):
        print("%s%s" % (term, self.annotation()))
        if self.left is not None:
            self.left.print_tree(term + "   ")
        if self.right is not None:
            self.right.print_tree(term + "   ")

    def init(self, min_depth, max_depth):
        """

        """
        depth = randint(min_depth, max_depth)

        if random() < 0.5:
            self.gen_random_tree(grow=True, max_depth=depth, depth=0)
        else:
            self.gen_random_tree(grow=False, max_depth=depth, depth=0)

    def gen_random_tree(self, grow, max_depth, depth=0):
        """

        """
        if depth >= max_depth:
            self.data = config.TERMINALS[randint(0, config.NUM_TERMINALS - 1)]
        else:
            if grow is True:
                if random() < 0.5 or depth < config.MIN_DEPTH:
                    self.data = config.FUNCTIONS[randint(0, config.NUM_FUNCTIONS - 1)]
                else:
                    self.data = config.TERMINALS[randint(0, config.NUM_TERMINALS - 1)]
            else:
                self.data = config.FUNCTIONS[randint(0, config.NUM_FUNCTIONS - 1)]

            if self.data in config.FUNCTIONS:
                self.left = ParseTree()
                self.left.gen_random_tree(grow, max_depth, depth =depth + 1)

                self.right = ParseTree()
                self.right.gen_random_tree(grow, max_depth, depth =depth + 1)

    def mutate(self):
        """

        """
        self.subtree_mutation()

    def subtree_mutation(self):
        """

        """
        if random() < config.MUTATION_RATE:
            self.gen_random_tree(grow=True, max_depth=config.SUBTREE_DEPTH)
        elif self.left: self.left.subtree_mutation()
        elif self.right: self.right.subtree_mutation()
