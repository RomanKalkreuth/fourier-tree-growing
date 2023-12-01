# Copyright (C) 2023 -
# Roman Kalkreuth (Roman.Kalkreuth@lip6.fr)
# Computer Lab of Paris 6, Sorbonne Université (Paris, France)
import numpy as np

import gp_config as config
import gp_mutation as mutation
import gp_print as printer
import gp_util as util
from gp_simple import GPSimple
from random import random, randint
import queue

__author__ = 'Roman Kalkreuth'
__copyright__ = 'Copyright (C) 2023, Roman Kalkreuth'
__version__ = '1.0'
__email__ = 'Roman.Kalkreuth@lip6.fr'


class GPNode:
    """
    Represents a binary parse tree that is annotated with a symbol. Besides the symbol itself,
    it also stores the references to the (left and right) child nodes and the parent node (if existing).

    A node where is parent node reference is None is the root. If the left and right references are None, the
    node is a leaf.

    The class provides the basic functionality that is commonly used for the evaluation and variation of
    parse tree's in genetic programming (e.g. random initialization,
    subtree selection and replacement, interpretation, ...)
    """

    def __init__(self, symbol=None, left=None, right=None, parent=None):
        self.symbol = symbol
        self.left = left
        self.right = right
        self.parent = parent

    def init_tree(self, min_depth: int, max_depth: int, grow=True):
        """
        Ramped Half-n-Half (RHH) GP tree initialization method.

        To initialize a tree, RHH selects either the grow or full method with 50 percent probability.

        :param min_depth: minimum depth of the tree to be constructed
        :type min_depth: int
        :param max_depth: maximum depth of the tree to be constructed
        :type max_depth: int
        """
        rand_depth = randint(min_depth, max_depth)

        if random() < 0.5:
            self.random_tree(grow=grow, min_depth=min_depth, max_depth=rand_depth)
        else:
            self.random_tree(grow=False, min_depth=min_depth, max_depth=rand_depth)

    def random_tree(self, grow: bool, min_depth: int, max_depth: int, depth: int = 0):
        """
        Recursively samples a random tree either with the GROW or with the FULL method.

        When FULL is used nodes are selected from the function set until the maximum tree depth is reached.

        In contrast, GROW allows the selection of nodes from the function and terminal set
        until the depth limit is reached.


        :param grow: determines whether grow is used
        :type grow: bool
        :param min_depth: minimum depth of the tree
        :type min_depth: int
        :param max_depth: maximum depth of the tree
        :type max_depth: int
        :param depth: current depth of the tree sample
        :type depth: int
        """
        if depth >= max_depth - 1:
            self.symbol = GPSimple.config.terminals[randint(0, GPSimple.config.num_terminals - 1)]
        else:
            if grow is True:
                if random() < 0.5 or depth < min_depth:
                    self.symbol = GPSimple.config.functions[randint(0, GPSimple.config.num_functions - 1)]
                else:
                    self.symbol = GPSimple.config.terminals[randint(0, GPSimple.config.num_terminals - 1)]
            else:
                self.symbol = GPSimple.config.functions[randint(0, GPSimple.config.num_functions - 1)]

            if self.symbol in GPSimple.config.functions:

                arity = GPSimple.config.function_class.arity(self.symbol)

                self.left = GPNode()
                self.left.parent = self
                self.left.random_tree(grow, min_depth, max_depth, depth=depth + 1)

                if arity > 1:
                    self.right = GPNode()
                    self.right.parent = self
                    self.right.random_tree(grow, min_depth, max_depth, depth=depth + 1)
                else:
                    self.right = None


    def evaluate(self, input: np.array) -> object:
        """
        Provides a simple GP parse tree interpreter.

        Distinguishes between the type of the respective symbol (function, variable or constant):

        - symbol is a function:
            The respective funtion is called if the symbol is part of the function set.

        - symbol is a variable:
            The associated data value is returned (from the respective data set that is used for training)

        - symbol is a constant:
            The value of the constant is returned from the terminal set.

        :param input: data batch that contains the values of the used variables
        :type input: list or np.array

        :return: result of the function call, value of the respective variable or constant
        """
        if self.symbol in GPSimple.config.functions:
            arity = GPSimple.config.function_class.arity(self.symbol)
            if arity == 1:
                return self.symbol(self.left.evaluate(input))
            else:
                return self.symbol(self.left.evaluate(input), self.right.evaluate(input))
        elif self.symbol in GPSimple.config.variables:
            if GPSimple.config.num_variables > 1:
                input_index = GPSimple.config.variables.index(self.symbol)
                return input[input_index]
            else:
                return input
        else:
            return self.symbol

    def get_symbol(self) -> str:
        """
        Returns the name of the symbol.

        :return: symbol as string
        """
        if self.symbol in config.FUNCTIONS:
            return self.symbol.__name__
        else:
            return str(self.symbol)

    def size(self, left: int = 0, right: int = 0) -> int:
        """
        Recursively determines the size of a tree which is defined as the number of nodes.

        :param left: node counter for the left branch
        :type left: int
        :param right: node counter for the right branch
        :param right: int
        :return: size
        :rtype int
        """
        if self.symbol is None:
            return 0

        if self.left is not None:
            left = self.left.size()
        if self.right is not None:
            right = self.right.size()

        return 1 + left + right

    def height(self, left: int = 0, right: int = 0):
        """
        Recursively determines the maximum depth of the tree's branches.

        :param left: maximum depth of the left branch
        :type left: int
        :param right: maximum depth of the right branch
        :param right: int
        :return: maximum depth
        :rtype: int
        """
        if self.left is not None:
            left = self.left.height()
        if self.right is not None:
            right = self.right.height()

        return max(left + 1, right + 1)

    def mutate(self, mutation_rate: float, subtree_depth: int = 6):
        """
        Mutates the tree with subtree mutation.

        :param mutation_rate: rate which determines the strength of the mutation
                              and is passed on to the mutation operator
        :type mutation_rate float
        """
        mutation.subtree_mutation(self, mutation_rate=mutation_rate, max_depth=subtree_depth)

    def subtree_at(self, node_num: int):
        """
        Searches the subtree at a specified node number.
        The tree is traversed by using breadth first search (BFS).

        :param node_num: node number at which the subtree is returned
        :type node_num: int
        :return: subtree at specified node
        :rtype GPNode
        """
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
        """
        Returns a clone of the subtree at a specified node number.

        :param node_num: node number at which the subtree is cloned and returned
        :type node_num: int
        :return: subtree at specified node
        """
        subtree = self.subtree_at(node_num)

        if subtree is None:
            raise RuntimeError('Subtree not found')

        subtree.parent = None

        return subtree.clone()

    def replace_subtree(self, replacement: object, node: int):
        """
        Replaces a given subtree at a specified node index.

        :param replacement: subtree used as replacement
        :type replacement GPNode
        :param node: node index at which the subtree is placed
        :type int
        """
        subtree = self.subtree_at(node)
        subtree.symbol = replacement.symbol
        subtree.left = replacement.left
        subtree.right = replacement.right

    def is_root(self) -> bool:
        """
        Returns if the current node is the root of the tree.
        :return root status of node
        :rtype bool
        """
        return self.parent is None

    def clone(self, root: object = None, tree_clone: object = None) -> object:
        """
        Clones the tree and returns it afterwards.
        The tree is traversed recursively.

        :param root: current root node of the traversal
        :type root: GPNode
        :param tree_clone: tree clone object to which the nodes
                           of the original tree are attached
        :type tree_clone: GPNode
        :return: clone of the tree
        :rtype GPNode
        """
        if tree_clone is None:
            tree_clone = GPNode(symbol=self.symbol)
        if root is None:
            root = self

        if root.left is not None:
            tree_clone.left = GPNode(symbol=root.left.symbol, parent=root)
            self.clone(root=root.left, tree_clone=tree_clone.left)
        if root.right is not None:
            tree_clone.right = GPNode(symbol=root.right.symbol, parent=root)
            self.clone(root=root.right, tree_clone=tree_clone.right)

        return tree_clone

    def print_tree(self):
        """
        Prints the tree in human-readable vertical form.
        """
        printer.print_tree_vertically(self)

    def print_expression(self):
        """
        Prints the generated symbolic expression of the tree.
        """
        expression = util.generate_symbolic_expression(self)
        print(expression)
