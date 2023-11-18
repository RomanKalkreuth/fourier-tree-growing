# Copyright (C) 2023 -
# Roman Kalkreuth (Roman.Kalkreuth@lip6.fr)
# Computer Lab of Paris 6, Sorbonne Université (Paris, France)

import gp_config as config
import queue
from collections import deque


__author__ = 'Roman Kalkreuth'
__copyright__ = 'Copyright (C) 2023, Roman Kalkreuth'
__version__ = '1.0'
__email__ = 'Roman.Kalkreuth@lip6.fr'


def get_variables_from_terminals(terminals):
    variables = []
    for terminal in terminals:
        if type(terminal) == str:
            variables.append(terminal)
    return variables


def generate_symbolic_expression(tree, expression=""):
    symbol = tree.get_symbol()
    expression += symbol

    if tree.symbol in config.FUNCTIONS:
        expression += "("
        if tree.left is not None:
            expression += generate_symbolic_expression(tree.left)
        if tree.right is not None:
            expression += ", "
            expression += generate_symbolic_expression(tree.right)
        expression += ")"

    return expression


def generate_bracket_notation(tree, expression="", init=True):
    if init:
        expression += "{"
    symbol = tree.get_symbol()
    if tree.symbol in config.FUNCTIONS:
        expression += symbol + "{"
        if tree.left is not None:
            expression += generate_bracket_notation(tree.left, init=False)
        if tree.right is not None:
            expression += generate_bracket_notation(tree.right, init=False)
        expression += "}"
    else:
        expression += "{" + symbol + "}"
    return expression


def pre_order_traversal(node, index=1, node_list=None):
    if node_list is None:
        node_list = []

    symbol = node.get_symbol()
    node_list.append(symbol)

    if node.right is not None:
        pre_order_traversal(node.right, index, node_list)
    if node.left is not None:
        pre_order_traversal(node.left, index, node_list)
    return node_list


def generate_node_list(tree):
    node_list = []
    node_stack = [tree]
    node_dict = {}
    count = 0
    while node_stack:
        node = node_stack.pop()
        symbol = node.get_symbol()
        node_list.append(symbol)

        if node.parent is not None:
            count += 1

        if node not in node_dict:
            node_dict[node] = count

        if node.left is not None:
            node_stack.append(node.left)
        if node.right is not None:
            node_stack.append(node.right)

    return node_list, node_dict


def generate_adjacency_dict(tree):
    queue = deque()
    queue.append(tree)
    adj_dict = {}

    while queue:
        node = queue.pop()

        if node not in adj_dict:
            adj_dict[node] = []

        if node.left is not None:
            queue.append(node.left)
            adj_dict[node].append(node.left)
        if node.right is not None:
            queue.append(node.right)
            adj_dict[node].append(node.right)
    return adj_dict


def convert_list_format(tree):
    adj_dict = generate_adjacency_dict(tree)
    node_list, node_dict = generate_node_list(tree)
    adj_list = []

    for key in adj_dict.keys():
        adj_nodes = adj_dict[key]
        temp = []
        for node in adj_nodes:
            temp.append(node_dict[node])
        adj_list.append(temp[::-1])
    return node_list, adj_list


def validate_tree(tree, functions: list, terminals: list) -> int:
    """
    Cases of invalidity:
        - a terminal node has edges
        - a function node has no edges
        - a function node has only one edge
        - a node is referenced (seen) twice (cycle)
    """

    q = queue.Queue()
    err = 0
    visited = {}
    q.put(tree)

    while not q.empty():
        node = q.get()

        if node.symbol not in functions:
            if node.symbol not in terminals:
                err += 1

        if node.symbol in functions:
            if node.left is None:
                err += 1
            if node.right is None:
                err += 1

        if node.symbol in terminals:
            if node.left is not None:
                err += 1
            if node.right is not None:
                err += 1

        if node in visited:
            err += 1
        else:
            visited[node] = True

        if node.left is not None:
            q.put(node.left)
        if node.right is not None:
            q.put(node.right)

    return err
