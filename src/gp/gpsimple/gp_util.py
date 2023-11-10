# Copyright (C) 2023 -
# Roman Kalkreuth (Roman.Kalkreuth@lip6.fr)
# Computer Lab of Paris 6, Sorbonne Université (Paris, France)

import gp_config as config
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
    while len(node_stack) > 0:
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

    while len(queue) > 0:
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