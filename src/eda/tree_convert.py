import sys

sys.path.insert(0, '../gp/gpsimple')

import copy

from gp_tree import GPNode
from collections import deque
from queue import Queue


def convert_node_adj_list(tree):
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


def tree_to_list(tree):
    symbols = []
    q = Queue()
    q.put(tree)

    while not q.empty():
        node = q.get()
        symbol = node.symbol

        symbols.append(symbol)

        if node.left is not None:
            q.put(node.left)

        if node.right is not None:
            q.put(node.right)

    return symbols


def list_to_tree(symbols, functions):
    if not symbols:
        return GPNode()

    symbol = symbols.pop(0)
    root = GPNode(symbol=symbol)

    q = Queue()
    q.put(root)

    while not q.empty():
        node = q.get()
        symbol = node.symbol

        if symbol in functions:
            symbol = symbols.pop(0)
            node.left = GPNode(parent=node, symbol=symbol)
            q.put(node.left)

            symbol = symbols.pop(0)
            node.right = GPNode(parent=node, symbol=symbol)
            q.put(node.right)
    return root


def symbols_to_string(symbols, functions):
    size = len(symbols)

    i = 0
    while i < size:
        if symbols[i] in functions:
            symbols[i] = symbols[i].__name__
        else:
            symbols[i] = str(symbols[i])
        i += 1


def symbols_to_type(symbols, functions, terminals):
    size = len(symbols)
    func_map = {}
    terminal_map = {}

    for func in functions:
        func_map[func.__name__] = func

    for term in terminals:
        terminal_map[str(term)] = term

    i = 0
    while i < size:
        symbol = symbols[i]
        if symbol in func_map:
            symbols[i] = func_map[symbol]
        else:
            symbols[i] = terminal_map[symbol]
        i += 1


def validate_structure(symbols, functions):
    q = Queue()

    s = copy.copy(symbols)
    q.put(s.pop(0))

    while not q.empty():
        symbol = q.get()

        if symbol in functions:
            for i in range(2):
                if s:
                    q.put(s.pop(0))
                else:
                    return False
    return True
