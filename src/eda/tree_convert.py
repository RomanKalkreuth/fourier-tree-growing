import sys

sys.path.insert(0, '../gp/gpsimple')

from gp_tree import GPNode
from collections import deque
from queue import Queue
import copy


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


def tree_to_list(tree):
    tree_list = []
    q = Queue()
    q.put(tree)

    while not q.empty():
        node = q.get()
        symbol = node.symbol
        tree_list.append(symbol)

        if node.left is not None:
            q.put(node.left)

        if node.right is not None:
            q.put(node.right)

    return tree_list


def list_to_tree(tree_list, functions):
    symbol = tree_list.pop(0)
    root = GPNode(symbol=symbol)

    q = Queue()
    q.put(root)

    while not q.empty():
        node = q.get()
        symbol = node.symbol

        if symbol in functions:
            symbol = tree_list.pop(0)
            node.left = GPNode(parent=node, symbol=symbol)
            q.put(node.left)

            symbol = tree_list.pop(0)
            node.right = GPNode(parent=node, symbol=symbol)
            q.put(node.right)
    return root
