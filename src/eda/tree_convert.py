from collections import deque

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
