import random
import queue
import config

def random_symbol(symbols):
    return symbols[random.randint(0, len(symbols) - 1)]

def single_node_variation(tree):
    num_nodes = tree.size()
    rand_node = random.randint(0, num_nodes - 1)

    q = queue.Queue()
    q.put(tree)

    count = 0

    while not q.empty():
        node = q.get()

        if count == rand_node:
            if node.symbol in config.FUNCTIONS:
                symbol_arity = config.FUNCTION_CLASS.arity(node.symbol)

                rand_symbol = random_symbol(config.FUNCTIONS)
                rand_arity = config.FUNCTION_CLASS.arity(rand_symbol)

                while not rand_arity == symbol_arity:
                    rand_symbol = random_symbol(config.FUNCTIONS)
                    rand_arity = config.FUNCTION_CLASS.arity(rand_symbol)
            else:
                rand_symbol = random_symbol(config.TERMINALS)

            transition = (node.symbol, rand_symbol)
            node.symbol = rand_symbol
            return transition

        count += 1

        if node.left is not None:
            q.put(node.left)
        if node.right is not None:
            q.put(node.right)