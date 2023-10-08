from gp_tree import ParseTree
import gp_config as config
from random import seed

seed()

tree = ParseTree()
tree.init(config.MIN_DEPTH, config.MAX_DEPTH)
tree.print_tree()
size = tree.size()
depth = tree.height()

print(size)
print(depth)

#values = {'x': 1.523}

#fitness = tree.evaluate(values)
#print(fitness)

