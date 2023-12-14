from parse_tree import ParseTree

def sample_uniform(min_depth, max_depth, n):
    sample = []

    for i in range(0, n):
        rand_tree = ParseTree()
        rand_tree.init_tree(min_depth, max_depth)

        sample.append(rand_tree)

    return sample

#def expansion():
#def shrinkage()

