def absolute_error(self, individual, dataset):
    """

    """
    return sum([abs(individual.absolute_error(ds[0]) - ds[1]) for ds in dataset])

def mean_squared_error(self, individual, dataset):
    """

    """

def root_mean_squared_error(self, individual, dataset):
    """

    """


def is_better(fitness1, fitness2, minimizing_fitness=True, strict=True):
    if minimizing_fitness:
        if strict:
            if fitness1 < fitness2: return True
        else:
            if fitness1 <= fitness2: return True
    else:
        if strict:
            if fitness1 > fitness2: return True
        else:
            if fitness1 >= fitness2: return True
    return False