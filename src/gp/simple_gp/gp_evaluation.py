def evaluate(self, individual, dataset):
    return sum([abs(individual.evaluate(ds[0]) - ds[1]) for ds in dataset])


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