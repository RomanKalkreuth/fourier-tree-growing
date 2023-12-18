from sympy import Matrix
import numpy as np

def linear_dependency(vectors):
    n = len(vectors)
    m = stack_vectors(vectors)
    mr, indexes = m.T.rref()

    if len(indexes) == n:
        return False
    else:
        return True

def stack_vectors(vecs):
    m = Matrix(vecs[0])
    for vec in vecs[1:]:
        m = m.row_join(Matrix(vec))
    return m

