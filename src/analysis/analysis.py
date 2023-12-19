import numpy as np

def linear_dependency(vectors):
    M = np.array(vectors)
    return np.linalg.matrix_rank(M @ M.T) != len(vectors)
