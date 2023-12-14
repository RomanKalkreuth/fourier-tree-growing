import sympy
import numpy as np

def linear_dependency(v1, v2):
    m = np.array([v1, v2])
    print(m)
    mr, indexes = sympy.Matrix(m).T.rref()

    if len(indexes) == 2:
        return False
    else:
        return True