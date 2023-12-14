import numpy as np
import config

class TransitionMatrix:

    def __init__(self, n):
        self.matrix = np.zeros((n, n))

    def __len__(self):
        return len(self.matrix)

    def __str__(self):
        return self.matrix.__str__()

    def update(self, s1, s2, val):
        i1 = config.FUNCTIONS.index(s1)
        i2 = config.FUNCTIONS.index(s2)
        self.matrix[i1][i2] += val


