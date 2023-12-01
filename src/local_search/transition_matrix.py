
import numpy as np

class TransitionMatrix:

    def __init__(self, n):
        self.matrix = np.zeros((n, n))

    def __len__(self):
        return len(self.matrix)

    def update(self, s1, s2, val):
        self.matrix[s1][s2] += val

