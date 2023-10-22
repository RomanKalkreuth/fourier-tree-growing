from abc import ABC, abstractmethod
import numpy as np
class GPProblem(ABC):
    """

    """

    def __init__(self, ground_truth):
        self.ground_truth = ground_truth
        self.n = len(ground_truth)

    @abstractmethod
    def evaluate(self, tree):
        pass

class RegressionProblem(GPProblem):
    def evaluate(self, tree):
        prediction = np.zeros(self.n)
        for index, data in enumerate(self.ground_truth):
            prediction[index] = tree.evaluate(data)
        return prediction
