# Copyright (C) 2023 -
# Roman Kalkreuth (Roman.Kalkreuth@lip6.fr)
# Computer Lab of Paris 6, Sorbonne Université (Paris, France)

__author__ = 'Roman Kalkreuth'
__copyright__ = 'Copyright (C) 2023, Roman Kalkreuth'
__version__ = '1.0'
__email__  = 'Roman.Kalkreuth@lip6.fr'

from abc import ABC, abstractmethod
import numpy as np
class GPProblem(ABC):
    """

    """
    @abstractmethod
    def evaluate(self, tree):
        pass

class RegressionProblem(GPProblem):

    def __init__(self, X_train,  y_train, X_test=None, y_test=None):
        self.X_train = X_train
        self.y_train = y_train

        self.X_test = X_test
        self.y_test = y_test

        self.n = len(X_train)

    def evaluate(self, tree):
        prediction = np.zeros(self.n)
        for index, X in enumerate(self.X_train):
            prediction[index] = tree.eval(X)
        return prediction
