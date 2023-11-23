import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class Benchmark:

    def __init__(self, data, name, num_instances, num_features):
        self.outputs = None
        self.inputs = None
        self.num_inputs = None
        self.num_outputs = None

        self.name = name
        self.data = data

        self.num_instances = num_instances
        self.num_features = num_features

    def init_inputs(self, num_inputs, data):
        self.num_inputs = num_inputs
        self.inputs = np.array(data)

    def init_outputs(self, num_outputs, data):
        self.num_outputs = num_outputs
        self.outputs = np.array(data)

    def split_data(self, test_size):
        X_train, X_test, y_train, y_test = train_test_split(self.inputs, self.outputs,
                                                            test_size=test_size, random_state=42)

        return X_train, X_test, y_train, y_test
