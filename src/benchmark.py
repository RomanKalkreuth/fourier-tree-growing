import numpy as np
import pandas as pd

class Benchmark:
    """

    """

    def __init__(self, data, name, num_instances, num_features):
        """

        :param name:
        :param data:
        :param num_instances:
        :param num_features:
        """

        self.name = name
        self.data = data

        self.num_instances = num_instances
        self.num_features = num_features


        def init_inputs(num_inputs):
            """

            :return:
            """
            self.num_inputs = num_inputs

        def init_outputs(num_outputs):
            """

            :return:
            """
            self.num_outputs = num_outputs


