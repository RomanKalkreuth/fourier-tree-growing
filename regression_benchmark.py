class RegressionBenchmark:
    """

    """


    def __init__(self, name, num_inputs, num_outputs):
        """
        :param num_inputs
        :param num_outputs
        """

        self.name = name

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.input_data = []
        self.output_data = []
