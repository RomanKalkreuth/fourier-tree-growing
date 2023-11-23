from ucimlrepo import fetch_ucirepo
from benchmark import Benchmark
import pandas as pd


class UCIMLReader:

    def read_from_repo(self, repo_id):
        dataset = fetch_ucirepo(repo_id)
        name = dataset.metadata.name
        num_instances = dataset.metadata.num_instances
        num_features = dataset.metadata.num_features

        benchmark = Benchmark(dataset.get_symbol, name, num_instances, num_features)

        return benchmark

    def read_from_file(self, file_path, name, num_inputs=None, num_outputs=None, separator=',', omit_header=None):
        data = pd.read_csv(file_path, sep=separator)

        num_instances = data.shape[0]
        num_features = data.shape[1]

        benchmark = Benchmark(data, name, num_instances, num_features)

        if num_inputs is not None:
            inputs = data.iloc[:, 0:num_inputs]
            benchmark.init_inputs(num_inputs, inputs)
        if num_outputs is not None:
            outputs = data.iloc[:, num_inputs:num_inputs + num_outputs]
            benchmark.init_outputs(num_outputs, outputs)

        return benchmark
