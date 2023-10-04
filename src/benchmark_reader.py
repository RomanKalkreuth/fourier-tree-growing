from ucimlrepo import fetch_ucirepo
from benchmark import Benchmark
import pandas as pd


def read_from_repo(repo_id):
    """
    :param repo_id

    :return benchmark
    """

    dataset = fetch_ucirepo(repo_id)
    name = dataset.metadata.name
    num_instances = dataset.metadata.num_instances
    num_features = dataset.metadata.num_features

    benchmark = Benchmark(dataset.data, name, num_instances, num_features)

    return benchmark


def read_from_file(file_path, name):
    """

    :param file_path:
    :param name

    :return:
    """

    data = pd.read_csv(file_path)

    num_instances = len(data)
    num_features = len(data[0])

    benchmark = Benchmark(data, name, num_instances, num_features)

    return benchmark


def read_from_file(file_path, name, num_inputs, num_outputs, separator, omit_header):
    """

    :param file_path:
    :param name
    :param num_inputs:
    :param num_outputs:
    :param separator
    :param omit_header

    :return:
    """
    data = []

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            values = line.split(separator)
            data.append(values)
    f.close()

    num_instances = len(data)
    num_features = len(data[0])

    benchmark = Benchmark(data, name, num_instances, num_features)

    benchmark.init_inputs(num_inputs, data[0:num_inputs])
    benchmark.init_outputs(num_outputs, data[num_inputs + 1:num_outputs])