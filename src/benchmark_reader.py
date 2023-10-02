from ucimlrepo import fetch_ucirepo
import benchmark as bm


def read_from_repo(self, id):
    """
    :param id

    :return benchmark
    """

    dataset = fetch_ucirepo(id)
    name = dataset.metadata.name
    num_instances = dataset.metadata.num_instances
    num_features = dataset.metadata.num_features

    benchmark = bm.Benchmark(dataset.data, name, num_instances, num_features)

    return benchmark



def read_from_file(self, benchmark, file_path, num_inputs, num_outputs, separator, omit_header):
    """

    :param benchmark:
    :param file_path:
    :param num_inputs:
    :param num_outputs:
    :return:
    """

    data = []

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            values = line.split('\t')
            data.append(values)
    f.close()

    for values in data:
        for index, value in enumerate(values):
            if index < num_inputs:
                benchmark.input.append(value)
            else:
                benchmark.output.append(value)
    print(data)


