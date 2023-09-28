#from ucimlrepo import fetch_ucirepo
class BenchmarkReader:
    """

    """

    def __init__(self):
        """

        """

    def read_file(self, benchmark, file_path, num_inputs, num_outputs, separator, omit_header):
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


