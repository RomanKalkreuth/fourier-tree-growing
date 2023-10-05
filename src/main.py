from benchmark_reader import UCIMLReader
import dataset_generator as dg

#reader = UCIMLReader()
#benchmark = reader.read_from_file('../benchmarks/regression/uciml/airfoil_self_noise.dat',
#                                  "airfoil", separator='\t', num_inputs=5, num_outputs=1)


grid = dg.evenly_spaced_grid(-5, 5,0.4)
