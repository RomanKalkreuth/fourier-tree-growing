from src.benchmark.symbolic_regression import functions as fn, dataset_generator as dg
import matplotlib.pyplot as plt

#reader = UCIMLReader()
#benchmark = reader.read_from_file('../benchmarks/regression/uciml/airfoil_self_noise.dat',
#                                  "airfoil", separator='\t', num_inputs=5, num_outputs=1)


#grid = dg.evenly_spaced_grid(-5, 5,0.4)

koza_sample = dg.random_samples_float(-1.0, 1.0, 20)
grid = dg.evenly_spaced_grid(-100, 100,0.05)
ground_truth = fn.koza2(grid)

print(grid)
print(ground_truth)

plt.plot(grid, ground_truth)
plt.show()


