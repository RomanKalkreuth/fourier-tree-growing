# Source code of Fourier Tree Growing (FTG) and related experiments with Genetic Programming (GP) 
 
#### Overview
 
This project consists of several components designed to conduct experiments, analyze algorithms, and perform symbolic regression and other computational tasks. The structure is modular, with each directory serving a specific purpose in the project's ecosystem.
 
#### Directory Structure
 
- `experiments/`: Contains scripts and data related to experimental runs, including implementation of conventional GP, implementation of the proposed FTG and implementation of LSP benchmark.
- `constants/`: Defines constants used for the trees, facilitating easy modification and access to project-wide settings.
- `selection/`: Implements selection algorithms that are likely used to choose individuals in evolutionary algorithms or other optimization procedures.
- `variation/`: Contains operators for variation, such as mutation and crossover, which are essential components of evolutionary algorithms.
- `ftg/`: Contains the implementation of the Fourier Tree Growing (FTG) algorithm and visualization of the curve-fitting lines
- `representation/`: Manages the representation of individuals or solutions, such as genetic programming trees.
- `benchmark/`: Provides benchmark problems, data generation scripts, and utilities for evaluating the performance of algorithms.
  - `symbolic_regression/`: A subset of benchmarks specifically designed for testing symbolic regression algorithms.
- `util/`: Includes utility functions and helpers that provide general-purpose functionality used by other parts of the project.
 
#### Key Components
 
Each directory contains Python scripts with specific roles within the project. For example, `selection.py` in the `selection/` directory would define how individuals are selected for reproduction or survival, while `variation.py` in the `variation/` directory would detail the mechanisms for generating diversity within the population.

 
#### Detailed Documentation of FTG
 
- Implementation of experiments on conventional GP benchmarks are implemented in `src/experiments/ftg-expr.py`
- Implementation of experiments on LSP are implemented in `src/experiments/large_scale_poly_2.py`
 
#### Examples
 
- In order to run experiments with `ftg` with the budget of 1000 evaluations on benchmarking function `koza1` when trees have only one constant `1`, random seed is 0 and results are save in the directory with name `tmp` we run the following command in the folder `src/experiments`:

```python ftg-expr.py --algorithm ftg --evaluations 1000 --benchmark koza1 --constant 1 --dirname tmp --instance 0```
- In order to run experiments with `ftg` with the budget of 1000 evaluations on LSP with polynomial of degree `100` when trees have only one constant `1`, random seed is 0 and results are save in the directory with name `tmp` we run the following command in the folder `src/experiments`:

```python large_scale_poly_2.py --algorithm ftg --evaluations 1000 --degree 100 --constant 1 --dirname tmp --instance 0```
