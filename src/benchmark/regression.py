from gplearn.genetic import SymbolicRegressor


def gplearn_regressor(benchmark, population_size=5000,
                      generations=20, stopping_criteria=0.01,
                      p_crossover=0.7, p_subtree_mutation=0.1,
                      p_hoist_mutation=0.05, p_point_mutation=0.1,
                      max_samples=0.9, verbose=1,
                      parsimony_coefficient=0.01, random_state=0):
    """
    """

    est_gp = SymbolicRegressor(population_size=population_size,
                                   generations=generations, stopping_criteria=stopping_criteria,
                                   p_crossover=p_crossover, p_subtree_mutation=p_subtree_mutation,
                                   p_hoist_mutation=p_hoist_mutation, p_point_mutation=p_point_mutation,
                                   max_samples=max_samples, verbose=verbose,
                                   parsimony_coefficient=parsimony_coefficient, random_state=random_state)

    X_train, X_test, y_train, y_test = benchmark.split_data(0.3)

    est_gp.fit(X_train, y_train)
