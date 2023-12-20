import numpy as np

def linear_dependency(vectors):
    M = np.array(vectors)
    return np.linalg.matrix_rank(M @ M.T) != len(vectors)


def polynomial_degree_fit(x, y, max_deg):
    min_ssr = None
    best_deg = None
    MIN_ERROR = 10e-10
    for deg in range(1,max_deg+1):
        c, stats = np.polynomial.polynomial.polyfit(x, y, deg, full=True)
        ssr = stats[0]
        if min_ssr is None or ssr < min_ssr:
            min_ssr = ssr
            best_deg = deg
        if min_ssr <= MIN_ERROR:
            return best_deg
    return best_deg





