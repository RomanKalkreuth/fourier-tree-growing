import numpy as np
from src.representation.parse_tree import ParseTree, FUNCTIONS, VARIABLES, TERMINALS
from src.functions.functions import Mathematical


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

def function_values(tree, min, max, n):

    X = np.linspace(min, max, num=n)
    y = np.zeros(n)

    for index, x in enumerate(X):
        inp = np.array(x)
        y[index] = tree.evaluate(inp)

    return X, y


def normalize_polynomial(tree):
    if not tree:
        return None, None
    degrees_left, constants_left = normalize_polynomial(tree.left)
    degrees_right, constants_right = normalize_polynomial(tree.right)
    degrees, constants = [], []
    if tree.symbol in VARIABLES:
        degrees.append(1)
        constants.append(1)
    elif tree.symbol in TERMINALS:
        degrees.append(0)
        constants.append(int(tree.symbol))
    elif tree.symbol == Mathematical.add or tree.symbol == Mathematical.sub:
        it_left, it_right = 0, 0
        while it_left < len(degrees_left) and it_right < len(degrees_right):
            if degrees_left[it_left] < degrees_right[it_right]:
                degrees.append(degrees_left[it_left])
                constants.append(constants_left[it_left])
                it_left += 1
            elif degrees_left[it_left] > degrees_right[it_right]:
                degrees.append(degrees_right[it_right])
                constants.append(tree.symbol(0, constants_right[it_right]))
                it_right += 1
            else:
                degrees.append(degrees_left[it_left])
                constants.append(tree.symbol(constants_left[it_left], constants_right[it_right]))
                it_left += 1
                it_right += 1
        while it_left < len(degrees_left):
            degrees.append(degrees_left[it_left])
            constants.append(constants_left[it_left])
            it_left += 1
        while it_right < len(degrees_right):
            degrees.append(degrees_right[it_right])
            constants.append(tree.symbol(0, constants_right[it_right]))
            it_right += 1
    elif tree.symbol == Mathematical.mul:
        ds = np.zeros(len(degrees_left) * len(degrees_right), dtype=int)
        cs = np.zeros(len(degrees_left) * len(degrees_right), dtype=int)
        cnt = 0
        for i in range(len(degrees_left)):
            for j in range(len(degrees_right)):
                ds[cnt] = degrees_left[i] + degrees_right[j]
                cs[cnt] = constants_left[i] * constants_right[j]
                cnt += 1
        sorted_ids = np.argsort(ds)
        prv_degree = ds[sorted_ids[0]]
        c = 0
        for i in sorted_ids:
            if prv_degree != ds[i]:
                degrees.append(prv_degree)
                constants.append(c)
                c = 0
            c += cs[i]
            prv_degree = ds[i]
        degrees.append(prv_degree)
        constants.append(c)
    else:
        raise ValueError(f'Symbol {tree.symbol} is not supported')
    return np.array(degrees), np.array(constants)


def sorted_degrees_constants_to_str(sorted_degrees, constants):
    a = []
    for i in range(len(sorted_degrees)):
        if i > 0:
            s = f'{abs(constants[i])}*x^{sorted_degrees[i]}'
            if constants[i] < 0:
                a.append(f' - {s}')
            else:
                a.append(f' + {s}')
        else:
            s = f'{constants[i]}*x^{sorted_degrees[i]}'
            a.append(s)
    return ''.join(a)
