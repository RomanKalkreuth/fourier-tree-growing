from analysis import *
from src.representation.parse_tree import ParseTree
from src.functions.functions import Mathematical

def test_simple():
    t = ParseTree()
    t.symbol = Mathematical.sub
    t.left = ParseTree()
    t.left.symbol = 'x'
    t.right = ParseTree()
    t.right.symbol = Mathematical.sub
    t.right.left = ParseTree()
    t.right.left.symbol = Mathematical.mul
    t.right.left.left = ParseTree()
    t.right.left.left.symbol = 'x'
    t.right.left.right = ParseTree()
    t.right.left.right.symbol = 'x'
    t.right.right = ParseTree()
    t.right.right.symbol = 'x'
    print(t)
    ds, cs = normalize_polynomial(t)
    s = sorted_degrees_constants_to_str(ds, cs)
    print(s)

def test_rnd():
    t = ParseTree()
    t.init_tree(1, 10)
    ds, cs = normalize_polynomial(t)
    s = sorted_degrees_constants_to_str(ds, cs)
    print(s)


def test_2():
    for i in range(int(1e5)):
        t = ParseTree()
        t.init_tree(1, 10)
        ds, cs = normalize_polynomial(t)
        N = 10
        X = np.random.uniform(-1, 1, N)
        for x in X:
            tans = t.evaluate(np.array(x))
            pans = sum(cs[i]*x**ds[i] for i in range(len(ds)))
            if abs(tans - pans) > 1e-5:
                print('tans', tans, 'pans', pans)
                print(t)
                s = sorted_degrees_constants_to_str(ds, cs)
                print(s)
                return

test_simple()
# test_rnd()
test_2()
