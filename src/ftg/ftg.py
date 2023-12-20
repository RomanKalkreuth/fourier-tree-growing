import sys
sys.path.insert(0, '../gp/gpsimple')

import copy
from gp_tree import GPNode
from gp_simple import GPSimple
import gp_config as config
import gp_problem as problem
import gp_fitness as fitness
import gp_util as util

import src.benchmark.symbolic_regression.dataset_generator as generator
import src.benchmark.symbolic_regression.benchmark_functions as benchmarks

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import sys
from sympy import *


def eprintf(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def toseq(x_bar, f):
    return np.vectorize(f)(x_bar)


def inner_prod(x1, x2):
    return np.dot(x1.losses, x2.losses)


def minus_fun(x1, x2):
    fun = Fun(None, None)
    fun.vec = x1.losses - x2.losses
    return fun


def gram(fs):
    g = np.zeros((len(fs), len(fs)))
    for i in range(len(fs)):
        for j in range(i, len(fs)):
            g[i][j] = inner_prod(fs[j], fs[i])
            g[j][i] = g[i][j]
    return g


def proj_to_basis(x, basis):
    proj = np.zeros(len(basis))
    for i in range(len(basis)):
        proj[i] = inner_prod(x, basis[i])
    return proj


def proj_to_hyperplane(x, basis):
    G = gram(basis)
    eprintf('det', np.linalg.det(G))
    eprintf('cond', np.linalg.cond(G))
    return np.dot(np.linalg.inv(gram(basis)), proj_to_basis(x, basis))


class Fun:
    def __init__(self, x_bar, f):
        if f:
            self.f = f
            self.vec = toseq(x_bar, f)


def gen_tree(fs):
    GPSimple.init(None, None)
    tree = GPNode()
    tree.init_tree(1, 1)
    tree.symbol = 'x'
    tree.print_tree()
    print(tree.evaluate(4))


def loss_mse(X_train, F, F_hat):
    loss = 0
    for i in range(len(X_train)):
        loss += (F(X_train[i]) - F_hat(X_train[i]))**2
    return loss


def linear_combination_of_trees_to_function(alpha, trees):
    return lambda x: np.dot(alpha, [t.evaluate(x) for t in trees])


def is_inner_prod_zero(inner_prod_value):
    return abs(inner_prod_value) < 1e-5


def is_linear_dependent(Z, f_hat_fun, gi_fun):
    ip = inner_prod(minus_fun(Z, f_hat_fun), gi_fun)
    eprintf('ip', ip)
    return is_inner_prod_zero(ip)


def vis1(X_train, F, F_hat):
    visx = np.linspace(-1, 1, 1000)
    plt.scatter(X_train, np.vectorize(F)(X_train))
    plt.plot(visx, np.vectorize(F)(visx), linestyle='dashed')
    plt.plot(visx, np.vectorize(F_hat)(visx))
    plt.savefig('vis-1.pdf')
    plt.close()


class Vis:
    def __init__(self, minx, maxx, X_train, F):
        self.minx = minx
        self.maxx = maxx
        self.visx = np.linspace(minx, maxx, 1000)
        self.list_values = []
        self.fig, self.ax = plt.subplots()
        self.ax.plot(self.visx, np.vectorize(F)(self.visx), linestyle='dashed', c='black')
        self.ax.scatter(X_train, np.vectorize(F)(X_train), c='black')
    def add_function(self, f):
        self.list_values.append(np.vectorize(f)(self.visx))
    def build_save_close(self, figname):
        cmap = mpl.cm.jet
        for i in range(len(self.list_values)):
            self.ax.plot(self.visx, self.list_values[i], c=cmap(i/len(self.list_values)))
        self.fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap), shrink=0.5, aspect=5, ax=self.ax)
        plt.savefig(figname)
        plt.close()


def gentree():
    gi = GPNode()
    gi.init_tree(1, 10)
    return gi


def main2():
    GPSimple.init(None, None)
    F = benchmarks.nguyen5
    minx, maxx = -1, 1
    X_train = np.random.uniform(minx, maxx, 20)

    F_fun = Fun(X_train, F)
    t1 = GPNode()
    t1.symbol = 1
    ts = [t1]

    alpha = [sum(F(x) for x in X_train)/len(X_train)]
    F_hat = linear_combination_of_trees_to_function(alpha, ts)
    F_hat_fun = Fun(X_train, F_hat)
    basis = [Fun(X_train, lambda x: t1.evaluate(x))]
    loss = loss_mse(X_train, F, F_hat)
    eprintf(loss)
    print('loss', loss)

    vis = Vis(minx, maxx, X_train, F)
    vis.add_function(F_hat)

    spent_budget = 2
    for i in range(2, 10):
        if loss < 1e-5:
            break
        ts.append(gentree())
        basis.append(Fun(X_train, lambda x: ts[-1].evaluate(x)))
        spent_budget += 1
        while is_linear_dependent(F_fun, F_hat_fun, basis[-1]):
            eprintf('ti', util.generate_symbolic_expression(ts[-1]))
            spent_budget += 2
            ts[-1] = gentree()
            basis[-1] = Fun(X_train, lambda x: ts[-1].evaluate(x))
            spent_budget += 1
        eprintf('ti', util.generate_symbolic_expression(ts[-1]))
        alpha = proj_to_hyperplane(F_fun, basis)
        spent_budget += len(basis)*(len(basis)+1)/2 + len(basis)
        F_hat = linear_combination_of_trees_to_function(alpha, ts)
        F_hat_fun = Fun(X_train, F_hat)
        spent_budget += 1
        vis.add_function(F_hat)
        loss = loss_mse(X_train, F, F_hat)
        eprintf(loss)
        print('loss', loss)
    vis.build_save_close('vis-1.pdf')
    eprintf(spent_budget)
    print('budget', spent_budget)


def main1():
    GPSimple.init(None, None)
    # X_train = generator.random_samples_float(-1.0, 1.0, 20)
    X_train = np.linspace(-1.0, 1.0, 20)
    F = benchmarks.koza1
    Z = Fun(X_train, F)
    f1 = GPNode()
    f1.symbol = 1
    fs = [f1]
    g = []
    alpha = [sum(F(x) for x in X_train)/len(X_train)]
    # print(alpha)
    # f1_fun = Fun(X_train, lambda x: f1.evaluate(x))
    # alpha = proj_to_hyperplane(Z, [f1_fun]).tolist()
    # print(alpha)
    vis = Vis(-1, 1, X_train, F)
    vis.add_function(linear_combination_of_trees_to_function(alpha, fs))
    for i in range(2, 10):
        F_hat = linear_combination_of_trees_to_function(alpha, fs)
        fun_hat = Fun(X_train, F_hat)
        eprintf(loss_mse(X_train, F, F_hat))
        print(loss_mse(X_train, F, F_hat))
        gi = GPNode()
        gi.init_tree(1, 4)
        gi_fun = Fun(X_train, lambda x: gi.evaluate(x))
        while is_linear_dependent(Z, fun_hat, gi_fun):
            gi = GPNode()
            gi.init_tree(1, int(4*1.001))
            gi_fun = Fun(X_train, lambda x: gi.evaluate(x))
        alpha_new = proj_to_hyperplane(Z, [fun_hat, gi_fun])
        for i in range(len(alpha)):
            alpha[i] *= alpha_new[0]
        alpha.append(alpha_new[-1])
        fs.append(gi)
        eprintf('alphas', alpha_new)
        expr = util.generate_symbolic_expression(gi)
        eprintf('new expression', expr)
        vis.add_function(linear_combination_of_trees_to_function(alpha, fs))
    vis.build_save_close('vis-1.pdf')
    eprintf(loss_mse(X_train, F, F_hat))
    print(loss_mse(X_train, F, F_hat))


def main():
    minx, maxx = 0, 6
    x_bar = np.linspace(minx, maxx, 20)
    def F(x): return 10*np.sin(x) + 50*np.cos(x)**2
    visx = np.linspace(minx, maxx, 1000)
    plt.scatter(x_bar, np.vectorize(F)(x_bar))
    plt.plot(visx, np.vectorize(F)(visx), linestyle='dashed')
    Z = Fun(x_bar, F)
    Fs = [
        Fun(x_bar, lambda x: 1),
        Fun(x_bar, lambda x: x),
        Fun(x_bar, lambda x: x**2),
        Fun(x_bar, lambda x: x**3),
        Fun(x_bar, lambda x: x**4),
        Fun(x_bar, lambda x: x**5),
        Fun(x_bar, lambda x: x**6),
        Fun(x_bar, lambda x: x**7),
        Fun(x_bar, lambda x: x**8),
        Fun(x_bar, lambda x: 1/(x+1)),
        # Fun(x_bar, lambda x: 1/(x+1)**2)
    ]
    # M = np.zeros((6,6), dtype=np.double)
    # for j in range(0, 6):
        # for i in range(0, 6):
            # M[i][j] = Fs[j].vec[i]
    # eigv1 = np.linalg.eig(M.T @ M)[0]
    eigv2 = np.linalg.eig(gram(Fs))[0]
    # print('eignv M.T @ M, cond', eigv1, eigv1[0]/eigv1[5])
    print('eignv gram, cond', eigv2.tolist(), eigv2[0]/eigv2[-1])
    # print('cond(M.T @ M)', np.linalg.cond(M.T @ M))
    print('cond(gram)', np.linalg.cond(gram(Fs)))
    M = Matrix(gram(Fs))
    print(M.rref())
    # print('det(M.T)', np.linalg.det(M.T))
    # print('det(M)', np.linalg.det(M))
    # print('det(M.T @ M)', np.linalg.det(np.dot(M.T, M)))
    print('det(gram)', np.linalg.det(gram(Fs)))
    # for i in range(1, 4):
    # Fs.append(Fun(x_bar, lambda x: np.power(x, i)))
    st = time.time()
    alpha = proj_to_hyperplane(Z, Fs)
    en = time.time()
    print('time', en - st, 'secs')
    def f(x): return np.dot(alpha, [fun.f(x) for fun in Fs])
    print('alphas', alpha.tolist())
    print('mse loss', np.sum((Z.vec - np.vectorize(f)(x_bar))**2))
    plt.plot(visx, np.vectorize(f)(visx))
    plt.savefig('vis.pdf')
    plt.close()


# gen_tree([])
main2()
