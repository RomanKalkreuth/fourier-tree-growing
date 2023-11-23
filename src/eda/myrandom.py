# @author Kirill Antonov

import random
import math
import numpy as np


class RandomEngine:
    def __init__(self, seed=None):
        if seed:
            random.seed(seed)

    @staticmethod
    def sample_uniform01():
        return random.random()

    @staticmethod
    def sample_symmetrical_Bernoulli():
        return random.randint(0, 1)

    @staticmethod
    def sample_Bernoulli(success_p):
        return random.random() < success_p

    @staticmethod
    def sample_int_uniform(from_incl, to_excl):
        return random.randint(from_incl, to_excl - 1)

    @staticmethod
    def sample_Binomial(n, success_p):
        success = 0
        for i in range(n):
            if RandomEngine.sample_uniform01() < success_p:
                success += 1
        return success

    @staticmethod
    def sample_discrete_dist(p):
        r = RandomEngine.sample_uniform01()
        cdf = 0
        for i in range(1, len(p) + 1):
            cdf = cdf + p[i - 1]
            if cdf > r:
                return i - 1

    class TruncatedExponentialDistribution:
        def __init__(self):
            self.a = 0.
            self.b = 1.
            self.lambda0 = 0.
            self.lambda1 = 0.

        def build(self, m, acc):
            lb = -1. / m
            ub = -float(acc)
            while ub - lb >= acc:
                lambda_ = (lb + ub) / 2.
                left = np.exp(lambda_) * (lambda_ - 1) / (np.exp(lambda_) - 1) + 1 / (np.exp(lambda_) - 1)
                right = m * lambda_
                if left >= right:
                    lb = lambda_
                else:
                    ub = lambda_
            self.lambda1 = lb
            self.lambda0 = np.log(self.lambda1 / (np.exp(self.lambda1) - 1))
            return self

        def pdf(self, x):
            return np.exp(self.lambda0 + self.lambda1 * x)

        def inverse_cdf(self, p):
            return (np.log(self.lambda1 * p + np.exp(self.lambda0)) - self.lambda0) / self.lambda1

        def sample(self):
            p = random.random()
            return self.inverse_cdf(p)

    @staticmethod
    def Cnk(n, k):
        if n == 0 and k == 0:
            return 1
        if k == 0:
            return 1
        if k > n:
            return 0
        return math.factorial(n) // math.factorial(n - k) // math.factorial(k)

    @staticmethod
    def get_combination_by_number(n, k, pos):
        sel = [0] * k
        cur = 0
        prv = 0
        for i in range(k):
            isok = False
            for j in range(prv + 1, n + 1):
                d = RandomEngine.Cnk(n - j, k - i - 1)
                if cur + d >= pos:
                    sel[i] = j
                    prv = j
                    isok = True
                    break
                cur += d
            if not isok:
                return None
        return sel

    @staticmethod
    def sample_combination_uniform(n, k):
        pos = RandomEngine.sample_int_uniform(1, RandomEngine.Cnk(n, k) + 1)
        return RandomEngine.get_combination_by_number(n, k, pos)
