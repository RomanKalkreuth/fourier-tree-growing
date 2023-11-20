import math
import numpy as np


def koza1(x):
    """
    x^4 + x^3 + x^2 + x
    """
    return x ** 4 + x ** 3 + x ** 2 + x


def koza2(x):
    """
    x^5 - 2x^3 + x
    """
    return x ** 5 - 2 * x ** 3 + x


def koza3(x):
    """
    x^6 - 2x^4 + x^2
    """
    return x ** 6 - 2 * x ** 4 + x ** 2

def nguyen4(x):
    """
    x^6 + x^5 + x^4 + x^3 + x^2 + x
    """
    return x ** 6 + x ** 5 + x ** 4 + x ** 3 + x ** 2 + x

def nguyen5(x):
    """
    sin(x^2) cos(x) - 1
    """
    return math.sin(x**2) * math.cos(x) - 1

def nguyen6(x):
    """
    sin(x) + sin(x + x^2 )
    """
    return math.sin(x) + math.sin(x+x**2)


def nguyen7(x):
    """
    ln(x + 1) + ln(x^2 + 1)
    """
    return math.log(x + 1) + math.log(x ** 2 + 1)


def nguyen9(x, y):
    """

    """
    return math.sin(x) + math.sin(y ** 2)


def nguyen10(x, y):
    """

    """
    return 2 * math.sin(x) + math.cos(y)


def keijzer6(x):
    """

    """
    s = 0
    fx = math.floor(x)
    for i in range(1, fx + 1):
        s += (1.0 / i)
    return s


def vladislavleva4(xs):
    """

    """
    s = 0
    for i in range(0, 5):
        s += (xs[i] - 3) * (xs[i] - 3)

    return 10.0 / (5.0 + s)


def pagie1(x, y):
    """
    1 / (1 + x^-4) + 1 / (1 + y^-4)
    """
    return 1 / (1 + math.pow(x, -4)) + 1 / (1 + math.pow(y, -4))


def pagie2(x, y, z):
    """
    (1 / (1 + x^-4) + 1 / (1 + y^-4) + 1 / (1 + z^-4));
    """
    return 1 / (1 + math.pow(x, -4)) + 1 / (1 + math.pow(y, -4)) + 1 / (1 + math.pow(z, -4))


def korns12(xs):
    """

    """
    return 2.0 - (2.1 * (math.cos(9.8 * xs[0]) * math.sin(1.3 * xs[4])))



