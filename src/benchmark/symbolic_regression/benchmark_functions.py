import math


def polynomial(x, degree):
    s = 0.0
    for d in range(1, degree + 1):
        s += x ** d
    return s


def koza1(x):
    """
    x^4 + x^3 + x^2 + x
    """
    return polynomial(x, 4)


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


def nguyen3(x):
    return polynomial(x, 5)


def nguyen4(x):
    """
    x^6 + x^5 + x^4 + x^3 + x^2 + x
    """
    return polynomial(x, 6)


def nguyen5(x):
    """
    sin(x^2) cos(x) - 1
    """
    return math.sin(x ** 2) * math.cos(x) - 1


def nguyen6(x):
    """
    sin(x) + sin(x + x^2 )
    """
    return math.sin(x) + math.sin(x + x ** 2)


def nguyen7(x):
    """
    ln(x + 1) + ln(x^2 + 1)
    """
    return math.log(x + 1) + math.log(x ** 2 + 1)


def nguyen8(x):
    """
    sqrt(x)
    """
    return math.sqrt(x)


def nguyen9(args):
    """
    sin(x) + sin(y^2)
    """
    return math.sin(args[0]) + math.sin(args[1] ** 2)


def nguyen10(args):
    """
    2 * sin(x) * cos(y)
    """
    return 2 * math.sin(args[0]) + math.cos(args[1])


def keijzer6(x):
    """
    sum_{i=1}^{x} 1/i
    """
    s = 0
    fx = math.floor(x)
    for i in range(1, fx + 1):
        s += (1.0 / i)
    return s


def vladislavleva4(args):
    """

    """
    s = 0
    for i in range(0, 5):
        s += (args[i] - 3) * (args[i] - 3)

    return 10.0 / (5.0 + s)


def pagie1(args):
    """
    1 / (1 + x^-4) + 1 / (1 + y^-4)
    """
    return 1 / (1 + math.pow(args[0], -4)) + 1 / (1 + math.pow(args[1], -4))


def pagie2(args):
    """
    (1 / (1 + x^-4) + 1 / (1 + y^-4) + 1 / (1 + z^-4));
    """
    return 1 / (1 + math.pow(args[0], -4)) + 1 / (1 + math.pow(args[1], -4)) + 1 / (1 + math.pow(args[2], -4))


def korns12(args):
    """
    2 - 2.1 * cos(9.8*x)*sin(1.3*w)
    """
    return 2.0 - (2.1 * (math.cos(9.8 * args[0]) * math.sin(1.3 * args[4])))
