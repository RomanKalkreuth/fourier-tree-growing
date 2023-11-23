# Copyright (C) 2023 -
# Roman Kalkreuth (Roman.Kalkreuth@lip6.fr)
# Computer Lab of Paris 6, Sorbonne Université (Paris, France)

from random import random, randint

__author__ = 'Roman Kalkreuth'
__copyright__ = 'Copyright (C) 2023, Roman Kalkreuth'
__version__ = '1.0'
__email__ = 'Roman.Kalkreuth@lip6.fr'

from random import random


def subtree_crossover(ptree1, ptree2, crossover_rate):
    if random() < crossover_rate:
        crossover_point1 = randint(1, ptree1.size() - 1)
        crossover_point2 = randint(1, ptree2.size() - 1)

        otree1 = ptree1.clone()
        otree2 = ptree2.clone()

        subtree1 = otree1.subtree(crossover_point1)
        subtree2 = otree2.subtree(crossover_point2)

        otree1.replace_subtree(subtree2, crossover_point1)
        otree2.replace_subtree(subtree1, crossover_point2)

        return otree1, otree2
    else:
        return ptree1, ptree2
