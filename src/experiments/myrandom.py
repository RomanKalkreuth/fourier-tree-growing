import random


def set_random_seed(seed):
    global RND
    RND = random.Random(seed)
