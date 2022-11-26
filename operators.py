import numpy as np


def productN(args, op):
    return np.product(args, axis=0)


def zadeh_t(args, op):  # min(x,y)
    return np.min(args, axis=0)


def zadeh_s(args, op):  # max(x,y)
    return np.max(args, axis=0)


def algebraic_t(args, op):  # x*y
    return np.product(args, axis=0)


def probabilistic_s(args, op):  # x+y-x*y
    return np.sum(args, axis=0) - np.product(args, axis=0)


def lukasiewicz_t(args, op):    # max(x+y-1.0,0)
    return np.clip(np.sum(args, axis=0) - 1.0, a_min=0, a_max=None)


def lukasiewicz_s(args, op): # min(x+y,1)
    return np.clip(np.sum(args, axis=0) + 1, a_min=None, a_max=1)


def fodor_t(args, op):
    mins = np.min(args, axis=0)
    zeros = np.zeros_like(mins)
    condition = np.sum(args, axis=0) > 1
    return np.where(condition, mins, zeros)


def fodor_s(args, op):
    maxs = np.max(args, axis=0)
    ones = np.ones_like(maxs)
    condition = np.sum(args, axis=0) < 1
    return np.where(condition, maxs, ones)


def drastic_t(args, op):
    assert len(args) == 2
    a, b = args
    return np.select([a == 0, b == 0], [b, a], default=1)


def drastic_s(args, op):
    return 1 - drastic_t(1 - args, op)  # from general c-norm definition


def einstein_t(args, op):
    num = np.product(args, axis=0)
    denom = 2 - ( np.sum(args, axis=0) - num )
    return num / denom


def einstein_s(args, op):
    num = np.sum(args, axis=0)
    denom = 1 + np.product(args, axis=0)
    return num / denom
