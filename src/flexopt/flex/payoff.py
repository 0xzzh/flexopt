import numpy as np


def put(qty, s, k):
    return np.maximum(qty * (k - s), 0.0)


def call(qty, s, k):
    return np.maximum(qty * (s - k), 0.0)
