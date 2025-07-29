import numpy as np
from numba import njit

@njit(fastmath=True)
def binary_f1_score(conf_matrix):
    # Calculate binary F1 score from confusion matrix
    f1_score = 0
    for label in (0, 1):
        if label == 0:
            tp, fp, fn, _ = conf_matrix
        else:
            _, fn, fp, tp = conf_matrix
        if (tp + fp) == 0 or (tp + fn) == 0:
            return -np.inf
        pr = tp / (tp + fp)
        re = tp / (tp + fn)
        if (pr + re) == 0:
            return -np.inf
        f1 = 2 * (pr * re) / (pr + re)
        f1_score += f1
    return f1_score / 2