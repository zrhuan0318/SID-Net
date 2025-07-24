import numpy as np
import itertools
from math import log2
from collections import defaultdict

def myhistogram(data, bins):
    """ Discretize each row of a matrix independently """
    out = np.zeros_like(data, dtype=int)
    for i in range(data.shape[0]):
        out[i] = np.digitize(data[i], np.histogram_bin_edges(data[i], bins=bins)) - 1
    return out

def entropy(p):
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

def joint_entropy(data):
    _, counts = np.unique(data, axis=1, return_counts=True)
    probs = counts / np.sum(counts)
    return entropy(probs)

def mutual_information(X, Y):
    H_X = joint_entropy(X)
    H_Y = joint_entropy(Y)
    H_XY = joint_entropy(np.vstack((X, Y)))
    return H_X + H_Y - H_XY

def powerset(iterable, max_comb):
    "powerset(['a','b','c'], 2) --> (a,) (b,) (c,) (a,b) (a,c) (b,c)"
    s = list(iterable)
    return list(itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(1, max_comb+1)))

def compute_all_mi_combinations(Y, nbins=5, max_combs=2):
    """ Perform SID decomposition: returns I_U, I_R, I_S, MI """
    nvars = Y.shape[0] - 1  # first row is target
    X = Y[1:]
    T = Y[0:1]
    bins_data = myhistogram(Y, nbins)

    combs = powerset(range(nvars), max_combs)

    I_R = {}
    I_S = {}
    I_U = np.zeros(nvars)

    # Precompute MI for each combination with target
    mi_with_target = {}
    for c in combs:
        key = tuple(sorted(c))
        joint = np.vstack([bins_data[i+1] for i in key])
        mi = mutual_information(joint, bins_data[0:1])
        mi_with_target[key] = mi

    # Compute unique contributions
    for i in range(nvars):
        single = (i,)
        I_U[i] = mi_with_target[single]
        I_R[single] = I_U[i]

    # Compute redundancy and synergy
    for c in combs:
        if len(c) < 2:
            continue
        joint_mi = mi_with_target[c]
        redundant = sum(mi_with_target[(i,)] for i in c)
        synergy = joint_mi - redundant
        I_R[c] = redundant
        I_S[c] = synergy

    MI_total = sum(I_U) + sum(I_S.values())

    return I_U, I_R, I_S, MI_total
