"""Activation matching (the second standard alignment arm, after Ainsworth
et al.): correspondences from activations on a frozen probe set.

Cost for axis p is the Pearson correlation between unit activations over
the probe buffer (post-ReLU, z-scored per unit), and each axis is one exact
LAP; unlike weight matching there is no coordinate descent, because a
unit's activations do not depend on how other axes are permuted. Constant
units (dead, or ReLU never active on the probe set) have undefined
correlation and get zero cost rows, so they match arbitrarily among
themselves, the same tie class the weight arm quotients.

Returns the same perms convention as wsl.align.weight_matching:
perms[p][i] = index into B matched to A's unit i.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from .activations import activation_matrices


def _zscore(X, eps=1e-12):
    X = X - X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    live = std[:, 0] > eps
    X = np.where(std > eps, X / np.maximum(std, eps), 0.0)
    return X, live


def activation_matching(model_a, model_b, states, device, batch_size=512):
    """Permutations aligning B's units to A's by activation correlation."""
    acts_a = activation_matrices(model_a, states, device, batch_size)
    acts_b = activation_matrices(model_b, states, device, batch_size)
    perms = {}
    for ax in acts_a:
        Za, _ = _zscore(acts_a[ax])
        Zb, _ = _zscore(acts_b[ax])
        C = Za @ Zb.T / Za.shape[1]
        ri, ci = linear_sum_assignment(-C)
        perms[ax] = ci[np.argsort(ri)]
    return perms
