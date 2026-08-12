"""Z4 supervised control: the Ainsworth MLP recipe on MNIST (three hidden
layers of 512, Adam 1e-3), with its own permutation machinery mirroring
wsl.align's conventions.

Deliberately self-contained rather than generalising the DQN-specific
align.py mid-project: the DQN machinery is verified against shipped
checkpoints and stays untouched; the duplication here is the price and it
is covered by the same test battery (function preservation, exact recovery
with restarts, live-identity self-match).

Permutation axes: h1, h2, h3, the three hidden layers. perms[ax][i] is the
index into B matched to A's unit i, as in wsl.align.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .align import DEAD_NORM

MLP_AXES = ("h1", "h2", "h3")

_ROWS = {"h1": [("fc1.weight", None), ("fc1.bias", None)],
         "h2": [("fc2.weight", "h1"), ("fc2.bias", None)],
         "h3": [("fc3.weight", "h2"), ("fc3.bias", None)]}
_COLS = {"h1": [("fc2.weight", "h2")],
         "h2": [("fc3.weight", "h3")],
         "h3": [("out.weight", None)]}

AXIS_LAYER_MLP = {"h1": "fc1.weight", "h2": "fc2.weight", "h3": "fc3.weight"}


class MLP(nn.Module):
    """784 -> hidden x3 -> 10, ReLU."""

    def __init__(self, hidden=512):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 10)

    def forward(self, x):
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)


def perm_sizes_mlp(sd):
    return {"h1": int(sd["fc1.weight"].shape[0]),
            "h2": int(sd["fc2.weight"].shape[0]),
            "h3": int(sd["fc3.weight"].shape[0])}


def _np(sd, name):
    return sd[name].detach().cpu().numpy().astype(np.float64)


def _rows_matrix(sd, name, other_perm_idx):
    w = _np(sd, name)
    if w.ndim == 1:
        return w[:, None]
    if other_perm_idx is not None:
        w = w[:, other_perm_idx]
    return w


def _cols_matrix(sd, name, other_perm_idx):
    w = _np(sd, name)
    if other_perm_idx is not None:
        w = w[other_perm_idx]
    return w.T


def axis_cost_matrix_mlp(sd_a, sd_b, ax, perms):
    n = perm_sizes_mlp(sd_a)[ax]
    C = np.zeros((n, n))
    for name, other in _ROWS[ax]:
        A = _rows_matrix(sd_a, name, None)
        B = _rows_matrix(sd_b, name, perms[other] if other else None)
        C += A @ B.T
    for name, other in _COLS[ax]:
        A = _cols_matrix(sd_a, name, None)
        B = _cols_matrix(sd_b, name, perms[other] if other else None)
        C += A @ B.T
    return C


def weight_matching_mlp(sd_a, sd_b, max_iter=100, seed=0):
    sizes = perm_sizes_mlp(sd_a)
    if sizes != perm_sizes_mlp(sd_b):
        raise ValueError("size mismatch between endpoints")
    rng = np.random.default_rng(seed)
    perms = {ax: np.arange(n) for ax, n in sizes.items()}
    names = list(sizes)
    for _ in range(max_iter):
        changed = False
        for ax in rng.permutation(names):
            C = axis_cost_matrix_mlp(sd_a, sd_b, ax, perms)
            ri, ci = linear_sum_assignment(-C)
            new = ci[np.argsort(ri)]
            if not np.array_equal(new, perms[ax]):
                perms[ax] = new
                changed = True
        if not changed:
            break
    return perms


def apply_perms_mlp(sd_b, perms):
    out = {k: v.clone() for k, v in sd_b.items()}
    p1, p2, p3 = (torch.as_tensor(perms[ax]) for ax in MLP_AXES)
    out["fc1.weight"] = out["fc1.weight"][p1]
    out["fc1.bias"] = out["fc1.bias"][p1]
    out["fc2.weight"] = out["fc2.weight"][p2][:, p1]
    out["fc2.bias"] = out["fc2.bias"][p2]
    out["fc3.weight"] = out["fc3.weight"][p3][:, p2]
    out["fc3.bias"] = out["fc3.bias"][p3]
    out["out.weight"] = out["out.weight"][:, p3]
    return out


def matching_objective_mlp(sd_a, sd_b, perms):
    aligned = apply_perms_mlp(sd_b, perms)
    return float(sum(((sd_a[k] - aligned[k]) ** 2).sum().item() for k in sd_a))


def weight_matching_mlp_restarts(sd_a, sd_b, n_restarts=10, max_iter=100):
    perms_list = [weight_matching_mlp(sd_a, sd_b, max_iter=max_iter, seed=s)
                  for s in range(n_restarts)]
    objectives = [matching_objective_mlp(sd_a, sd_b, p) for p in perms_list]
    return perms_list[int(np.argmin(objectives))], perms_list, objectives


def unit_norms_mlp(sd, ax):
    sq = np.zeros(perm_sizes_mlp(sd)[ax])
    for name, _ in _ROWS[ax]:
        sq += (_rows_matrix(sd, name, None) ** 2).sum(axis=1)
    for name, _ in _COLS[ax]:
        sq += (_cols_matrix(sd, name, None) ** 2).sum(axis=1)
    return np.sqrt(sq)


def random_perms_mlp(seed=0, hidden=512):
    rng = np.random.default_rng(seed)
    return {ax: rng.permutation(hidden) for ax in MLP_AXES}


def live_masks_mlp(sd):
    return {ax: unit_norms_mlp(sd, ax) > DEAD_NORM for ax in MLP_AXES}
