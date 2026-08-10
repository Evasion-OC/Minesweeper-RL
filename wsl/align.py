"""Permutation weight matching for the plain DQN, after Ainsworth et al.
(Git Re-Basin), specialised to this architecture.

The network's neuron-permutation symmetry group is a product of four
symmetric groups, one per hidden feature axis:

    p1: conv1 output channels (32)
    p2: conv2 output channels (64)
    p3: conv3 output channels (64)  - also permutes advantage_conv inputs and
        the first 64 input columns of value_fc1 (the global-pool features;
        the two scalar columns are fixed points)
    p4: value_fc1 hidden units (256)

weight_matching(A, B) finds permutations aligning B to A by coordinate
ascent: for each p_k in turn, build the similarity matrix summed over every
weight slice touching that axis (with all other axes permuted by the current
iterate) and solve the assignment problem exactly; repeat to a fixed point.
"""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

PERM_AXES = ("p1", "p2", "p3", "p4")

# Width-1 sizes. Everything below reads the actual sizes off the state dict
# via perm_sizes(), so width-scaled checkpoints work unchanged; this constant
# remains only as the default for random_perms and as a documented reference.
PERM_SIZES = {"p1": 32, "p2": 64, "p3": 64, "p4": 256}


def perm_sizes(sd):
    """Permutable axis sizes read off a state dict, so any channel width works."""
    return {"p1": int(sd["conv1.weight"].shape[0]),
            "p2": int(sd["conv2.weight"].shape[0]),
            "p3": int(sd["conv3.weight"].shape[0]),
            "p4": int(sd["value_fc1.weight"].shape[0])}


def pool_dim(sd):
    """Number of value_fc1 input columns carrying p3 (the globally pooled
    conv3 features). The remaining columns are the scalar inputs, which are
    fixed points of every permutation."""
    return int(sd["conv3.weight"].shape[0])

# (param, axis) pairs carrying each permutation. value_fc1.weight's input
# axis is split: columns 0:64 carry p3, columns 64:66 are fixed.
_ROW_PARAMS = {
    "p1": [("conv1.weight", None), ("conv1.bias", None)],
    "p2": [("conv2.weight", "p1"), ("conv2.bias", None)],
    "p3": [("conv3.weight", "p2"), ("conv3.bias", None)],
    "p4": [("value_fc1.weight", "p3"), ("value_fc1.bias", None)],
}
_COL_PARAMS = {
    "p1": [("conv2.weight", "p2")],
    "p2": [("conv3.weight", "p3")],
    "p3": [("advantage_conv.weight", None), ("value_fc1.weight", "p4")],
    "p4": [("value_fc2.weight", None)],
}


def _np(sd, name):
    return sd[name].detach().cpu().numpy().astype(np.float64)


def _rows_matrix(sd, name, other_perm_idx):
    """Param as (perm_axis, rest) with axis0 = permuted axis; the other
    permuted axis (if any) is aligned with other_perm_idx first."""
    w = _np(sd, name)
    if w.ndim == 1:
        return w[:, None]
    if name == "value_fc1.weight":
        w = w.copy()
        if other_perm_idx is not None:
            d = pool_dim(sd)
            w[:, :d] = w[:, :d][:, other_perm_idx]
        return w.reshape(w.shape[0], -1)
    if other_perm_idx is not None:
        w = w[:, other_perm_idx]
    return w.reshape(w.shape[0], -1)


def _cols_matrix(sd, name, other_perm_idx):
    """Param as (perm_axis, rest) with the INPUT axis first."""
    w = _np(sd, name)
    if name == "value_fc1.weight":
        w = w[:, :pool_dim(sd)]
    if other_perm_idx is not None:
        w = w[other_perm_idx]
    w = np.moveaxis(w, 1, 0)
    return w.reshape(w.shape[0], -1)


# Exact assignment ties require a unit's objective contribution (~norm^2) to
# sit at or below float64 resolution of the total cost (~1e-14 for O(100)
# objectives), i.e. norm below ~1e-7. Observed tie-class units are <= 4e-9;
# the next units up the norm distribution are >= ~1e-4 and never tie.
DEAD_NORM = 1e-7


def unit_norms(sd, p):
    """Per-unit norm over every weight slice carrying permutation axis p.
    Units below DEAD_NORM are numerically dead: an exact zero-cost tie class
    inside which any matching is arbitrary."""
    sq = np.zeros(perm_sizes(sd)[p])
    for name, _ in _ROW_PARAMS[p]:
        sq += (_rows_matrix(sd, name, None) ** 2).sum(axis=1)
    for name, _ in _COL_PARAMS[p]:
        sq += (_cols_matrix(sd, name, None) ** 2).sum(axis=1)
    return np.sqrt(sq)


def axis_cost_matrix(sd_a, sd_b, p, perms):
    """Similarity matrix for axis p (A units x B units), summed over every
    weight slice carrying p, with B's other axes aligned by `perms`."""
    n = perm_sizes(sd_a)[p]
    C = np.zeros((n, n))
    for name, other in _ROW_PARAMS[p]:
        A = _rows_matrix(sd_a, name, None)
        B = _rows_matrix(sd_b, name, perms[other] if other else None)
        C += A @ B.T
    for name, other in _COL_PARAMS[p]:
        A = _cols_matrix(sd_a, name, None)
        B = _cols_matrix(sd_b, name, perms[other] if other else None)
        C += A @ B.T
    return C


def weight_matching(sd_a, sd_b, max_iter=100, seed=0):
    """Returns perms dict: perms[p][i] = index into B matched to A's unit i."""
    sizes = perm_sizes(sd_a)
    if sizes != perm_sizes(sd_b):
        raise ValueError(f"width mismatch: A has {sizes}, B has {perm_sizes(sd_b)}")
    rng = np.random.default_rng(seed)
    perms = {p: np.arange(n) for p, n in sizes.items()}
    names = list(sizes)
    for _ in range(max_iter):
        changed = False
        for p in rng.permutation(names):
            C = axis_cost_matrix(sd_a, sd_b, p, perms)
            ri, ci = linear_sum_assignment(-C)
            new = ci[np.argsort(ri)]
            if not np.array_equal(new, perms[p]):
                perms[p] = new
                changed = True
        if not changed:
            break
    return perms


def apply_perms(sd_b, perms):
    """Return a copy of B's state dict re-indexed so it is aligned to A.
    The permuted network computes exactly the same function as B."""
    out = {k: v.clone() for k, v in sd_b.items()}
    p1, p2, p3, p4 = (torch.as_tensor(perms[p]) for p in ("p1", "p2", "p3", "p4"))

    out["conv1.weight"] = out["conv1.weight"][p1]
    out["conv1.bias"] = out["conv1.bias"][p1]
    out["conv2.weight"] = out["conv2.weight"][p2][:, p1]
    out["conv2.bias"] = out["conv2.bias"][p2]
    out["conv3.weight"] = out["conv3.weight"][p3][:, p2]
    out["conv3.bias"] = out["conv3.bias"][p3]
    out["advantage_conv.weight"] = out["advantage_conv.weight"][:, p3]
    d = pool_dim(sd_b)
    w = out["value_fc1.weight"]
    w = torch.cat([w[:, :d][:, p3], w[:, d:]], dim=1)
    out["value_fc1.weight"] = w[p4]
    out["value_fc1.bias"] = out["value_fc1.bias"][p4]
    out["value_fc2.weight"] = out["value_fc2.weight"][:, p4]
    return out


def matching_objective(sd_a, sd_b, perms):
    """||A - P(B)||^2, the quantity weight matching minimises. Lower is a
    better alignment; equal values mean the correspondences are tied."""
    aligned = apply_perms(sd_b, perms)
    return float(sum(((sd_a[k] - aligned[k]) ** 2).sum().item() for k in sd_a))


def weight_matching_restarts(sd_a, sd_b, n_restarts=10, max_iter=100):
    """Coordinate descent from several axis orderings.

    Single-run weight matching is a heuristic on an NP-hard problem and lands
    in a local optimum often: on exactly-permuted copies of this architecture,
    where a unique global optimum exists by construction, one run recovers it
    about 40 percent of the time and best-of-10 recovers it always. Anything
    that reads a converged matching should therefore use restarts.

    Returns (best_perms, perms_list, objectives).
    """
    perms_list = [weight_matching(sd_a, sd_b, max_iter=max_iter, seed=s)
                  for s in range(n_restarts)]
    objectives = [matching_objective(sd_a, sd_b, p) for p in perms_list]
    return perms_list[int(np.argmin(objectives))], perms_list, objectives


def random_perms(seed=0, sd=None):
    """Random permutation per axis. Sizes come from `sd` when given, else the
    width-1 defaults."""
    rng = np.random.default_rng(seed)
    sizes = perm_sizes(sd) if sd is not None else PERM_SIZES
    return {p: rng.permutation(n) for p, n in sizes.items()}


def interpolate(sd_a, sd_b, lam):
    """(1-lam) * A + lam * B, elementwise."""
    return {k: (1.0 - lam) * sd_a[k] + lam * sd_b[k] for k in sd_a}
