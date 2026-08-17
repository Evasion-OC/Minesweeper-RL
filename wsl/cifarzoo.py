"""CIFAR-10 width arm: a VGG-style convolutional net with BatchNorm and its
permutation machinery, mirroring wsl.align's conventions.

Six 3x3 convolutions (widths 64,64,128,128,256,256 scaled by a multiplier),
BatchNorm and ReLU after each, max-pool after each pair, global average pool
into a linear classifier. Sequential structure only: each channel axis
couples to its own conv+BN parameters and the next layer's input, so the
matching is the same coordinate-descent-over-LAPs as the other zoos, with
BatchNorm's learnable parameters included in the cost and its running
statistics permuted alongside. VGG/CIFAR-10 is a canonical Git Re-Basin
setting, which is the point of this arm.

perms[ax][i] = index into B matched to A's unit i, as everywhere else.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .align import DEAD_NORM

BASE_WIDTHS = (64, 64, 128, 128, 256, 256)
CIFAR_AXES = tuple(f"a{i}" for i in range(1, 7))

_ROWS = {f"a{i}": [(f"conv{i}.weight", f"a{i-1}" if i > 1 else None),
                   (f"bn{i}.weight", None), (f"bn{i}.bias", None)]
         for i in range(1, 7)}
_COLS = {f"a{i}": [(f"conv{i+1}.weight", f"a{i+1}")] for i in range(1, 6)}
_COLS["a6"] = [("fc.weight", None)]

AXIS_LAYER_CIFAR = {f"a{i}": f"conv{i}.weight" for i in range(1, 7)}


class CifarVGG(nn.Module):
    """conv(3->w1)...conv(w5->w6) with BN/ReLU, pools after pairs, GAP, fc."""

    def __init__(self, mult=1.0, num_classes=10):
        super().__init__()
        w = [max(1, int(round(c * mult))) for c in BASE_WIDTHS]
        self.mult = mult
        chans = [3] + w
        for i in range(1, 7):
            self.add_module(f"conv{i}", nn.Conv2d(chans[i - 1], chans[i], 3,
                                                  padding=1, bias=False))
            self.add_module(f"bn{i}", nn.BatchNorm2d(chans[i]))
        self.fc = nn.Linear(w[-1], num_classes)

    def forward(self, x):
        for i in range(1, 7):
            x = F.relu(getattr(self, f"bn{i}")(getattr(self, f"conv{i}")(x)))
            if i % 2 == 0:
                x = F.max_pool2d(x, 2)
        x = x.mean(dim=[2, 3])
        return self.fc(x)


def perm_sizes_cifar(sd):
    return {f"a{i}": int(sd[f"conv{i}.weight"].shape[0]) for i in range(1, 7)}


def _np(sd, name):
    return sd[name].detach().cpu().numpy().astype(np.float64)


def _rows_matrix(sd, name, other_perm_idx):
    w = _np(sd, name)
    if w.ndim == 1:
        return w[:, None]
    if other_perm_idx is not None:
        w = w[:, other_perm_idx]
    return w.reshape(w.shape[0], -1)


def _cols_matrix(sd, name, other_perm_idx):
    w = _np(sd, name)
    if other_perm_idx is not None:
        w = w[other_perm_idx]
    w = np.moveaxis(w, 1, 0)
    return w.reshape(w.shape[0], -1)


def axis_cost_matrix_cifar(sd_a, sd_b, ax, perms):
    n = perm_sizes_cifar(sd_a)[ax]
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


def weight_matching_cifar(sd_a, sd_b, max_iter=100, seed=0):
    sizes = perm_sizes_cifar(sd_a)
    if sizes != perm_sizes_cifar(sd_b):
        raise ValueError("width mismatch between endpoints")
    rng = np.random.default_rng(seed)
    perms = {ax: np.arange(n) for ax, n in sizes.items()}
    names = list(sizes)
    for _ in range(max_iter):
        changed = False
        for ax in rng.permutation(names):
            C = axis_cost_matrix_cifar(sd_a, sd_b, ax, perms)
            ri, ci = linear_sum_assignment(-C)
            new = ci[np.argsort(ri)]
            if not np.array_equal(new, perms[ax]):
                perms[ax] = new
                changed = True
        if not changed:
            break
    return perms


def apply_perms_cifar(sd_b, perms):
    out = {k: v.clone() for k, v in sd_b.items()}
    prev = None
    for i in range(1, 7):
        p = torch.as_tensor(perms[f"a{i}"])
        w = out[f"conv{i}.weight"][p]
        if prev is not None:
            w = w[:, prev]
        out[f"conv{i}.weight"] = w
        for suffix in ("weight", "bias", "running_mean", "running_var"):
            out[f"bn{i}.{suffix}"] = out[f"bn{i}.{suffix}"][p]
        prev = p
    out["fc.weight"] = out["fc.weight"][:, prev]
    return out


def matching_objective_cifar(sd_a, sd_b, perms):
    aligned = apply_perms_cifar(sd_b, perms)
    return float(sum(((sd_a[k] - aligned[k]) ** 2).sum().item()
                     for k in sd_a if sd_a[k].dtype.is_floating_point))


def weight_matching_cifar_restarts(sd_a, sd_b, n_restarts=10, max_iter=100):
    perms_list = [weight_matching_cifar(sd_a, sd_b, max_iter=max_iter, seed=s)
                  for s in range(n_restarts)]
    objectives = [matching_objective_cifar(sd_a, sd_b, p) for p in perms_list]
    return perms_list[int(np.argmin(objectives))], perms_list, objectives


def unit_norms_cifar(sd, ax):
    sq = np.zeros(perm_sizes_cifar(sd)[ax])
    for name, _ in _ROWS[ax]:
        sq += (_rows_matrix(sd, name, None) ** 2).sum(axis=1)
    for name, _ in _COLS[ax]:
        sq += (_cols_matrix(sd, name, None) ** 2).sum(axis=1)
    return np.sqrt(sq)


def live_masks_cifar(sd):
    return {ax: unit_norms_cifar(sd, ax) > DEAD_NORM for ax in CIFAR_AXES}


def random_perms_cifar(seed, sd):
    rng = np.random.default_rng(seed)
    return {ax: rng.permutation(n) for ax, n in perm_sizes_cifar(sd).items()}


def reset_bn_stats(model, loader, device, batches=50):
    """REPAIR in its canonical form: re-estimate BatchNorm running statistics
    of an interpolated network by forwarding training data."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
    model.train()
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= batches:
                break
            model(x.to(device))
    model.eval()
    return model
