"""Pre-committed alignment instability probes for the weight-matching study.

Three probes per checkpoint pair, each resolved per permutation axis,
defined in advance (no probe shopping):

  perturbation sensitivity: add Gaussian noise of fixed relative Frobenius
      norm (eps per parameter tensor) to endpoint B, re-run matching, and
      report the fraction of live assignments that changed, averaged over
      noise draws.
  assignment gap: at the converged matching, the gap between the best and
      second-best total assignment cost on the axis's live submatrix. The
      same measurement as the merging brief's Section 1.5 filter
      identifiability gap, by design; reported raw and relative to |best|.
  restart disagreement: weight matching's only stochastic component is the
      coordinate-descent axis order (the `seed` argument), so restarts vary
      that. Reported as the fraction of live units whose assignment is not
      unanimous across restarts. The realised-barrier variance over the
      same restarts needs rollouts and lives in the driver script.

Dead units (norm below wsl.align.DEAD_NORM, an exact zero-cost tie class;
see scripts/wsl_nulls.py) are excluded from every count. Churn inside the
dead class is trivially arbitrary and would inflate every probe.
"""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .align import (PERM_AXES, DEAD_NORM, unit_norms, axis_cost_matrix,
                    weight_matching)

NOISE_EPS = 1e-2
NOISE_DRAWS = 10
RESTART_SEEDS = 10


def live_masks(sd):
    return {p: unit_norms(sd, p) > DEAD_NORM for p in PERM_AXES}


def _eligible(live_a, live_b, perm):
    """A-side units whose base assignment pairs two live units."""
    return live_a & live_b[perm]


def perturbation_sensitivity(sd_a, sd_b, base_perms=None, eps=NOISE_EPS,
                             draws=NOISE_DRAWS, rng_seed=0):
    """Fraction of live assignments changed by re-matching against a
    perturbed B, per axis, averaged over noise draws."""
    if base_perms is None:
        base_perms = weight_matching(sd_a, sd_b)
    live_a, live_b = live_masks(sd_a), live_masks(sd_b)
    gen = torch.Generator().manual_seed(rng_seed)
    churn = {p: [] for p in PERM_AXES}
    for _ in range(draws):
        sd_bp = {}
        for k, v in sd_b.items():
            noise = torch.randn(v.shape, generator=gen, dtype=v.dtype)
            n = noise.norm()
            if n > 0:
                noise *= eps * v.norm() / n
            sd_bp[k] = v + noise
        perms = weight_matching(sd_a, sd_bp)
        for p in PERM_AXES:
            el = _eligible(live_a[p], live_b[p], base_perms[p])
            churn[p].append(float(np.mean(perms[p][el] != base_perms[p][el]))
                            if el.any() else 0.0)
    return {p: {"churn_mean": float(np.mean(churn[p])),
                "churn_std": float(np.std(churn[p])),
                "eps": eps, "draws": draws} for p in PERM_AXES}


def lap_gap(C):
    """(best, second best) total assignment value for similarity matrix C.
    Second best forbids, in turn, each edge the best assignment uses."""
    ri, ci = linear_sum_assignment(-C)
    v1 = float(C[ri, ci].sum())
    v2 = -np.inf
    for i, j in zip(ri, ci):
        C2 = C.copy()
        C2[i, j] = -1e18
        r2, c2 = linear_sum_assignment(-C2)
        v2 = max(v2, float(C2[r2, c2].sum()))
    return v1, v2


def assignment_gap(sd_a, sd_b, perms=None):
    """Best minus second-best assignment cost per axis, on the live
    submatrix at the converged matching."""
    if perms is None:
        perms = weight_matching(sd_a, sd_b)
    live_a, live_b = live_masks(sd_a), live_masks(sd_b)
    out = {}
    for p in PERM_AXES:
        C = axis_cost_matrix(sd_a, sd_b, p, perms)
        C = C[np.ix_(live_a[p], live_b[p])]
        v1, v2 = lap_gap(C)
        out[p] = {"gap": v1 - v2, "rel_gap": (v1 - v2) / max(abs(v1), 1e-12),
                  "best": v1, "second": v2,
                  "n_live": [int(live_a[p].sum()), int(live_b[p].sum())]}
    return out


def restart_disagreement(sd_a, sd_b, seeds=tuple(range(RESTART_SEEDS)),
                         base_perms=None):
    """Fraction of live units whose assignment is not unanimous across
    coordinate-descent order seeds. Returns (stats, perm_list) so callers
    can also evaluate realised barriers per restart."""
    perm_list = []
    for s in seeds:
        if s == seeds[0] and base_perms is not None:
            perm_list.append(base_perms)
        else:
            perm_list.append(weight_matching(sd_a, sd_b, seed=s))
    live_a, live_b = live_masks(sd_a), live_masks(sd_b)
    stats = {}
    for p in PERM_AXES:
        targets = np.stack([perms[p] for perms in perm_list])
        el = _eligible(live_a[p], live_b[p], targets[0])
        varies = (targets != targets[0]).any(axis=0)
        stats[p] = {"disagree_frac": float(np.mean(varies[el])) if el.any() else 0.0,
                    "n_restarts": len(perm_list)}
    return stats, perm_list
