"""Pre-committed per-layer degeneracy statistics for the alignment study.

Two statistics, chosen in advance of any correlation analysis (the paper
brief forbids statistic shopping). Both are computed on the live spectrum,
the singular values at or above LIVE_TOL times the largest one; the zero
tail below it is exact degeneracy of a trivial kind (dead units, see
scripts/wsl_nulls.py) and is reported separately as rank deficiency,
mirroring the live-unit quotient in wsl.instability.

  spectral entropy: entropy of the normalised squared spectrum,
      p_i = sigma_i^2 / sum_j sigma_j^2, H = -sum p_i ln p_i / ln(r).
      1 means maximally flat (degenerate), 0 means rank one.
  min relative gap: min_i (sigma_i - sigma_{i+1}) / sigma_i. Small means
      two nearly equal singular values somewhere; Wedin's sin-theta
      theorem bounds singular subspace sensitivity by 1/gap, so small gap
      predicts ill-determined matching directions.
"""

import numpy as np

from .spectra import layer_spectra, relative_gaps

LIVE_TOL = 1e-6

# permutation axis -> the weight matrix whose output units it permutes
AXIS_LAYER = {"p1": "conv1.weight", "p2": "conv2.weight",
              "p3": "conv3.weight", "p4": "value_fc1.weight"}


def degeneracy_stats(sigma):
    """Both pre-committed statistics for one descending spectrum."""
    s = np.asarray(sigma, dtype=np.float64)
    if s.size == 0 or s[0] <= 0:
        return {"entropy": float("nan"), "min_rel_gap": float("nan"),
                "live_rank": 0, "n": int(s.size)}
    live = s[s >= LIVE_TOL * s[0]]
    r = live.size
    p = live ** 2 / (live ** 2).sum()
    entropy = 0.0 if r == 1 else float(-(p * np.log(p)).sum() / np.log(r))
    gaps = relative_gaps(live)
    min_rel_gap = float(gaps.min()) if gaps.size else float("nan")
    return {"entropy": entropy, "min_rel_gap": min_rel_gap,
            "live_rank": int(r), "n": int(s.size)}


def layer_degeneracy(sd, layers=None):
    """{layer: degeneracy_stats} for a checkpoint state dict."""
    return {name: degeneracy_stats(sig)
            for name, sig in layer_spectra(sd, layers).items()}
