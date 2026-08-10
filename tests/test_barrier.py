"""The barrier convention, on curves whose answer is known by hand."""

import numpy as np
import pytest
import torch

from minesweeper.models import DQN
from wsl.align import PERM_AXES, apply_perms, random_perms, weight_matching_restarts
from wsl.barrier import (barrier, leave_one_axis_perms,
                         per_axis_barrier_contribution)

LAMS = [0.0, 0.25, 0.5, 0.75, 1.0]


def test_barrier_is_measured_against_the_better_endpoint():
    # endpoints 0.10 and 0.20, dip to 0.02 at the midpoint
    b = barrier([0.10, 0.06, 0.02, 0.08, 0.20], LAMS)
    assert b["barrier"] == pytest.approx(0.18)      # 0.20 - 0.02, not 0.10 - 0.02
    assert b["argmax_lam"] == pytest.approx(0.5)
    assert b["rel_barrier"] == pytest.approx(0.9)   # 0.18 / 0.20


def test_no_barrier_when_the_interior_beats_both_endpoints():
    b = barrier([0.10, 0.30, 0.40, 0.30, 0.20], LAMS)
    assert b["barrier"] < 0  # negative: the interior is better, not worse


def test_lower_is_better_flips_the_sign():
    # a loss-like metric: endpoints 1.0 and 2.0, interior spikes to 5.0
    b = barrier([1.0, 3.0, 5.0, 3.0, 2.0], LAMS, higher_is_better=False)
    assert b["barrier"] == pytest.approx(4.0)  # 5.0 - min(1.0, 2.0)
    assert b["argmax_lam"] == pytest.approx(0.5)


def test_endpoints_are_required():
    with pytest.raises(ValueError):
        barrier([0.1, 0.2, 0.3], [0.25, 0.5, 0.75])


def test_leave_one_axis_resets_only_that_axis():
    perms = random_perms(seed=7)
    held = leave_one_axis_perms(perms, "p2")
    assert np.array_equal(held["p2"], np.arange(len(perms["p2"])))
    for p in PERM_AXES:
        if p != "p2":
            assert np.array_equal(held[p], perms[p])
    assert not np.array_equal(perms["p2"], held["p2"])  # original untouched


def test_permuted_copy_has_no_barrier_and_every_axis_contributes():
    """B is an exact permuted copy of A. Aligning everything makes the
    interpolation path constant, so there is no barrier; leaving any one
    axis unaligned must not improve on that."""
    torch.manual_seed(0)
    a = DQN().eval()
    sd_a = a.state_dict()
    sd_b = apply_perms(sd_a, random_perms(seed=11))
    # restarts, not a single run: seed 11 is one of the cases where a single
    # coordinate-descent pass lands in a local optimum and never recovers the
    # true permutation, even though it is unique here by construction
    perms, _, _ = weight_matching_restarts(sd_a, sd_b)

    def curve_fn(x, y, lams):
        # a scalar summary of the interpolated weights: mean squared distance
        # to endpoint A, which is zero along a perfectly aligned path
        from wsl.align import interpolate
        vals = []
        for lam in lams:
            mid = interpolate(x, y, lam)
            vals.append(float(sum(((mid[k] - x[k]) ** 2).sum() for k in x)))
        return vals

    res = per_axis_barrier_contribution(sd_a, sd_b, perms, curve_fn, LAMS,
                                        higher_is_better=False)
    assert res["aligned"]["barrier"] == pytest.approx(0.0, abs=1e-6)
    for p in PERM_AXES:
        assert res["per_axis"][p]["contribution"] >= -1e-6, p
