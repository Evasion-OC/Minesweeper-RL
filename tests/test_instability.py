"""Instability probes: exact answers on constructed cases, invariances on
random networks, and sanity on existing checkpoints when available."""

from pathlib import Path

import numpy as np
import pytest
import torch

from minesweeper.models import DQN
from wsl.align import PERM_SIZES, apply_perms, random_perms
from wsl.instability import (lap_gap, assignment_gap, perturbation_sensitivity,
                             restart_disagreement)

CKPT_DIR = Path(__file__).resolve().parents[1] / "checkpoints/zoo"


def test_lap_gap_known_matrix():
    # best is the diagonal (6). Any deviation must derange at least two
    # rows; the cheapest keeps the 3 and zeroes the other two rows, so the
    # second-best total is 3 (verified by enumerating all six assignments).
    C = np.diag([3.0, 2.0, 1.0])
    v1, v2 = lap_gap(C)
    assert v1 == pytest.approx(6.0)
    assert v2 == pytest.approx(3.0)


def test_lap_gap_degenerate_matrix_has_zero_gap():
    # two identical columns: swapping their assignments costs nothing
    C = np.array([[5.0, 5.0, 0.0], [4.0, 4.0, 0.0], [0.0, 0.0, 3.0]])
    v1, v2 = lap_gap(C)
    assert v1 == pytest.approx(v2)


def test_zero_noise_means_zero_churn():
    torch.manual_seed(0)
    sd_a = DQN().state_dict()
    sd_b = DQN().state_dict()
    out = perturbation_sensitivity(sd_a, sd_b, eps=0.0, draws=2)
    for p in PERM_SIZES:
        assert out[p]["churn_mean"] == 0.0


def test_permuted_copy_is_stable_under_probes():
    # B is an exact permuted copy of A: the optimum is strict, so restarts
    # agree and the assignment gap is positive on every axis
    torch.manual_seed(1)
    sd_a = DQN().state_dict()
    sd_b = apply_perms(sd_a, random_perms(seed=4))
    stats, perm_list = restart_disagreement(sd_a, sd_b, seeds=(0, 1, 2))
    assert len(perm_list) == 3
    gaps = assignment_gap(sd_a, sd_b, perms=perm_list[0])
    for p in PERM_SIZES:
        assert stats[p]["disagree_frac"] == 0.0
        assert gaps[p]["gap"] > 0.0
        assert 0.0 < gaps[p]["rel_gap"] < 1.0


@pytest.mark.skipif(not (CKPT_DIR / "zoo_seed1_best.pt").exists(),
                    reason="checkpoints not in the clone")
def test_checkpoint_pair_probes_run_and_quotient_dead_units():
    sd_a = torch.load(CKPT_DIR / "zoo_seed1_best.pt", map_location="cpu")
    sd_b = torch.load(CKPT_DIR / "zoo_seed2_best.pt", map_location="cpu")
    gaps = assignment_gap(sd_a, sd_b)
    for p in PERM_SIZES:
        n_a, n_b = gaps[p]["n_live"]
        assert 0 < n_a <= PERM_SIZES[p] and 0 < n_b <= PERM_SIZES[p]
        assert gaps[p]["gap"] >= 0.0
    # trained pairs carry dead value_fc1 units; the live count must drop
    assert gaps["p4"]["n_live"][0] < PERM_SIZES["p4"]
