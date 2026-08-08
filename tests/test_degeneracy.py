"""The two pre-committed degeneracy statistics on spectra with known answers,
and on an existing checkpoint when one is available."""

from pathlib import Path

import numpy as np
import pytest
import torch

from wsl.degeneracy import degeneracy_stats, layer_degeneracy, AXIS_LAYER

CKPT = Path(__file__).resolve().parents[1] / "checkpoints/zoo/zoo_seed1_best.pt"


def test_flat_spectrum_is_maximally_degenerate():
    st = degeneracy_stats(np.ones(8))
    assert st["entropy"] == pytest.approx(1.0)
    assert st["min_rel_gap"] == pytest.approx(0.0)
    assert st["live_rank"] == 8


def test_rank_one_spectrum():
    st = degeneracy_stats(np.array([3.0]))
    assert st["entropy"] == 0.0
    assert np.isnan(st["min_rel_gap"])  # no consecutive pair exists
    assert st["live_rank"] == 1


def test_distinct_spectrum_gap_and_entropy():
    st = degeneracy_stats(np.array([4.0, 2.0, 1.0]))
    # gaps: (4-2)/4 = 0.5 and (2-1)/2 = 0.5
    assert st["min_rel_gap"] == pytest.approx(0.5)
    assert 0.0 < st["entropy"] < 1.0


def test_zero_tail_is_trimmed_not_counted_as_degeneracy():
    st = degeneracy_stats(np.array([1.0, 1.0, 0.0, 0.0]))
    assert st["live_rank"] == 2
    assert st["entropy"] == pytest.approx(1.0)
    assert st["min_rel_gap"] == pytest.approx(0.0)
    assert st["n"] == 4


def test_near_degenerate_pair_beats_well_separated_pair():
    near = degeneracy_stats(np.array([1.0, 0.999, 0.1]))
    far = degeneracy_stats(np.array([1.0, 0.5, 0.1]))
    assert near["min_rel_gap"] < far["min_rel_gap"]


@pytest.mark.skipif(not CKPT.exists(), reason="checkpoints not in the clone")
def test_checkpoint_layer_stats_are_sane():
    sd = torch.load(CKPT, map_location="cpu")
    stats = layer_degeneracy(sd)
    assert set(AXIS_LAYER.values()) <= set(stats)
    for name, st in stats.items():
        assert 0.0 < st["entropy"] <= 1.0, name
        assert 0.0 <= st["min_rel_gap"] < 1.0, name
        assert 0 < st["live_rank"] <= st["n"], name
