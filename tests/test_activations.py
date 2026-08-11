"""Activation covariance spectra: shape, sanity, and the duplicate-unit
degeneracy mechanism."""

import numpy as np
import pytest
import torch

from minesweeper.models import DQN
from wsl.activations import AXIS_MODULE, activation_spectra

STATES = (torch.randn(64, 2, 8, 8), torch.rand(64, 2))
DEVICE = torch.device("cpu")


def test_spectra_are_descending_and_finite():
    torch.manual_seed(0)
    model = DQN().eval()
    spectra = activation_spectra(model, STATES, DEVICE)
    assert set(spectra) == set(AXIS_MODULE)
    for ax, s in spectra.items():
        assert np.all(np.isfinite(s)), ax
        assert np.all(s[:-1] >= s[1:] - 1e-12), ax
        assert np.all(s >= 0), ax


def test_repeat_call_is_deterministic_and_hooks_are_removed():
    torch.manual_seed(1)
    model = DQN().eval()
    s1 = activation_spectra(model, STATES, DEVICE)
    s2 = activation_spectra(model, STATES, DEVICE)
    for ax in AXIS_MODULE:
        assert np.allclose(s1[ax], s2[ax])
    assert not model.conv1._forward_hooks  # no hook leak


def test_duplicated_filter_creates_exact_activation_degeneracy():
    """Copying one conv1 filter onto another makes two activation rows
    identical, so the covariance loses a rank: the exact-degeneracy
    mechanism the paper's claim is built on, visible in activations."""
    torch.manual_seed(2)
    base = DQN().eval()
    spectra_before = activation_spectra(base, STATES, DEVICE)

    dup = DQN().eval()
    dup.load_state_dict(base.state_dict())
    with torch.no_grad():
        dup.conv1.weight[1] = dup.conv1.weight[0]
        dup.conv1.bias[1] = dup.conv1.bias[0]
    spectra_after = activation_spectra(dup, STATES, DEVICE)

    def live_rank(s):
        return int((s >= 1e-6 * s[0]).sum())

    assert live_rank(spectra_after["p1"]) == live_rank(spectra_before["p1"]) - 1
