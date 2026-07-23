"""Alignment sanity: permuting a network and matching it back must recover
the original exactly, and the permuted network must compute the same function."""

import torch

from minesweeper.models import DQN
from wsl.align import weight_matching, apply_perms, random_perms, interpolate


def test_apply_perms_preserves_function():
    torch.manual_seed(0)
    a = DQN().eval()
    sd_b = apply_perms(a.state_dict(), random_perms(seed=1))
    b = DQN().eval()
    b.load_state_dict(sd_b)
    x = torch.randn(4, 2, 8, 8)
    s = torch.rand(4, 2)
    with torch.no_grad():
        assert torch.allclose(a(x, s), b(x, s), atol=1e-5)


def test_weight_matching_recovers_permutation():
    torch.manual_seed(0)
    a = DQN()
    sd_a = a.state_dict()
    true = random_perms(seed=2)
    # B is A scrambled by `true`; matching B to A must undo it exactly
    sd_b = apply_perms(sd_a, true)
    found = weight_matching(sd_a, sd_b)
    aligned = apply_perms(sd_b, found)
    for k in sd_a:
        assert torch.allclose(sd_a[k], aligned[k], atol=1e-6), k


def test_midpoint_of_aligned_permuted_copy_is_identity():
    torch.manual_seed(0)
    a = DQN()
    sd_a = a.state_dict()
    sd_b = apply_perms(sd_a, random_perms(seed=3))
    found = weight_matching(sd_a, sd_b)
    aligned = apply_perms(sd_b, found)
    mid = interpolate(sd_a, aligned, 0.5)
    for k in sd_a:
        assert torch.allclose(sd_a[k], mid[k], atol=1e-6), k
