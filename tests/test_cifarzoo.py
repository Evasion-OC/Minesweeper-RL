"""The CIFAR zoo machinery, same battery as the other arms: permutation
preserves function (BatchNorm statistics included), matching recovers an
exact permutation with restarts, self-match is identity on live units, and
widths scale."""

import numpy as np
import pytest
import torch

from wsl.cifarzoo import (CIFAR_AXES, CifarVGG, apply_perms_cifar,
                          live_masks_cifar, perm_sizes_cifar,
                          random_perms_cifar, weight_matching_cifar,
                          weight_matching_cifar_restarts)

MULT = 0.125  # widths 8,8,16,16,32,32: small enough for exhaustive-ish tests


def fresh(mult=MULT, seed=0):
    torch.manual_seed(seed)
    m = CifarVGG(mult=mult).eval()
    # nudge BN stats off their init so permuting them is actually tested
    with torch.no_grad():
        m(torch.randn(16, 3, 32, 32))
    m.eval()
    return m


def test_apply_perms_preserves_function_including_bn():
    a = fresh(seed=0)
    sd_b = apply_perms_cifar(a.state_dict(), random_perms_cifar(3, a.state_dict()))
    b = CifarVGG(mult=MULT).eval()
    b.load_state_dict(sd_b)
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(a(x), b(x), atol=1e-5)


def test_matching_recovers_permutation_with_restarts():
    a = fresh(seed=1)
    sd_a = a.state_dict()
    sd_b = apply_perms_cifar(sd_a, random_perms_cifar(4, sd_a))
    best, _, objs = weight_matching_cifar_restarts(sd_a, sd_b, n_restarts=10)
    assert min(objs) < 1e-8
    aligned = apply_perms_cifar(sd_b, best)
    for k, v in sd_a.items():
        if v.dtype.is_floating_point:
            assert torch.allclose(v, aligned[k], atol=1e-6), k


def test_self_match_is_identity_on_live_units():
    a = fresh(seed=2)
    sd = a.state_dict()
    perms = weight_matching_cifar(sd, sd)
    live = live_masks_cifar(sd)
    for ax in CIFAR_AXES:
        idx = np.arange(len(perms[ax]))
        assert (perms[ax][live[ax]] == idx[live[ax]]).all(), ax


@pytest.mark.parametrize("mult", [0.25, 1.0])
def test_width_multiplier_scales_axes(mult):
    sd = CifarVGG(mult=mult).state_dict()
    sizes = perm_sizes_cifar(sd)
    assert sizes["a1"] == int(round(64 * mult))
    assert sizes["a6"] == int(round(256 * mult))
    with pytest.raises(ValueError):
        weight_matching_cifar(sd, CifarVGG(mult=mult / 2).state_dict())
