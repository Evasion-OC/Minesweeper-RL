"""The MLP zoo machinery must satisfy the same battery as the DQN arm:
permutation preserves function, matching recovers an exact permutation
(with restarts, since single-run coordinate descent is unreliable), and
self-match is identity on live units."""

import numpy as np
import torch

from wsl.mlpzoo import (MLP, MLP_AXES, apply_perms_mlp, live_masks_mlp,
                        random_perms_mlp, weight_matching_mlp,
                        weight_matching_mlp_restarts)

HIDDEN = 64  # small for test speed; the machinery is size-agnostic


def test_apply_perms_preserves_function():
    torch.manual_seed(0)
    a = MLP(hidden=HIDDEN).eval()
    sd_b = apply_perms_mlp(a.state_dict(), random_perms_mlp(seed=1, hidden=HIDDEN))
    b = MLP(hidden=HIDDEN).eval()
    b.load_state_dict(sd_b)
    x = torch.randn(8, 1, 28, 28)
    with torch.no_grad():
        assert torch.allclose(a(x), b(x), atol=1e-5)


def test_matching_recovers_permutation_with_restarts():
    torch.manual_seed(1)
    sd_a = MLP(hidden=HIDDEN).state_dict()
    sd_b = apply_perms_mlp(sd_a, random_perms_mlp(seed=2, hidden=HIDDEN))
    best, _, objs = weight_matching_mlp_restarts(sd_a, sd_b, n_restarts=10)
    aligned = apply_perms_mlp(sd_b, best)
    assert min(objs) < 1e-8
    for k in sd_a:
        assert torch.allclose(sd_a[k], aligned[k], atol=1e-6), k


def test_self_match_is_identity_on_live_units():
    torch.manual_seed(2)
    sd = MLP(hidden=HIDDEN).state_dict()
    perms = weight_matching_mlp(sd, sd)
    live = live_masks_mlp(sd)
    for ax in MLP_AXES:
        idx = np.arange(len(perms[ax]))
        assert (perms[ax][live[ax]] == idx[live[ax]]).all(), ax
