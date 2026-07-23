"""Numerical equivariance of the p4m DQN and D4-TTA.

The claims under test:
  EquivariantDQN:  Q(g·x) = g·Q(x)   exactly (up to float32), all 8 g
  plain DQN + TTA: the TTA-averaged Q-map is D4-equivariant by construction
"""

import torch

from minesweeper import d4
from minesweeper.models import DQN
from minesweeper.equivariant import EquivariantDQN


def _qmap(model, spatial, scalars, H, W):
    return model(spatial, scalars).reshape(-1, H, W)


def test_equivariant_dqn_full_d4():
    torch.manual_seed(0)
    model = EquivariantDQN().eval()
    x = torch.randn(2, 2, 6, 6)
    s = torch.rand(2, 2)
    with torch.no_grad():
        q = _qmap(model, x, s, 6, 6)
        for g in range(8):
            qg = _qmap(model, d4.apply(x, g), s, 6, 6)
            err = (qg - d4.apply(q, g)).abs().max().item()
            assert err < 1e-4, f"element {g}: max abs err {err}"


def test_equivariant_dqn_nonsquare_subgroup():
    torch.manual_seed(1)
    model = EquivariantDQN().eval()
    x = torch.randn(1, 2, 4, 7)
    s = torch.rand(1, 2)
    with torch.no_grad():
        q = _qmap(model, x, s, 4, 7)
        for g in d4.orbit_elements(square=False):
            qg = model(d4.apply(x, g), s).reshape(1, 4, 7)
            err = (qg - d4.apply(q, g)).abs().max().item()
            assert err < 1e-4, f"element {g}: max abs err {err}"


def test_plain_dqn_is_not_equivariant():
    # the control: if this ever passes, the test above is vacuous
    torch.manual_seed(2)
    model = DQN().eval()
    x = torch.randn(1, 2, 6, 6)
    s = torch.rand(1, 2)
    with torch.no_grad():
        q = _qmap(model, x, s, 6, 6)
        worst = max((_qmap(model, d4.apply(x, g), s, 6, 6) - d4.apply(q, g)).abs().max().item()
                    for g in range(1, 8))
    assert worst > 1e-3


def test_tta_is_equivariant():
    torch.manual_seed(3)
    model = DQN().eval()
    x = torch.randn(1, 2, 6, 6)
    s = torch.rand(1, 2)
    with torch.no_grad():
        q = d4.tta_q_values(model, x, s).reshape(1, 6, 6)
        for g in range(8):
            qg = d4.tta_q_values(model, d4.apply(x, g), s).reshape(1, 6, 6)
            err = (qg - d4.apply(q, g)).abs().max().item()
            assert err < 1e-4, f"element {g}: max abs err {err}"


def test_tta_is_equivariant_nonsquare():
    # value-level coverage for the rectangular branch: rotations change the
    # board shape mid-orbit and the un-mapping must still land exactly
    torch.manual_seed(4)
    model = DQN().eval()
    x = torch.randn(1, 2, 4, 7)
    s = torch.rand(1, 2)
    full = list(range(8))
    with torch.no_grad():
        q = d4.tta_q_values(model, x, s, elements=full)
        assert q.shape == (1, 28)
        q = q.reshape(1, 4, 7)
        for g in full:
            xg = d4.apply(x, g)
            Hg, Wg = xg.shape[-2:]
            qg = d4.tta_q_values(model, xg, s, elements=full).reshape(1, Hg, Wg)
            err = (qg - d4.apply(q, g)).abs().max().item()
            assert err < 1e-4, f"element {g}: max abs err {err}"
        # default subgroup path on a non-square board must run and keep shape
        assert d4.tta_q_values(model, x, s).shape == (1, 28)
