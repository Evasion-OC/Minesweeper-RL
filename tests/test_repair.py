"""REPAIR: corrected statistics must hit their targets, and repairing an
endpoint against itself must be (near) a no-op."""

import pytest
import torch

from minesweeper.models import DQN
from wsl.align import interpolate
from wsl.repair import HIDDEN, STD_FLOOR, repair, unit_stats

STATES = (torch.randn(256, 2, 8, 8), torch.rand(256, 2))
DEVICE = torch.device("cpu")


def make_model(sd):
    m = DQN().to(DEVICE)
    m.load_state_dict(sd)
    m.eval()
    return m


def test_repair_moves_stats_to_the_interpolated_targets():
    torch.manual_seed(0)
    sd_a, sd_b = DQN().state_dict(), DQN().state_dict()
    stats_a = unit_stats(make_model(sd_a), STATES, DEVICE, HIDDEN)
    stats_b = unit_stats(make_model(sd_b), STATES, DEVICE, HIDDEN)
    mid = interpolate(sd_a, sd_b, 0.5)
    fixed = repair(mid, stats_a, stats_b, 0.5, make_model, STATES, DEVICE)
    got = unit_stats(make_model(fixed), STATES, DEVICE, HIDDEN)
    for layer in HIDDEN:
        m_t = 0.5 * (stats_a[layer][0] + stats_b[layer][0])
        s_t = 0.5 * (stats_a[layer][1] + stats_b[layer][1])
        live = s_t > STD_FLOOR
        assert torch.allclose(got[layer][0][live], m_t[live],
                              atol=5e-3, rtol=5e-2), layer
        assert torch.allclose(got[layer][1][live], s_t[live],
                              atol=5e-3, rtol=5e-2), layer


def test_repairing_an_endpoint_is_a_near_noop():
    torch.manual_seed(1)
    sd_a, sd_b = DQN().state_dict(), DQN().state_dict()
    stats_a = unit_stats(make_model(sd_a), STATES, DEVICE, HIDDEN)
    stats_b = unit_stats(make_model(sd_b), STATES, DEVICE, HIDDEN)
    fixed = repair({k: v.clone() for k, v in sd_a.items()},
                   stats_a, stats_b, 0.0, make_model, STATES, DEVICE)
    for k, v in sd_a.items():
        assert torch.allclose(fixed[k], v, atol=1e-4, rtol=1e-3), k


def test_repair_changes_the_function_at_the_midpoint():
    # not a no-op in general: that is the entire point of the arm
    torch.manual_seed(2)
    sd_a, sd_b = DQN().state_dict(), DQN().state_dict()
    stats_a = unit_stats(make_model(sd_a), STATES, DEVICE, HIDDEN)
    stats_b = unit_stats(make_model(sd_b), STATES, DEVICE, HIDDEN)
    mid = interpolate(sd_a, sd_b, 0.5)
    fixed = repair(mid, stats_a, stats_b, 0.5, make_model, STATES, DEVICE)
    diff = max((fixed[k] - mid[k]).abs().max().item() for k in mid)
    assert diff > 1e-3
