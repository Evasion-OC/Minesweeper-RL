"""Width-scaled models: width=1 must reproduce the shipped architecture
exactly (so existing checkpoints keep loading), and the alignment machinery
must work unchanged at any width."""

from pathlib import Path

import pytest
import torch

from minesweeper.models import BASE_CHANNELS, DQN, infer_width, load_dqn
from wsl.align import apply_perms, perm_sizes, pool_dim, random_perms, weight_matching

CKPT = Path(__file__).resolve().parents[1] / "checkpoints/zoo/zoo_seed1_best.pt"


def test_width_one_is_the_shipped_architecture():
    sd = DQN().state_dict()
    assert sd["conv1.weight"].shape == (32, 2, 3, 3)
    assert sd["conv2.weight"].shape == (64, 32, 3, 3)
    assert sd["conv3.weight"].shape == (64, 64, 3, 3)
    assert sd["value_fc1.weight"].shape == (256, 66)
    assert sd["value_fc2.weight"].shape == (1, 256)


@pytest.mark.parametrize("width", [1, 2, 4])
def test_shapes_and_inference_scale_with_width(width):
    sd = DQN(width=width).state_dict()
    c1, c2, c3, h = (c * width for c in BASE_CHANNELS)
    assert infer_width(sd) == width
    assert perm_sizes(sd) == {"p1": c1, "p2": c2, "p3": c3, "p4": h}
    assert pool_dim(sd) == c3
    # value_fc1 takes the pooled conv3 features plus the two scalars
    assert sd["value_fc1.weight"].shape == (h, c3 + 2)


@pytest.mark.parametrize("width", [1, 2])
def test_forward_is_board_size_agnostic_at_any_width(width):
    model = DQN(width=width).eval()
    for rows, cols in ((8, 8), (5, 7)):
        q = model(torch.randn(2, 2, rows, cols), torch.rand(2, 2))
        assert q.shape == (2, rows * cols)


@pytest.mark.parametrize("width", [2, 4])
def test_permutation_preserves_function_and_is_recovered(width):
    torch.manual_seed(0)
    a = DQN(width=width).eval()
    sd_a = a.state_dict()
    true = random_perms(seed=5, sd=sd_a)
    sd_b = apply_perms(sd_a, true)
    b = DQN(width=width).eval()
    b.load_state_dict(sd_b)
    x, s = torch.randn(4, 2, 8, 8), torch.rand(4, 2)
    with torch.no_grad():
        assert torch.allclose(a(x, s), b(x, s), atol=1e-5)
    recovered = apply_perms(sd_b, weight_matching(sd_a, sd_b))
    for k in sd_a:
        assert torch.allclose(sd_a[k], recovered[k], atol=1e-6), k


def test_matching_across_widths_is_refused():
    sd_a, sd_b = DQN(width=1).state_dict(), DQN(width=2).state_dict()
    with pytest.raises(ValueError, match="width mismatch"):
        weight_matching(sd_a, sd_b)


def test_invalid_width_is_refused():
    for bad in (0, -1, 1.5):
        with pytest.raises(ValueError):
            DQN(width=bad)


@pytest.mark.skipif(not CKPT.exists(), reason="checkpoints not in the clone")
def test_existing_checkpoint_still_loads_at_inferred_width():
    sd = torch.load(CKPT, map_location="cpu")
    assert infer_width(sd) == 1
    model = load_dqn(CKPT, device=torch.device("cpu"))
    assert model.width == 1
