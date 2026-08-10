"""Q networks.

DQN is the original fully convolutional dueling architecture, made explicitly
board-size agnostic: no layer has a size-dependent weight shape (convs +
1x1 advantage head + globally pooled value head), so one set of weights runs
on any (H, W). Parameter names match the monolith, so shipped checkpoints
load unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def pick_device(name=None):
    """CUDA > Apple-Silicon MPS > CPU, or force by name."""
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available() and mps_backend.is_built():
        return torch.device("mps")
    return torch.device("cpu")


BASE_CHANNELS = (32, 64, 64, 256)  # conv1, conv2, conv3, value_fc1 hidden


class DQN(nn.Module):
    """Dueling DQN. One Q value per cell, any board size.

    Inputs:
        spatial: (B, 2, H, W) - covered mask, clue plane
        scalars: (B, 2) - covered_ratio, mine_density
    Output:
        Q values of shape (B, H*W), row-major to match the env's action encoding.

    Unlike the monolith, the constructor takes no (height, width, num_actions):
    the output size is H*W of whatever input it sees, which is what allows one
    checkpoint to be evaluated across board sizes.

    `width` scales every hidden channel count by an integer multiplier, for the
    width sweep. width=1 reproduces the shipped architecture exactly, so
    existing checkpoints load unchanged.
    """

    def __init__(self, num_scalars=2, width=1):
        super().__init__()
        if width < 1 or width != int(width):
            raise ValueError(f"width must be a positive integer, got {width!r}")
        c1, c2, c3, h = (c * int(width) for c in BASE_CHANNELS)
        self.width = int(width)
        self.conv1 = nn.Conv2d(2, c1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, padding=1)
        self.advantage_conv = nn.Conv2d(c3, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(c3 + num_scalars, h)
        self.value_fc2 = nn.Linear(h, 1)

    def forward(self, spatial, scalars):
        x = F.relu(self.conv1(spatial))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        adv = self.advantage_conv(x)               # (B, 1, H, W)
        adv = adv.flatten(1)                       # (B, H*W)
        adv = adv - adv.mean(dim=1, keepdim=True)  # zero-mean (dueling)

        v = x.mean(dim=[2, 3])                     # (B, 64) global avg pool
        v = torch.cat([v, scalars], dim=1)
        v = F.relu(self.value_fc1(v))
        v = self.value_fc2(v)                      # (B, 1)

        return v + adv


def infer_width(state):
    """Width multiplier of a plain-DQN state dict, from conv1's output channels."""
    out = state["conv1.weight"].shape[0]
    width, rem = divmod(out, BASE_CHANNELS[0])
    if rem or width < 1:
        raise ValueError(f"conv1 has {out} channels, not a multiple of {BASE_CHANNELS[0]}")
    return width


def load_dqn(path, device=None):
    """Load a checkpoint onto device, auto-detecting the variant from its
    state dict (equivariant checkpoints carry lifting-layer keys) and, for
    plain checkpoints, the width multiplier."""
    device = device or pick_device()
    state = torch.load(path, map_location=device)
    if any(k.startswith("lift.") for k in state):
        from .equivariant import EquivariantDQN
        model = EquivariantDQN().to(device)
    else:
        model = DQN(width=infer_width(state)).to(device)
    model.load_state_dict(state)
    model.eval()
    return model
