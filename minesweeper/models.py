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
    """

    def __init__(self, num_scalars=2):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.advantage_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(64 + num_scalars, 256)
        self.value_fc2 = nn.Linear(256, 1)

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


def load_dqn(path, device=None):
    """Load a checkpoint onto device, auto-detecting the variant from its
    state dict (equivariant checkpoints carry lifting-layer keys)."""
    device = device or pick_device()
    state = torch.load(path, map_location=device)
    if any(k.startswith("lift.") for k in state):
        from .equivariant import EquivariantDQN
        model = EquivariantDQN().to(device)
    else:
        model = DQN().to(device)
    model.load_state_dict(state)
    model.eval()
    return model
