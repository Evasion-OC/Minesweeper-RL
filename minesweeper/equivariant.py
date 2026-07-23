"""Hand-rolled p4m (D4) group-equivariant dueling DQN.

Group convolutions after Cohen & Welling (2016), specialised to D4 on the
board grid, with no external dependency: each layer materialises one big
Conv2d weight from a small parameter tensor plus the group's kernel
transforms, so a layer is a single conv2d call.

Structure:
  lifting conv:  (B, 2, H, W)      -> (B, C, 8, H, W)
  group convs:   (B, C, 8, H, W)   -> (B, C', 8, H, W)
  group pool:    mean over the group axis -> scalar field (B, C', H, W)
  advantage:     1x1 conv -> per-cell Q-map (equivariant: Q(g·x) = g·Q(x))
  value:         global spatial mean + scalars -> MLP (invariant)

Biases are shared across the group axis, which is required for equivariance.
tests/test_equivariance.py checks the whole network numerically on all 8
group elements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import d4


def _transform_kernel(weight, g):
    """Apply group element g to the spatial dims of a conv kernel."""
    return d4.apply(weight, g)


class LiftingConv(nn.Module):
    """Z2 -> D4 lifting layer. out[:, :, g] = corr(x, g·psi)."""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.out_channels = out_channels
        self.padding = padding
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x):
        B = x.shape[0]
        big = torch.cat([_transform_kernel(self.weight, g) for g in range(d4.ORDER)], dim=0)
        out = F.conv2d(x, big, padding=self.padding)          # (B, 8*C, H, W)
        H, W = out.shape[-2:]
        out = out.reshape(B, d4.ORDER, self.out_channels, H, W).transpose(1, 2)
        return out + self.bias[None, :, None, None, None]      # (B, C, 8, H, W)


class GroupConv(nn.Module):
    """D4 -> D4 group convolution.

    out[:, :, g] = sum_h corr(f[:, :, h], g·psi[:, :, g^{-1}h])
    materialised as one big conv over C_in*8 input planes.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, d4.ORDER, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def _big_weight(self):
        k = self.weight.shape[-1]
        blocks = []
        for g in range(d4.ORDER):
            # gather psi[:, :, g^{-1}h] for h = 0..7, then rotate spatially by g
            ginv = d4.INV[g]
            perm = [d4.MUL[ginv][h] for h in range(d4.ORDER)]
            w = self.weight[:, :, perm]                        # (Co, Ci, 8, k, k)
            w = _transform_kernel(w, g)
            blocks.append(w.reshape(self.out_channels, self.in_channels * d4.ORDER, k, k))
        return torch.cat(blocks, dim=0)                        # (8*Co, Ci*8, k, k)

    def forward(self, x):
        B, C, G, H, W = x.shape
        out = F.conv2d(x.reshape(B, C * G, H, W), self._big_weight(), padding=self.padding)
        out = out.reshape(B, d4.ORDER, self.out_channels, H, W).transpose(1, 2)
        return out + self.bias[None, :, None, None, None]


class EquivariantDQN(nn.Module):
    """Drop-in dueling DQN with D4 equivariance built in.

    Channel widths are the plain DQN's divided by ~sqrt(8) so parameter count
    stays comparable (each group channel carries 8 orientations).
    """

    def __init__(self, num_scalars=2, channels=(12, 24, 24)):
        super().__init__()
        c1, c2, c3 = channels
        self.lift = LiftingConv(2, c1)
        self.gconv2 = GroupConv(c1, c2)
        self.gconv3 = GroupConv(c2, c3)
        self.advantage_conv = nn.Conv2d(c3, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(c3 + num_scalars, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, spatial, scalars):
        x = F.relu(self.lift(spatial))
        x = F.relu(self.gconv2(x))
        x = F.relu(self.gconv3(x))

        m = x.mean(dim=2)                          # group pool -> (B, C, H, W)

        adv = self.advantage_conv(m)               # (B, 1, H, W)
        adv = adv.flatten(1)
        adv = adv - adv.mean(dim=1, keepdim=True)

        v = m.mean(dim=[2, 3])                     # (B, C), invariant
        v = torch.cat([v, scalars], dim=1)
        v = F.relu(self.value_fc1(v))
        v = self.value_fc2(v)

        return v + adv
