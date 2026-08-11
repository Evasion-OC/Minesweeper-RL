"""Activation covariance spectra on a frozen probe set (null N4).

For each permutation axis, the post-ReLU activations of the layer it
permutes, collected over a fixed probe buffer and centred per unit. The
spectrum reported is the singular values of the centred activation matrix
scaled by 1/sqrt(samples) (equivalently the square roots of the covariance
eigenvalues), so degeneracy_stats keeps the same semantics it has on the
weight spectra: entropy of the squared spectrum, min relative gap on the
live part.

The convolution layers apply ReLU functionally in forward(), so module
forward hooks see pre-activation outputs; ReLU is applied to the captured
tensors here to give the activations the next layer actually consumes.
"""

import numpy as np
import torch

from .degeneracy import degeneracy_stats

AXIS_MODULE = {"p1": "conv1", "p2": "conv2", "p3": "conv3", "p4": "value_fc1"}


def activation_matrices(model, states, device, batch_size=512):
    """{axis: post-ReLU activation matrix (units x samples)} over the
    probe buffer."""
    acts = {ax: [] for ax in AXIS_MODULE}
    hooks = []

    def grab(ax):
        def fn(_module, _inputs, out):
            a = torch.relu(out.detach())
            if a.ndim == 4:  # (B, C, H, W) -> (C, B*H*W)
                a = a.permute(1, 0, 2, 3).reshape(a.shape[1], -1)
            else:            # (B, H) -> (H, B)
                a = a.T
            acts[ax].append(a.cpu())
        return fn

    for ax, name in AXIS_MODULE.items():
        hooks.append(getattr(model, name).register_forward_hook(grab(ax)))
    try:
        sp, sc = states
        model.eval()
        with torch.no_grad():
            for i in range(0, sp.shape[0], batch_size):
                model(sp[i:i + batch_size].to(device),
                      sc[i:i + batch_size].to(device))
    finally:
        for h in hooks:
            h.remove()
    return {ax: torch.cat(acts[ax], dim=1).double().numpy()
            for ax in AXIS_MODULE}


def activation_spectra(model, states, device, batch_size=512):
    """{axis: descending singular values of the centred post-ReLU
    activation matrix} over the probe buffer."""
    out = {}
    for ax, X in activation_matrices(model, states, device, batch_size).items():
        X = X - X.mean(axis=1, keepdims=True)
        out[ax] = np.linalg.svd(X / np.sqrt(X.shape[1]), compute_uv=False)
    return out


def activation_degeneracy(model, states, device, batch_size=512):
    """{axis: degeneracy_stats of the activation spectrum}."""
    return {ax: degeneracy_stats(sig)
            for ax, sig in activation_spectra(model, states, device,
                                              batch_size).items()}
