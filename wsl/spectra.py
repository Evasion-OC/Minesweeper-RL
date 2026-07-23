"""Per-layer singular value spectra of DQN checkpoints.

Conv kernels are reshaped to (C_out, C_in * k * k); linear layers used as-is.
"""

import numpy as np
import torch

LAYERS = ["conv1.weight", "conv2.weight", "conv3.weight", "value_fc1.weight"]


def layer_spectra(sd, layers=None):
    """Returns {layer: descending singular values (np.ndarray)}."""
    out = {}
    for name in (layers or LAYERS):
        w = sd[name].detach().cpu().numpy()
        if w.ndim == 4:
            w = w.reshape(w.shape[0], -1)
        out[name] = np.linalg.svd(w, compute_uv=False)
    return out


def relative_gaps(sigma):
    """(sigma_i - sigma_{i+1}) / sigma_i for a descending spectrum."""
    s = np.asarray(sigma)
    return (s[:-1] - s[1:]) / np.maximum(s[:-1], 1e-12)
