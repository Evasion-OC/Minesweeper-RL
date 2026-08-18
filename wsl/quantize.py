"""E6: weight-only symmetric fake quantization and per-layer sensitivity.

Hand-written rather than torch.ao by design: the paper needs exact control
of the scheme and makes no latency claim, so quantize-dequantize in fp32 is
exactly sufficient. Symmetric with restricted range: for b bits the code
range is [-(2^(b-1)-1), +(2^(b-1)-1)] and the scale is max|W| over that
range (per tensor, or per output channel along dim 0).

The dynamic-range covariate max|W| / RMS(W) lives here too, because E6's
analysis is meaningless without it: first-order quantization error is
governed by dynamic range, not by the spectrum, and the partial correlation
given this covariate is the arm's only reported number.
"""

import torch

SCHEMES = ("int8_tensor", "int8_channel", "int4_tensor", "int4_channel")


def fake_quantize(w, bits, per_channel=False):
    """Quantize-dequantize a float tensor, symmetric, restricted range."""
    qmax = 2 ** (bits - 1) - 1
    if per_channel and w.ndim >= 2:
        amax = w.abs().flatten(1).max(dim=1).values
        scale = (amax / qmax).clamp_min(1e-12)
        scale = scale.view(-1, *([1] * (w.ndim - 1)))
    else:
        scale = (w.abs().max() / qmax).clamp_min(1e-12)
    return torch.clamp(torch.round(w / scale), -qmax, qmax) * scale


def quantize_layer(sd, name, scheme):
    """Copy of sd with one weight tensor fake-quantized."""
    bits = 8 if scheme.startswith("int8") else 4
    per_channel = scheme.endswith("channel")
    out = dict(sd)
    out[name] = fake_quantize(sd[name], bits, per_channel)
    return out


def dynamic_range(w):
    """max|W| / RMS(W), the covariate quantization error actually follows."""
    return float(w.abs().max() / w.pow(2).mean().sqrt().clamp_min(1e-12))
