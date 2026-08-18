"""The E6 quantizer: agreement with torch's own quantization on one tensor
as a sanity check, plus scheme behaviour."""

import pytest
import torch

from wsl.quantize import SCHEMES, dynamic_range, fake_quantize, quantize_layer


def test_matches_torch_reference_per_tensor_int8():
    torch.manual_seed(0)
    w = torch.randn(64, 32)
    scale = float(w.abs().max() / 127)
    ref = torch.dequantize(torch.quantize_per_tensor(w, scale, 0, torch.qint8))
    ours = fake_quantize(w, bits=8, per_channel=False)
    # torch's qint8 range is [-128, 127]; ours is symmetric [-127, 127].
    # They may differ only where torch uses code -128, i.e. at w = -max|W|.
    mask = torch.round(w / scale) > -128
    assert torch.allclose(ours[mask], ref[mask], atol=1e-6)
    assert (ours - ref).abs().max() <= scale + 1e-6


def test_int8_error_much_smaller_than_int4():
    torch.manual_seed(1)
    w = torch.randn(128, 128)
    e8 = (fake_quantize(w, 8) - w).pow(2).mean()
    e4 = (fake_quantize(w, 4) - w).pow(2).mean()
    assert e4 > 20 * e8


def test_per_channel_rescues_rows_wrecked_by_a_shared_scale():
    torch.manual_seed(2)
    w = torch.randn(32, 64)
    w[0] *= 50  # one outlier row inflates the shared scale
    et = (fake_quantize(w, 4, per_channel=False) - w).pow(2)
    ec = (fake_quantize(w, 4, per_channel=True) - w).pow(2)
    # totals: the outlier row's own error dominates both schemes equally,
    # so per-channel is better but not dramatically
    assert ec.mean() <= et.mean() * 1.01
    # on the non-outlier rows, per-channel's advantage is enormous
    assert ec[1:].mean() < et[1:].mean() / 5


def test_quantize_layer_touches_exactly_one_tensor():
    sd = {"a.weight": torch.randn(8, 8), "b.weight": torch.randn(8, 8)}
    for scheme in SCHEMES:
        out = quantize_layer(sd, "a.weight", scheme)
        assert not torch.equal(out["a.weight"], sd["a.weight"])
        assert torch.equal(out["b.weight"], sd["b.weight"])


def test_dynamic_range_flags_outliers():
    w = torch.randn(64, 64)
    base = dynamic_range(w)
    w2 = w.clone()
    w2[0, 0] = 100.0
    assert dynamic_range(w2) > 5 * base
