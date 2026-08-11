"""REPAIR for a normalisation-free net (Jordan et al., ICLR 2023): after
interpolating two aligned networks, hidden pre-activation statistics
collapse; the fix resets each hidden unit's pre-activation mean and std to
the interpolation of the endpoints' statistics.

This is the closed-form rescaling variant, since the architecture has no
normalisation layers to reset: for each hidden layer in order, measure the
interpolated net's per-unit pre-activation statistics on the frozen probe
buffer, then rescale the unit's incoming weights and shift its bias so the
statistics match the target. Walking layers in forward order matters,
because correcting a layer changes every downstream distribution.

Units whose measured std is numerically zero (dead units) keep scale 1 and
only their mean is shifted, to avoid amplifying numerical noise.
"""

import torch

HIDDEN = ("conv1", "conv2", "conv3", "value_fc1")
STD_FLOOR = 1e-8


def unit_stats(model, states, device, layers, batch_size=512):
    """{layer: (mean, std)} of per-unit PRE-activation outputs over the
    probe buffer (module outputs; ReLU is applied functionally later)."""
    grabbed = {name: [] for name in layers}
    hooks = [getattr(model, name).register_forward_hook(
        (lambda nm: lambda _m, _i, out: grabbed[nm].append(out.detach().cpu()))(name))
        for name in layers]
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
    out = {}
    for name in layers:
        X = torch.cat(grabbed[name], dim=0)
        if X.ndim == 4:  # (B, C, H, W): stats per channel
            X = X.permute(1, 0, 2, 3).reshape(X.shape[1], -1)
        else:            # (B, H): stats per hidden unit
            X = X.T
        out[name] = (X.mean(dim=1), X.std(dim=1))
    return out


def repair(sd_mid, stats_a, stats_b, lam, make_model, states, device):
    """Return a corrected copy of sd_mid whose hidden pre-activation
    statistics match (1-lam)*A + lam*B on the probe buffer.

    stats_a/stats_b: endpoint unit_stats over HIDDEN (computed once by the
    caller). make_model(sd) must return a loaded model on `device`.
    """
    work = {k: v.clone() for k, v in sd_mid.items()}
    for layer in HIDDEN:
        model = make_model(work)
        m_mid, s_mid = unit_stats(model, states, device, (layer,))[layer]
        m_t = (1.0 - lam) * stats_a[layer][0] + lam * stats_b[layer][0]
        s_t = (1.0 - lam) * stats_a[layer][1] + lam * stats_b[layer][1]
        scale = torch.where(s_mid > STD_FLOOR, s_t / s_mid.clamp_min(STD_FLOOR),
                            torch.ones_like(s_mid))
        w = work[f"{layer}.weight"]
        work[f"{layer}.weight"] = w * scale.view(-1, *([1] * (w.ndim - 1)))
        work[f"{layer}.bias"] = (work[f"{layer}.bias"] - m_mid) * scale + m_t
    return work
