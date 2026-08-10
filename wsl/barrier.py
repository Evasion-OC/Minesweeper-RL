"""Interpolation barriers, defined once and used everywhere.

Barrier convention, fixed here and shared with `Spec_Merging_Extension.md`:
the worst interpolation degradation relative to the BETTER endpoint. For a
higher-is-better metric m over lambdas including 0 and 1,

    barrier = max_{0 < lam < 1} [ max(m(0), m(1)) - m(lam) ]

reported absolutely and relative to the better endpoint. For a
lower-is-better metric (Q-value MSE) the sign is flipped, so a barrier is
always "how much worse the interior gets", and positive means a barrier.

Two curves are supported, per the paper brief:
  win rate, greedy play over a fixed set of boards (seeded, so every model
      in a population meets the same boards);
  Q-value MSE against each endpoint on a frozen buffer of states, which
      avoids the non-stationarity trap of re-collecting states per model.

The probe buffer is collected from a designated reference agent that is not
a member of any population being compared (or from random play if none is
given), so it is neutral across the pairs it is used to score. Random play
in this environment dies early and therefore undersamples late-game states;
prefer a reference checkpoint when one is available.
"""

import numpy as np
import torch

from .align import PERM_AXES, apply_perms, interpolate


def barrier(values, lambdas, higher_is_better=True):
    """Worst interior degradation relative to the better endpoint.

    Returns {'barrier', 'rel_barrier', 'argmax_lam', 'endpoints'}. Positive
    barrier means the interior is worse than both endpoints.
    """
    v = np.asarray(values, dtype=np.float64)
    lam = np.asarray(lambdas, dtype=np.float64)
    if not (np.isclose(lam.min(), 0.0) and np.isclose(lam.max(), 1.0)):
        raise ValueError("lambdas must include both endpoints 0 and 1")
    end = np.array([v[np.isclose(lam, 0.0)][0], v[np.isclose(lam, 1.0)][0]])
    best = end.max() if higher_is_better else end.min()
    interior = ~(np.isclose(lam, 0.0) | np.isclose(lam, 1.0))
    if not interior.any():
        raise ValueError("need at least one interior lambda")
    deg = (best - v[interior]) if higher_is_better else (v[interior] - best)
    k = int(np.argmax(deg))
    denom = abs(best) if abs(best) > 1e-12 else 1.0
    return {"barrier": float(deg[k]), "rel_barrier": float(deg[k] / denom),
            "argmax_lam": float(lam[interior][k]),
            "endpoints": [float(end[0]), float(end[1])]}


def collect_probe_states(env, n_states, seed=0, policy=None, device=None,
                         epsilon=0.1, max_steps=None):
    """Frozen buffer of (spatial, scalars) states.

    policy: a model to roll out (epsilon-greedy), or None for uniform-random
    valid actions. Seeded, so the buffer is reproducible.
    """
    import random
    rng_state = (random.getstate(), np.random.get_state())
    random.seed(seed)
    np.random.seed(seed)
    spatials, scalars_l = [], []
    max_steps = max_steps or max(2 * env.n_rows * env.n_cols, 200)
    try:
        while len(spatials) < n_states:
            spatial, scalars = env.reset()
            for _ in range(max_steps):
                if len(spatials) >= n_states:
                    break
                valid = env.valid_actions()
                if not valid:
                    break
                spatials.append(np.asarray(spatial, dtype=np.float32).copy())
                scalars_l.append(np.asarray(scalars, dtype=np.float32).copy())
                if policy is None or random.random() < epsilon:
                    action = random.choice(valid)
                else:
                    sp = torch.from_numpy(spatial).unsqueeze(0).to(device)
                    sc = torch.from_numpy(scalars).unsqueeze(0).to(device)
                    with torch.no_grad():
                        q = policy(sp, sc)
                    action = valid[q[0, valid].argmax().item()]
                (spatial, scalars), _, done, _ = env.step(action)
                if done:
                    break
    finally:
        random.setstate(rng_state[0])
        np.random.set_state(rng_state[1])
    return (torch.from_numpy(np.stack(spatials)),
            torch.from_numpy(np.stack(scalars_l)))


def q_values(model, sd, states, device, batch_size=512):
    """Q values of `sd` on a frozen state buffer, as one (N, H*W) tensor."""
    model.load_state_dict(sd)
    model.eval()
    sp, sc = states
    out = []
    with torch.no_grad():
        for i in range(0, sp.shape[0], batch_size):
            out.append(model(sp[i:i + batch_size].to(device),
                             sc[i:i + batch_size].to(device)).cpu())
    return torch.cat(out)


def q_mse_curve(sd_a, sd_b, model, states, device, lambdas):
    """Mean of the Q-MSE against each endpoint, along the interpolation.

    Symmetric in the endpoints by construction, so it does not privilege
    either one; at lam 0 and 1 it is the distance to the other endpoint,
    which is the natural baseline the interior must beat.
    """
    qa = q_values(model, sd_a, states, device)
    qb = q_values(model, sd_b, states, device)
    out = []
    for lam in lambdas:
        q = q_values(model, interpolate(sd_a, sd_b, lam), states, device)
        mse = 0.5 * (torch.mean((q - qa) ** 2) + torch.mean((q - qb) ** 2))
        out.append(float(mse))
    return out


def leave_one_axis_perms(perms, axis):
    """Copy of `perms` with `axis` reset to the identity, for measuring how
    much aligning that one axis contributes."""
    out = {p: v.copy() for p, v in perms.items()}
    out[axis] = np.arange(len(perms[axis]))
    return out


def per_axis_barrier_contribution(sd_a, sd_b, perms, curve_fn, lambdas,
                                  higher_is_better=True):
    """Barrier with every axis aligned, and with each axis in turn left
    unaligned. contribution[p] = barrier(all but p) - barrier(all): how much
    worse the barrier gets when axis p is not aligned.

    curve_fn(sd_a, sd_b_variant, lambdas) -> list of metric values.
    """
    full = barrier(curve_fn(sd_a, apply_perms(sd_b, perms), lambdas),
                   lambdas, higher_is_better)
    out = {"aligned": full, "per_axis": {}}
    for p in PERM_AXES:
        held = apply_perms(sd_b, leave_one_axis_perms(perms, p))
        b = barrier(curve_fn(sd_a, held, lambdas), lambdas, higher_is_better)
        out["per_axis"][p] = {"barrier": b,
                              "contribution": b["barrier"] - full["barrier"]}
    return out
