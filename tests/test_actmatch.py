"""Activation matching must recover an exact permutation and preserve the
perms convention shared with weight matching."""

import torch

from minesweeper.models import DQN
from wsl.actmatch import activation_matching
from wsl.align import apply_perms, random_perms

STATES = (torch.randn(128, 2, 8, 8), torch.rand(128, 2))
DEVICE = torch.device("cpu")


def test_recovers_exact_permutation_on_probe_active_units():
    """Activation matching can only identify units the probe set excites:
    units with zero activation variance z-score to nothing and match
    arbitrarily (the same tie class the weight arm quotients). On active
    units the exact permutation must be recovered."""
    import numpy as np

    from wsl.activations import activation_matrices

    torch.manual_seed(0)
    a = DQN().eval()
    true = random_perms(seed=9)
    sd_b = apply_perms(a.state_dict(), true)
    b = DQN().eval()
    b.load_state_dict(sd_b)
    found = activation_matching(a, b, STATES, DEVICE)
    acts_a = activation_matrices(a, STATES, DEVICE)
    acts_b = activation_matrices(b, STATES, DEVICE)
    for ax, perm in found.items():
        expected = np.argsort(true[ax])  # A's unit i sits at B position inv(true)[i]
        active = acts_a[ax].std(axis=1) > 1e-12
        assert active.mean() > 0.5, f"{ax}: probe excites too few units to test"
        # off-expected assignments are acceptable only on an exact
        # correlation tie (barely-active units spiking on the same states)
        def z(X):
            Xc = X - X.mean(axis=1, keepdims=True)
            return Xc / np.maximum(X.std(axis=1, keepdims=True), 1e-12)
        Za, Zb = z(acts_a[ax]), z(acts_b[ax])
        for i in np.where(active)[0]:
            if perm[i] == expected[i]:
                continue
            corr = float(Za[i] @ Zb[perm[i]] / Za.shape[1])
            assert corr >= 1.0 - 1e-9, (ax, i, int(perm[i]), corr)


def test_self_match_is_identity_up_to_exact_correlation_ties():
    """Self-match must return identity except where two units' activations
    are perfectly correlated on the probe set (e.g. barely-active units
    spiking on the same states): an exact tie class where the assignment is
    arbitrary, the activation-space analogue of the dead-unit class."""
    import numpy as np

    from wsl.activations import activation_matrices

    torch.manual_seed(1)
    a = DQN().eval()
    found = activation_matching(a, a, STATES, DEVICE)
    acts = activation_matrices(a, STATES, DEVICE)
    for ax, perm in found.items():
        X = acts[ax]
        Xc = X - X.mean(axis=1, keepdims=True)
        std = X.std(axis=1)
        Z = Xc / np.maximum(std[:, None], 1e-12)
        off = np.where(perm != np.arange(len(perm)))[0]
        for i in off:
            j = perm[i]
            if std[i] <= 1e-12 or std[j] <= 1e-12:
                continue  # inactive units: known arbitrary class
            corr = float(Z[i] @ Z[j] / Z.shape[1])
            assert corr >= 1.0 - 1e-9, (ax, i, j, corr)
