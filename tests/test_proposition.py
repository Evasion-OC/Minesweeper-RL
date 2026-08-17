"""Numerical verification of the E5 proposition before it goes in the paper.

Setting: W with a singular value of multiplicity k, and the family
W_R = G_R W where G_R = I + U_k (R - I) U_k^T rotates the degenerate left
subspace, R in SO(k).

Claims checked exhaustively at small size:
  (i)   every W_R has exactly W's singular values;
  (ii)  an orthogonal map aligns W_R to W at zero cost, for every R;
  (iii) for generic U_k, NO permutation aligns any W_R (R != I) exactly:
        the best permutation cost is bounded away from zero;
  (iv)  when the degenerate subspace IS coordinate-aligned (dead units),
        the continuum meets the permutation group and the matching LAP has
        exactly tied optima, the measured N5 tie class.
"""

import itertools

import numpy as np
import pytest

RNG = np.random.default_rng(0)


def make_degenerate(n=6, k=2, sigma=2.0):
    """W = U diag(sigma x k, distinct rest) V^T with generic U, V."""
    U, _ = np.linalg.qr(RNG.normal(size=(n, n)))
    V, _ = np.linalg.qr(RNG.normal(size=(n, n)))
    svals = np.array([sigma] * k + list(3.0 + np.arange(n - k)))
    order = np.argsort(svals)[::-1]
    return (U * svals) @ V.T, U[:, :k], svals[order]


def mix(W, U_k, theta):
    k = U_k.shape[1]
    assert k == 2
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    G = np.eye(W.shape[0]) + U_k @ (R - np.eye(k)) @ U_k.T
    return G @ W


def best_perm_cost(A, B):
    n = A.shape[0]
    best = np.inf
    for p in itertools.permutations(range(n)):
        best = min(best, float(np.sum((A - B[list(p)]) ** 2)))
    return best


def orthogonal_cost(A, B):
    # Procrustes: min_O ||A - O B||_F via SVD of B A^T
    U, s, Vt = np.linalg.svd(B @ A.T)
    O = (U @ Vt).T
    return float(np.sum((A - O @ B) ** 2))


def test_mixing_preserves_singular_values():
    W, U_k, svals = make_degenerate()
    for theta in (0.3, 1.1, 2.9):
        got = np.sort(np.linalg.svd(mix(W, U_k, theta), compute_uv=False))[::-1]
        assert np.allclose(got, svals, atol=1e-9)


def test_orthogonal_alignment_is_exact_along_the_continuum():
    W, U_k, _ = make_degenerate()
    for theta in (0.3, 1.1, 2.9):
        assert orthogonal_cost(W, mix(W, U_k, theta)) < 1e-16


def test_no_permutation_aligns_a_generic_mix():
    W, U_k, _ = make_degenerate()
    for theta in (0.3, 1.1, 2.9):
        assert best_perm_cost(W, mix(W, U_k, theta)) > 1e-2


def test_identity_mix_is_recovered_exactly():
    W, U_k, _ = make_degenerate()
    assert best_perm_cost(W, mix(W, U_k, 0.0)) < 1e-18


def test_coordinate_aligned_degeneracy_gives_exact_lap_ties():
    """Dead units: rows i, j identical (here zero). The degenerate left
    subspace is span(e_i, e_j); swapping them is simultaneously a
    permutation and an in-subspace rotation, so the LAP optimum is tied."""
    from scipy.optimize import linear_sum_assignment

    W, _, _ = make_degenerate()
    W = W.copy()
    W[4] = 0.0
    W[5] = 0.0
    C = W @ W.T
    ri, ci = linear_sum_assignment(-C)
    v_identity = C[np.arange(6), np.arange(6)].sum()
    swap = np.arange(6)
    swap[[4, 5]] = [5, 4]
    v_swapped = C[np.arange(6), swap].sum()
    assert v_identity == pytest.approx(C[ri, ci].sum())
    assert v_swapped == pytest.approx(v_identity)  # exactly tied optima
