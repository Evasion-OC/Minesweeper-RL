"""Group-theory sanity: the numerically derived D4 tables really are D4."""

import torch

from minesweeper import d4


def test_group_axioms():
    assert d4.ORDER == 8
    # closure + identity + inverses come from the table construction; check
    # associativity explicitly on the full table
    for a in range(8):
        for b in range(8):
            for c in range(8):
                assert d4.MUL[d4.MUL[a][b]][c] == d4.MUL[a][d4.MUL[b][c]]
    for a in range(8):
        assert d4.MUL[a][d4.INV[a]] == 0
        assert d4.MUL[d4.INV[a]][a] == 0
        assert d4.MUL[a][0] == a and d4.MUL[0][a] == a


def test_apply_matches_table():
    x = torch.randn(2, 4, 6)  # rectangular: pins table validity beyond squares
    for a in range(8):
        for b in range(8):
            ab = d4.MUL[a][b]
            assert torch.equal(d4.apply(d4.apply(x, b), a), d4.apply(x, ab))


def test_apply_inverse_roundtrip():
    x = torch.randn(3, 2, 4, 7)
    for g in range(8):
        assert torch.equal(d4.apply_inverse(d4.apply(x, g), g), x)


def test_nonsquare_rotation_shapes():
    x = torch.randn(1, 2, 4, 6)
    for g in range(8):
        y = d4.apply(x, g)
        k, _ = d4.ELEMENTS[g]
        if k % 2 == 1:
            assert y.shape[-2:] == (6, 4)
        else:
            assert y.shape[-2:] == (4, 6)
