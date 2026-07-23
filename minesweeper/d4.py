"""The dihedral group D4 acting on board planes and Q-maps.

Elements are indexed 0..7 as (k, j): x -> rot90^k(flip^j(x)), flip along the
last (width) axis, rotations counter-clockwise in the (H, W) plane. The
Cayley and inverse tables are derived numerically by composing the actual
index maps, so there is no hand-derived sign convention to get wrong; the
equivariance unit tests then check everything built on top.

Used two ways:
  - test-time augmentation for any per-cell Q model (D4 orbit averaging)
  - kernel transforms + group indices for the equivariant network
"""

import torch

# element i = (k_i, j_i), action: rot90^k ∘ flip^j
ELEMENTS = [(k, j) for j in (0, 1) for k in (0, 1, 2, 3)]
ORDER = len(ELEMENTS)  # 8


def apply(x, idx):
    """Apply element idx to the last two dims of x."""
    k, j = ELEMENTS[idx]
    if j:
        x = torch.flip(x, dims=[-1])
    if k:
        x = torch.rot90(x, k, dims=[-2, -1])
    return x


def _index_grid(n=4):
    return torch.arange(n * n).reshape(n, n)


def _build_tables():
    """mul[a][b] = index of a∘b (apply b first), inv[a] = index of a^{-1}."""
    grid = _index_grid()
    images = [apply(grid, i) for i in range(ORDER)]

    def find(img):
        for i, ref in enumerate(images):
            if torch.equal(img, ref):
                return i
        raise RuntimeError("composition left the group; transform set is inconsistent")

    mul = [[find(apply(images[b], a)) for b in range(ORDER)] for a in range(ORDER)]
    inv = [None] * ORDER
    for a in range(ORDER):
        for b in range(ORDER):
            if mul[a][b] == 0:
                inv[a] = b
    return mul, inv


MUL, INV = _build_tables()


def apply_inverse(x, idx):
    return apply(x, INV[idx])


def orbit_elements(square):
    """Element indices usable for TTA. All 8 work for any board because the
    network is size-agnostic (a rotated non-square board is just a different
    input size); callers that must preserve shape mid-pipeline can restrict
    to the axis-preserving subgroup."""
    if square:
        return list(range(ORDER))
    return [i for i in range(ORDER) if ELEMENTS[i][0] % 2 == 0]


@torch.no_grad()
def tta_q_values(model, spatial, scalars, elements=None):
    """D4 test-time augmentation for per-cell Q models.

    q_tta(x) = (1/|G|) sum_g  g^{-1} · Q(g·x)

    spatial: (B, 2, H, W); returns (B, H*W) like the model.
    """
    B, _, H, W = spatial.shape
    if elements is None:
        elements = orbit_elements(square=(H == W))
    acc = torch.zeros(B, H, W, device=spatial.device, dtype=spatial.dtype)
    for g in elements:
        sp_g = apply(spatial, g)
        q = model(sp_g, scalars)                    # (B, Hg*Wg)
        Hg, Wg = sp_g.shape[-2:]
        q = q.reshape(B, Hg, Wg)
        acc += apply_inverse(q, g)
    return (acc / len(elements)).reshape(B, H * W)
