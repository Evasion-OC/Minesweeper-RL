"""E6 analysis: does spectral degeneracy predict quantization sensitivity
once dynamic range is controlled?

Pools (checkpoint, layer) points across families. Headline number, fixed in
advance: partial Spearman of min_rel_gap against int4-per-tensor sensitivity
given max|W|/RMS(W), with a checkpoint-cluster bootstrap CI and the
N3-style permutation test (gaps shuffled within checkpoints). N6 reports the
raw correlation next to the partial one. Layers whose spectrum has a single
live singular value (rank-1 output heads) have no defined gap and are
excluded, with the count stated.

    python3 scripts/wsl_e6_analysis.py results/wsl_e6_*.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats as sps  # noqa: E402

HEADLINE = "int4_tensor"
FAMILY_COLOR = {"dqn": "#eb6834", "mlp": "#2a78d6", "cifar": "#1baf7a"}


def load_rows(paths):
    rows = []
    for path in paths:
        with open(path) as f:
            d = json.load(f)
        family = Path(path).stem.replace("wsl_e6_", "")
        for ck, entry in d["results"].items():
            for layer, r in entry["layers"].items():
                rows.append({"family": family, "ckpt": ck, "layer": layer,
                             "gap": r["min_rel_gap"], "drange": r["drange"],
                             **{s: r["schemes"][s] for s in r["schemes"]}})
    kept = [r for r in rows if np.isfinite(r["gap"])]
    print(f"{len(rows)} (checkpoint, layer) points; {len(rows) - len(kept)} "
          "rank-one layers excluded (no defined gap)")
    return kept


def partial_spearman(x, y, z):
    rx, ry, rz = (sps.rankdata(v) for v in (x, y, z))
    rxy = np.corrcoef(rx, ry)[0, 1]
    rxz = np.corrcoef(rx, rz)[0, 1]
    ryz = np.corrcoef(ry, rz)[0, 1]
    return (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))


def analyse(rows, scheme, label):
    x = np.array([r["gap"] for r in rows])
    y = np.array([r[scheme] for r in rows])
    z = np.array([r["drange"] for r in rows])
    raw = sps.spearmanr(x, y)[0]
    part = partial_spearman(x, y, z)

    ckpts = sorted({r["ckpt"] for r in rows})
    by_ck = {c: [i for i, r in enumerate(rows) if r["ckpt"] == c] for c in ckpts}
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(2000):
        idx = np.concatenate([by_ck[c] for c in rng.choice(ckpts, len(ckpts))])
        boots.append(partial_spearman(x[idx], y[idx], z[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])

    null = []
    for _ in range(10000):
        xs = x.copy()
        for c in ckpts:
            ii = by_ck[c]
            xs[ii] = xs[rng.permutation(ii)]
        null.append(partial_spearman(xs, y, z))
    null = np.array(null)
    pval = float(np.mean(np.abs(null) >= abs(part)))

    print(f"{label:22s} raw {raw:+.3f}   partial {part:+.3f} "
          f"[{lo:+.3f}, {hi:+.3f}]   perm p {pval:.4f} (n={len(rows)})")
    return part


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsons", nargs="+")
    ap.add_argument("--fig", default="results/figure_e6.png")
    args = ap.parse_args()

    rows = load_rows(args.jsons)
    print(f"\nheadline scheme {HEADLINE} (N6: raw vs partial):")
    analyse(rows, HEADLINE, "pooled")
    for fam in sorted({r["family"] for r in rows}):
        analyse([r for r in rows if r["family"] == fam], HEADLINE, f"  {fam}")
    print("\nother schemes (pooled partial):")
    for scheme in ("int8_tensor", "int8_channel", "int4_channel"):
        analyse(rows, scheme, f"  {scheme}")

    # partial-residual figure for the headline scheme
    x = sps.rankdata([r["gap"] for r in rows])
    y = sps.rankdata([r[HEADLINE] for r in rows])
    z = sps.rankdata([r["drange"] for r in rows])
    bx = np.polyfit(z, x, 1)
    by = np.polyfit(z, y, 1)
    resx = x - np.polyval(bx, z)
    resy = y - np.polyval(by, z)
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for fam, color in FAMILY_COLOR.items():
        m = [i for i, r in enumerate(rows) if r["family"] == fam]
        if m:
            ax.scatter(resx[m], resy[m], s=14, alpha=0.55, color=color,
                       lw=0, label=fam.upper())
    ax.set_xlabel("spectral min gap | dynamic range (rank residual)")
    ax.set_ylabel(f"{HEADLINE} sensitivity | dynamic range (rank residual)")
    ax.set_title("E6: degeneracy vs quantization sensitivity, controlled",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    Path(args.fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig, dpi=300)
    print(f"\nwrote {args.fig}")


if __name__ == "__main__":
    main()
