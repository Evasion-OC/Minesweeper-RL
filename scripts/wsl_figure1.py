"""Figure 1 of the paper, three panels from committed results JSONs.

(a) The diagnostic: endpoint-measured spectral gap against matching
    instability, 120 DQN pairs x 4 layers, colored by layer.
(b) The regime map: performance retained along the interpolation, naive and
    aligned, MNIST MLPs (barrier closes) against Minesweeper DQNs (barrier
    stays open), one pipeline.
(c) The solver fix: fraction of exactly solvable instances recovered against
    the number of coordinate-descent restarts.

    python3 scripts/wsl_figure1.py --out-dir results
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

from wsl.degeneracy import AXIS_LAYER  # noqa: E402

# validated categorical palette (dataviz reference instance, light mode)
COLORS = {"p1": "#2a78d6", "p2": "#eb6834", "p3": "#1baf7a", "p4": "#eda100"}
BLUE, ORANGE = "#2a78d6", "#eb6834"
LAYER_NAMES = {"p1": "conv1", "p2": "conv2", "p3": "conv3", "p4": "value fc"}
LAMS = [0.0, 0.25, 0.5, 0.75, 1.0]


def panel_a(ax, e3_path):
    d = json.load(open(e3_path))
    deg = d["degeneracy"]
    for axname in AXIS_LAYER:
        xs, ys = [], []
        for e in d["pairs"]:
            layer = AXIS_LAYER[axname]
            xs.append(0.5 * (deg[e["a"]][layer]["min_rel_gap"]
                             + deg[e["b"]][layer]["min_rel_gap"]))
            ys.append(e["perturbation"][axname]["churn_mean"])
        ax.scatter(xs, ys, s=13, alpha=0.4, color=COLORS[axname], lw=0,
                   label=LAYER_NAMES[axname])
        mx, my = np.median(xs), np.median(ys)
        ax.scatter([mx], [my], s=110, color=COLORS[axname], edgecolor="black",
                   linewidth=1.0, zorder=5)
        ax.annotate(LAYER_NAMES[axname], (mx, my), textcoords="offset points",
                    xytext=(7, 7), fontsize=8.5, color="#0b0b0b", zorder=6)
    ax.set_xscale("log")
    ax.set_xlabel("spectral min relative gap (endpoints only)")
    ax.set_ylabel("matching instability (churn under $\\epsilon$-noise)")
    ax.set_title("(a) The diagnostic ranks layer reliability", fontsize=10)
    ax.set_ylim(-0.03, 1.03)
    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.9)


def _retained(curves, better):
    return np.array(curves) / max(better, 1e-9)


def panel_b(ax, dqn_path, mlp_path):
    dqn = json.load(open(dqn_path))
    mlp = json.load(open(mlp_path))["pairs"]
    groups = {}
    n_curves, a_curves = [], []
    for res in dqn:
        better = max(res["naive"][0]["win_rate"], res["naive"][-1]["win_rate"])
        n_curves.append(_retained([r["win_rate"] for r in res["naive"]], better))
        a_curves.append(_retained([r["win_rate"] for r in res["aligned"]], better))
    groups["DQN"] = (np.array(n_curves), np.array(a_curves), ORANGE)
    n_curves, a_curves = [], []
    for e in mlp:
        better = max(e["naive_acc"][0], e["naive_acc"][-1])
        n_curves.append(_retained(e["naive_acc"], better))
        a_curves.append(_retained(e["aligned_acc"], better))
    groups["MNIST"] = (np.array(n_curves), np.array(a_curves), BLUE)

    for name, (nc, ac, color) in groups.items():
        for curves, style, lw in ((ac, "-", 2.0), (nc, "--", 1.5)):
            med = np.median(curves, axis=0)
            lo, hi = np.percentile(curves, 25, axis=0), np.percentile(curves, 75, axis=0)
            ax.plot(LAMS, med, style, color=color, lw=lw)
            ax.fill_between(LAMS, lo, hi, color=color, alpha=0.13, lw=0)
    ax.annotate("MNIST aligned", (0.5, 1.005), fontsize=8.5, color=BLUE,
                ha="center", va="bottom")
    ax.annotate("MNIST naive", (0.5, 0.895), fontsize=8.5, color=BLUE, ha="center")
    ax.annotate("DQN aligned", (0.5, 0.30), fontsize=8.5, color=ORANGE, ha="center")
    ax.annotate("DQN naive", (0.5, 0.005), fontsize=8.5, color=ORANGE, ha="center")
    ax.set_xlabel("interpolation $\\lambda$")
    ax.set_ylabel("performance retained (÷ better endpoint)")
    ax.set_title("(b) One pipeline, two regimes", fontsize=10)
    ax.set_ylim(-0.05, 1.12)
    ax.plot([], [], "-", color="#52514e", lw=2.0, label="aligned (restarts)")
    ax.plot([], [], "--", color="#52514e", lw=1.5, label="naive")
    ax.legend(loc="center left", fontsize=7.5, framealpha=0.9)


def panel_c(ax, rec_path):
    rec = json.load(open(rec_path))["cumulative_recovery"]
    ks = np.arange(1, len(rec) + 1)
    ax.plot(ks, rec, "-o", color=BLUE, lw=2.0, ms=5.5)
    ax.annotate("single run: 40%", (1, rec[0]), textcoords="offset points",
                xytext=(8, -2), fontsize=8.5)
    ax.annotate("2 restarts: 100%", (2, rec[1]), textcoords="offset points",
                xytext=(8, -11), fontsize=8.5)
    ax.set_xlabel("coordinate-descent restarts (best of $k$)")
    ax.set_ylabel("exact permutation recovered")
    ax.set_title("(d) Restarts repair the matcher", fontsize=10)
    ax.set_xticks(ks)
    ax.set_ylim(0.30, 1.06)


def panel_d(ax, width_files, dqn_files):
    """Identifiability vs mergeability: median conv/hidden-layer assignment
    margin against the aligned relative barrier, MLP width family plus the
    DQN width family."""
    xs, ys, labels = [], [], []
    for wlabel, path in width_files:
        d = json.load(open(path))
        probed = [e for e in d["pairs"] if "assignment_gap" in e]
        axes_m = ("h1", "h2", "h3")
        marg = np.median([e["assignment_gap"][ax]["rel_gap"]
                          for e in probed for ax in axes_m])
        bar = np.median([e["aligned_barrier"]["rel_barrier"] for e in d["pairs"]])
        xs.append(max(marg, 1e-9)); ys.append(bar); labels.append(wlabel)
    ax.plot(xs, ys, "-o", color=BLUE, lw=2.0, ms=6, zorder=3)
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 6),
                    fontsize=8, color=BLUE)
    dx, dy, dl = [], [], []
    for wlabel, e3_path, bar_val in dqn_files:
        d = json.load(open(e3_path))
        marg = np.median([e["assignment_gap"][ax]["rel_gap"]
                          for e in d["pairs"] for ax in ("p1", "p2", "p3")])
        dx.append(max(marg, 1e-9)); dy.append(bar_val); dl.append(wlabel)
    ax.scatter(dx, dy, marker="s", s=42, color=ORANGE, zorder=3)
    for x, y, lab in zip(dx, dy, dl):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, -11),
                    fontsize=8, color=ORANGE)
    ax.annotate("MNIST MLPs:\nmargins collapse,\nbarrier closes", (2.2e-5, 0.30),
                fontsize=8, color=BLUE)
    ax.annotate("RL DQNs: determined\nbut insufficient", (2.5e-4, 0.60),
                fontsize=8, color=ORANGE)
    ax.set_xscale("log")
    ax.set_xlabel("assignment margin (median, behaviour-carrying layers)")
    ax.set_ylabel("aligned relative barrier")
    ax.set_title("(c) Identifiability vs mergeability", fontsize=10)
    ax.set_ylim(-0.06, 1.0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--e3", default="results/wsl_e3r_zoo16.json")
    p.add_argument("--dqn-barriers", default="results/wsl_barrier_w1/wsl_barriers.json")
    p.add_argument("--mlp", default="results/wsl_z4.json")
    p.add_argument("--recovery", default="results/wsl_solver_recovery.json")
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.22, "grid.linewidth": 0.6})
    fig, axes = plt.subplots(1, 4, figsize=(16.4, 3.6))
    panel_a(axes[0], args.e3)
    panel_b(axes[1], args.dqn_barriers, args.mlp)
    panel_d(axes[2], [("32", "results/wsl_z4w32.json"), ("64", "results/wsl_z4w64.json"),
                      ("128", "results/wsl_z4w128.json"), ("512", "results/wsl_z4.json")],
            [("DQN 1x", "results/wsl_e3r_zoo16.json", 0.792),
             ("2x", "results/wsl_e3r_w2.json", 0.812),
             ("4x", "results/wsl_e3r_w4.json", 0.789)])
    panel_c(axes[3], args.recovery)
    fig.tight_layout(w_pad=2.0)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"figure1.{ext}", dpi=300, bbox_inches="tight")
        print(f"wrote {out / f'figure1.{ext}'}")


if __name__ == "__main__":
    main()
