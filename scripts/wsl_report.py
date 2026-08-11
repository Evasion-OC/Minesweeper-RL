"""Weight-space report over the seed zoo: interpolation barriers before and
after permutation alignment, and per-layer SVD spectra.

For every checkpoint pair (or the first N pairs): evaluate the linear
interpolation (1-lam) A + lam B at a lambda grid, naive vs aligned
(Git Re-Basin weight matching), on shared eval seeds. Then plot per-layer
singular value spectra across the zoo.

    python3 scripts/wsl_report.py --zoo "checkpoints/zoo/*_best.pt" --pairs 4
"""

import argparse
import glob
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from minesweeper.env import MinesweeperEnv  # noqa: E402
from minesweeper.models import DQN, infer_width, pick_device  # noqa: E402
from minesweeper.eval import evaluate  # noqa: E402
from wsl.align import weight_matching_restarts, apply_perms, interpolate  # noqa: E402
from wsl.spectra import layer_spectra, LAYERS  # noqa: E402


def eval_sd(sd, env, episodes, device, seed):
    model = DQN(width=infer_width(sd)).to(device)
    model.load_state_dict(sd)
    model.eval()
    return evaluate(model, env, episodes, device, seed=seed)


def barrier_curves(sd_a, sd_b, env, episodes, device, lambdas, seed):
    perms, _, _ = weight_matching_restarts(sd_a, sd_b)
    sd_b_aligned = apply_perms(sd_b, perms)
    out = {"naive": [], "aligned": []}
    for lam in lambdas:
        r, w = eval_sd(interpolate(sd_a, sd_b, lam), env, episodes, device, seed)
        out["naive"].append({"lam": lam, "avg_reward": r, "win_rate": w})
        r, w = eval_sd(interpolate(sd_a, sd_b_aligned, lam), env, episodes, device, seed)
        out["aligned"].append({"lam": lam, "avg_reward": r, "win_rate": w})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zoo", default="checkpoints/zoo/*_best.pt")
    p.add_argument("--pairs", type=int, default=4)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--rows", type=int, default=8)
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--mines", type=int, default=10)
    p.add_argument("--lambdas", default="0,0.25,0.5,0.75,1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    device = pick_device(args.device)
    paths = sorted(q for pat in args.zoo.split(",") for q in glob.glob(pat))
    if len(paths) < 2:
        sys.exit(f"need at least 2 checkpoints matching {args.zoo!r}, found {len(paths)}")
    lambdas = [float(v) for v in args.lambdas.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sds = {p_: torch.load(p_, map_location="cpu") for p_ in paths}
    env = MinesweeperEnv(n_rows=args.rows, n_cols=args.cols, num_mines=args.mines)

    pairs = list(itertools.combinations(paths, 2))[:args.pairs]
    all_results = []
    for a, b in pairs:
        name = f"{Path(a).stem} vs {Path(b).stem}"
        print(f"--- {name}", flush=True)
        res = barrier_curves(sds[a], sds[b], env, args.episodes, device, lambdas, args.seed)
        res["pair"] = name
        all_results.append(res)
        for kind in ("naive", "aligned"):
            wr = [f"{r['win_rate']:.2%}" for r in res[kind]]
            print(f"  {kind:8s} win rates over lambda {lambdas}: {wr}", flush=True)

    with open(out_dir / "wsl_barriers.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # figure 1: barrier curves (mean over pairs)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for kind, style in (("naive", "--o"), ("aligned", "-s")):
        wins = np.array([[r["win_rate"] for r in res[kind]] for res in all_results])
        rews = np.array([[r["avg_reward"] for r in res[kind]] for res in all_results])
        axes[0].plot(lambdas, wins.mean(0), style, label=f"{kind} (n={len(all_results)} pairs)")
        axes[1].plot(lambdas, rews.mean(0), style, label=kind)
    axes[0].set_xlabel("interpolation $\\lambda$"); axes[0].set_ylabel("eval win rate")
    axes[1].set_xlabel("interpolation $\\lambda$"); axes[1].set_ylabel("eval avg reward")
    axes[0].set_title("Linear interpolation between independently trained DQNs")
    axes[1].set_title("(naive vs permutation-aligned)")
    for ax in axes:
        ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "wsl_barriers.png", dpi=160)
    print(f"wrote {out_dir/'wsl_barriers.png'}")

    # figure 2: per-layer SVD spectra across the zoo
    fig, axes = plt.subplots(1, len(LAYERS), figsize=(4 * len(LAYERS), 3.2))
    for ax, layer in zip(axes, LAYERS):
        for p_ in paths:
            s = layer_spectra(sds[p_], [layer])[layer]
            ax.semilogy(s / s[0], alpha=0.6, lw=1)
        ax.set_title(layer, fontsize=9)
        ax.set_xlabel("index"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("normalised singular value")
    fig.suptitle(f"Per-layer weight spectra across {len(paths)} independently trained agents")
    fig.tight_layout()
    fig.savefig(out_dir / "wsl_spectra.png", dpi=160)
    print(f"wrote {out_dir/'wsl_spectra.png'}")


if __name__ == "__main__":
    main()
