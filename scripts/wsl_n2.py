"""N2, the noise-matched perturbation control (the merging brief's form:
noise matched in norm to the distance between members).

The aligned midpoint (A + P(B))/2 outperforms the naive midpoint. This
control asks whether that gain is the correspondence or mere proximity: for
each pair, evaluate A + xi where xi is Gaussian noise scaled, per parameter
tensor, to the norm of (P(B) - A)/2, i.e. a random point at the aligned
midpoint's distance from A. If the noise point matches the aligned
midpoint's win rate, alignment adds nothing beyond staying close to a
working network. Uses the same pairs, episode count and eval seed as the
width-1 barrier arm, so the three numbers are directly comparable.

    python3 scripts/wsl_n2.py --barriers results/wsl_barrier_w1/wsl_barriers.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from minesweeper.env import MinesweeperEnv  # noqa: E402
from minesweeper.eval import evaluate  # noqa: E402
from minesweeper.models import DQN, infer_width, pick_device  # noqa: E402
from wsl.align import apply_perms, weight_matching_restarts  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--barriers", default="results/wsl_barrier_w1/wsl_barriers.json")
    p.add_argument("--ckpt-dir", default="checkpoints/zoo")
    p.add_argument("--draws", type=int, default=3)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--rows", type=int, default=8)
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--mines", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/wsl_n2_noise_matched.json")
    args = p.parse_args()

    with open(args.barriers) as f:
        barr = json.load(f)
    device = pick_device(args.device)
    env = MinesweeperEnv(n_rows=args.rows, n_cols=args.cols, num_mines=args.mines)

    results = []
    gen = torch.Generator().manual_seed(args.seed)
    model = None
    for k, res in enumerate(barr):
        a, b = res["pair"].split(" vs ")
        sd_a = torch.load(f"{args.ckpt_dir}/{a}.pt", map_location="cpu")
        sd_b = torch.load(f"{args.ckpt_dir}/{b}.pt", map_location="cpu")
        if model is None:
            model = DQN(width=infer_width(sd_a)).to(device)
        perms, _, _ = weight_matching_restarts(sd_a, sd_b)
        sd_bp = apply_perms(sd_b, perms)

        wins = []
        for _ in range(args.draws):
            sd_noise = {}
            for key, va in sd_a.items():
                target = 0.5 * (sd_bp[key] - va).norm()
                xi = torch.randn(va.shape, generator=gen, dtype=va.dtype)
                n = xi.norm()
                if n > 0:
                    xi *= target / n
                sd_noise[key] = va + xi
            model.load_state_dict(sd_noise)
            model.eval()
            _, w = evaluate(model, env, args.episodes, device, seed=args.seed)
            wins.append(w)

        lams = [r["lam"] for r in res["naive"]]
        mid = lams.index(0.5)
        entry = {"pair": res["pair"],
                 "naive_mid": res["naive"][mid]["win_rate"],
                 "aligned_mid": res["aligned"][mid]["win_rate"],
                 "noise_mid_mean": float(np.mean(wins)),
                 "noise_mid": wins}
        results.append(entry)
        print(f"pair {k + 1}/{len(barr)} {res['pair']}: naive {entry['naive_mid']:.2%} "
              f"aligned {entry['aligned_mid']:.2%} noise {entry['noise_mid_mean']:.2%}",
              flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_path}", flush=True)

    nv = np.mean([r["naive_mid"] for r in results])
    al = np.mean([r["aligned_mid"] for r in results])
    no = np.mean([r["noise_mid_mean"] for r in results])
    beats = np.mean([r["aligned_mid"] > r["noise_mid_mean"] for r in results])
    print(f"\nN2 over {len(results)} pairs: naive midpoint {nv:.2%}, aligned midpoint "
          f"{al:.2%}, noise-matched control {no:.2%}; aligned beats its noise "
          f"control in {beats:.0%} of pairs", flush=True)


if __name__ == "__main__":
    main()
