"""Three-way ablation on identical eval seeds: plain FCN / plain+D4 TTA /
D4-equivariant network. Writes results/ablation.csv.

    python3 scripts/run_ablation.py --plain checkpoints/plain_seed0_best.pt \
        --equi checkpoints/equi_seed0_best.pt --episodes 500
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from minesweeper.env import MinesweeperEnv  # noqa: E402
from minesweeper.models import DQN, pick_device  # noqa: E402
from minesweeper.equivariant import EquivariantDQN  # noqa: E402
from minesweeper.eval import evaluate  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plain", required=True)
    p.add_argument("--equi", required=True)
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--sizes", default="8x8x10,16x16x40")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/ablation.csv")
    args = p.parse_args()

    device = pick_device(args.device)
    plain = DQN().to(device)
    plain.load_state_dict(torch.load(args.plain, map_location=device))
    plain.eval()
    equi = EquivariantDQN().to(device)
    equi.load_state_dict(torch.load(args.equi, map_location=device))
    equi.eval()

    rows = []
    for size in args.sizes.split(","):
        r, c, m = (int(v) for v in size.split("x"))
        env = MinesweeperEnv(n_rows=r, n_cols=c, num_mines=m)
        for label, model, tta in [("plain", plain, False),
                                  ("plain+tta", plain, True),
                                  ("equivariant", equi, False)]:
            avg_r, win = evaluate(model, env, args.episodes, device,
                                  tta=tta, seed=args.seed)
            rows.append({"board": size, "model": label,
                         "avg_reward": round(avg_r, 3), "win_rate": round(win, 4)})
            print(f"{size:10s} {label:12s} avg_reward={avg_r:8.2f} win={win:.2%}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
