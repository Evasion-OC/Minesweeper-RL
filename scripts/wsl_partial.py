"""Diagnostic-guided partial alignment: align only the layers the diagnostic
marks reliable, and compare against full alignment, on win-rate curves.

Arms per pair (all from one restart-based matching):
  full:      align every axis;
  conv-only: align p1..p3, leave the value head (the tie-floor axis) alone;
  value-only: align p4 alone, the adversarial control;
  single:    full alignment from a single matching run (no restarts), the
             end-to-end cost of the silent solver failures.

    python3 scripts/wsl_partial.py --episodes 200
"""

import argparse
import glob
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from minesweeper.env import MinesweeperEnv  # noqa: E402
from minesweeper.eval import evaluate  # noqa: E402
from minesweeper.models import DQN, infer_width, pick_device  # noqa: E402
from wsl.align import (PERM_AXES, apply_perms, interpolate, weight_matching,  # noqa: E402
                       weight_matching_restarts)
from wsl.barrier import barrier  # noqa: E402

LAMS = [0.0, 0.25, 0.5, 0.75, 1.0]
W1PILOT = ",".join(f"checkpoints/zoo/zoo_seed{s}_best.pt" for s in range(7, 17))


def keep_axes(perms, axes):
    return {ax: (perms[ax] if ax in axes else np.arange(len(perms[ax])))
            for ax in perms}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zoo", default=W1PILOT)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--rows", type=int, default=8)
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--mines", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/wsl_partial.json")
    args = p.parse_args()

    device = pick_device(args.device)
    env = MinesweeperEnv(n_rows=args.rows, n_cols=args.cols, num_mines=args.mines)
    paths = sorted(q for pat in args.zoo.split(",") for q in glob.glob(pat))
    sds = {Path(q).stem: torch.load(q, map_location="cpu") for q in paths}
    model = DQN(width=infer_width(next(iter(sds.values())))).to(device)

    def win_at(sd):
        model.load_state_dict(sd)
        model.eval()
        _, w = evaluate(model, env, args.episodes, device, seed=args.seed)
        return w

    results = []
    pairs = list(itertools.combinations(sorted(sds), 2))
    for k, (a, b) in enumerate(pairs):
        sd_a, sd_b = sds[a], sds[b]
        perms, _, _ = weight_matching_restarts(sd_a, sd_b)
        single = weight_matching(sd_a, sd_b)
        arms = {"full": perms,
                "conv_only": keep_axes(perms, ("p1", "p2", "p3")),
                "value_only": keep_axes(perms, ("p4",)),
                "single": single}
        w_a, w_b = win_at(sd_a), win_at(sd_b)
        entry = {"pair": f"{a} vs {b}", "endpoints": [w_a, w_b], "arms": {}}
        for name, pm in arms.items():
            sd_bv = apply_perms(sd_b, pm)
            curve = [w_a] + [win_at(interpolate(sd_a, sd_bv, lam))
                             for lam in (0.25, 0.5, 0.75)] + [w_b]
            entry["arms"][name] = {"curve": curve, "barrier": barrier(curve, LAMS)}
        results.append(entry)
        mids = "  ".join(f"{n}:{entry['arms'][n]['curve'][2]:.2%}" for n in arms)
        print(f"pair {k + 1}/{len(pairs)} {a} vs {b}  midpoints  {mids}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out}\n", flush=True)
    for name in ("full", "conv_only", "value_only", "single"):
        mid = np.mean([e["arms"][name]["curve"][2] for e in results])
        rel = np.median([e["arms"][name]["barrier"]["rel_barrier"] for e in results])
        print(f"  {name:11s} midpoint {mid:.2%}   median rel barrier {rel:.3f}", flush=True)


if __name__ == "__main__":
    main()
