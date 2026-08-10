"""E3 sweep: per-layer degeneracy statistics and per-axis instability probes
over every checkpoint pair in a zoo.

For each checkpoint: the two pre-committed degeneracy statistics per layer
(wsl.degeneracy). For each pair: the three instability probes per axis
(wsl.instability), all at the converged weight matching. Optionally, with
--restart-episodes > 0, the realised midpoint win rate per restart, whose
variance is the eval-based form of the restart probe.

Prints a Spearman preview of degeneracy against instability pooled over
(pair, axis) points. That preview is a smoke signal, not the analysis; the
paper's correlation runs on the full zoos with confidence intervals and a
permutation test.

    python3 scripts/wsl_e3.py --zoo "checkpoints/zoo/*_best.pt"
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
from scipy import stats as sps  # noqa: E402

from minesweeper.env import MinesweeperEnv  # noqa: E402
from minesweeper.models import DQN, infer_width, pick_device  # noqa: E402
from minesweeper.eval import evaluate  # noqa: E402
from wsl.align import PERM_AXES, weight_matching_restarts, apply_perms, interpolate  # noqa: E402
from wsl.degeneracy import AXIS_LAYER, layer_degeneracy  # noqa: E402
from wsl.instability import (NOISE_DRAWS, NOISE_EPS, PERT_RESTARTS, RESTART_SEEDS,  # noqa: E402
                             assignment_gap, perturbation_sensitivity,
                             restart_disagreement)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zoo", default="checkpoints/zoo/*_best.pt")
    p.add_argument("--pairs", type=int, default=0, help="0 = all pairs")
    p.add_argument("--pair-start", type=int, default=0,
                   help="skip this many pairs first (for splitting a population)")
    p.add_argument("--eps", type=float, default=NOISE_EPS)
    p.add_argument("--draws", type=int, default=NOISE_DRAWS)
    p.add_argument("--restart-seeds", type=int, default=RESTART_SEEDS)
    p.add_argument("--pert-restarts", type=int, default=PERT_RESTARTS,
                   help="restarts inside the perturbation probe")
    p.add_argument("--restart-episodes", type=int, default=0,
                   help="> 0: also evaluate midpoint win rate per restart")
    p.add_argument("--rows", type=int, default=8)
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--mines", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/wsl_e3_zoo.json")
    args = p.parse_args()

    paths = sorted(glob.glob(args.zoo))
    if len(paths) < 2:
        sys.exit(f"need at least 2 checkpoints matching {args.zoo!r}, found {len(paths)}")
    sds = {Path(q).stem: torch.load(q, map_location="cpu") for q in paths}

    degeneracy = {stem: layer_degeneracy(sd) for stem, sd in sds.items()}
    for stem in sorted(degeneracy):
        d = degeneracy[stem]
        print(f"degeneracy {stem}: " + "  ".join(
            f"{ax}:H={d[l]['entropy']:.3f},g={d[l]['min_rel_gap']:.4f}"
            for ax, l in AXIS_LAYER.items()), flush=True)

    pairs = list(itertools.combinations(sorted(sds), 2))
    end = args.pair_start + args.pairs if args.pairs else None
    pairs = pairs[args.pair_start:end]

    env = model = None
    if args.restart_episodes > 0:
        env = MinesweeperEnv(n_rows=args.rows, n_cols=args.cols, num_mines=args.mines)
        model = DQN(width=infer_width(next(iter(sds.values())))).to(pick_device(args.device))

    results = []
    for a, b in pairs:
        sd_a, sd_b = sds[a], sds[b]
        perms0, _, objs0 = weight_matching_restarts(sd_a, sd_b, n_restarts=args.restart_seeds)
        pert = perturbation_sensitivity(sd_a, sd_b, base_perms=perms0,
                                        eps=args.eps, draws=args.draws,
                                        restarts=args.pert_restarts)
        gaps = assignment_gap(sd_a, sd_b, perms=perms0)
        restart, perm_list = restart_disagreement(
            sd_a, sd_b, seeds=tuple(range(args.restart_seeds)), base_perms=perms0)
        entry = {"pair": f"{a} vs {b}", "a": a, "b": b, "perturbation": pert,
                 "assignment_gap": gaps, "restart": restart,
                 "objectives": {"min": min(objs0), "max": max(objs0), "all": objs0}}
        if args.restart_episodes > 0:
            wins = []
            device = next(model.parameters()).device
            for perms in perm_list:
                mid = interpolate(sd_a, apply_perms(sd_b, perms), 0.5)
                model.load_state_dict(mid)
                model.eval()
                _, w = evaluate(model, env, args.restart_episodes, device, seed=args.seed)
                wins.append(w)
            entry["restart_midpoint_win"] = {"wins": wins,
                                             "std": float(np.std(wins))}
        results.append(entry)
        line = f"pair {a} vs {b}: "
        line += "  ".join(
            f"{p_}[churn={pert[p_]['churn_mean']:.3f} relgap={gaps[p_]['rel_gap']:.2e} "
            f"disagree={restart[p_]['disagree_frac']:.3f}]" for p_ in PERM_AXES)
        print(line, flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config": vars(args), "degeneracy": degeneracy, "pairs": results}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {out_path}", flush=True)

    # --- Spearman preview over (pair, axis) points
    rows = []
    for entry in results:
        for ax, layer in AXIS_LAYER.items():
            da, db = degeneracy[entry["a"]][layer], degeneracy[entry["b"]][layer]
            rows.append({
                "entropy": 0.5 * (da["entropy"] + db["entropy"]),
                "min_rel_gap": 0.5 * (da["min_rel_gap"] + db["min_rel_gap"]),
                "churn": entry["perturbation"][ax]["churn_mean"],
                "lap_rel_gap": entry["assignment_gap"][ax]["rel_gap"],
                "disagree": entry["restart"][ax]["disagree_frac"],
            })
    print(f"\nSpearman preview over {len(rows)} (pair, axis) points "
          "(endpoint-mean degeneracy vs instability):", flush=True)
    for x in ("entropy", "min_rel_gap"):
        for y in ("churn", "lap_rel_gap", "disagree"):
            xv = np.array([r[x] for r in rows])
            yv = np.array([r[y] for r in rows])
            rho, pval = sps.spearmanr(xv, yv)
            print(f"  {x:12s} vs {y:12s}  rho={rho:+.3f}  p={pval:.4f}", flush=True)


if __name__ == "__main__":
    main()
