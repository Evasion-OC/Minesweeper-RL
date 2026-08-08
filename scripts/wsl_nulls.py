"""N5 pipeline sanity nulls for the weight-space study.

Two checks that must behave as expected before any result downstream is
trusted:

  self:  matching a checkpoint to itself must return the identity
         permutation on every live unit. Units whose weights have collapsed
         to numerical zero (dead units; Adam plus coupled weight decay
         drives them there) are an exact zero-cost tie class, and the
         assignment is arbitrary within it - that is expected, and the dead
         class size is itself worth reporting.
  traj:  matching a checkpoint to a later checkpoint of the same run must
         stay near the identity on live units (training does not permute
         neurons), and the interpolation curve must not dip below the
         worse endpoint.

    python3 scripts/wsl_nulls.py --episodes 200
"""

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from minesweeper.env import MinesweeperEnv  # noqa: E402
from minesweeper.models import DQN, pick_device  # noqa: E402
from minesweeper.eval import evaluate  # noqa: E402
from wsl.align import (PERM_SIZES, DEAD_NORM, unit_norms,  # noqa: E402
                       weight_matching, apply_perms, interpolate)

TRAJ_PAIRS = [
    ("checkpoints/zoo/zoo_seed%d_batch1.pt" % s, "checkpoints/zoo/zoo_seed%d_best.pt" % s)
    for s in range(1, 7)
] + [
    ("checkpoints/plain_seed0_long_batch1.pt", "checkpoints/plain_seed0_long_best.pt"),
]

# trajectory pairs that also get the (more expensive) interpolation curves
EVAL_PAIRS = {"zoo_seed1_batch1 vs zoo_seed1_best",
              "plain_seed0_long_batch1 vs plain_seed0_long_best"}


def identity_fraction(perms, live=None):
    """Per-axis and overall fraction of units mapped to themselves,
    optionally restricted to live units (live: dict axis -> bool mask)."""
    per, sizes = {}, {}
    for p in perms:
        keep = live[p] if live is not None else np.ones(PERM_SIZES[p], bool)
        n = int(keep.sum())
        per[p] = float(np.mean(perms[p][keep] == np.arange(PERM_SIZES[p])[keep])) if n else 1.0
        sizes[p] = n
    total = sum(sizes.values())
    overall = sum(per[p] * sizes[p] for p in perms) / total if total else 1.0
    return per, overall


def eval_sd(sd, env, episodes, device, seed):
    model = DQN().to(device)
    model.load_state_dict(sd)
    model.eval()
    return evaluate(model, env, episodes, device, seed=seed)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-glob", default="checkpoints/zoo/*_best.pt")
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
    lambdas = [float(v) for v in args.lambdas.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"self": [], "traj": []}

    # --- self-match: expected identity on every live unit; dead units are
    # an exact zero-cost tie class and may permute among themselves
    for path in sorted(glob.glob(args.self_glob)):
        sd = torch.load(path, map_location="cpu")
        live = {p: unit_norms(sd, p) > DEAD_NORM for p in PERM_SIZES}
        dead = {p: int(PERM_SIZES[p] - live[p].sum()) for p in PERM_SIZES}
        perms = weight_matching(sd, sd)
        per_live, overall_live = identity_fraction(perms, live)
        ok = overall_live == 1.0
        report["self"].append({"ckpt": Path(path).stem, "identity_live": per_live,
                               "dead_units": dead, "ok": ok})
        flags = "" if ok else "  <-- LIVE UNITS NOT IDENTITY, investigate"
        print(f"self  {Path(path).stem:24s} live-identity per axis "
              f"{[f'{per_live[k]:.3f}' for k in sorted(per_live)]} "
              f"dead {[dead[k] for k in sorted(dead)]}{flags}", flush=True)

    # --- same-run trajectory match: expected near-identity, near-equal curves
    env = MinesweeperEnv(n_rows=args.rows, n_cols=args.cols, num_mines=args.mines)
    for early, late in TRAJ_PAIRS:
        if not (Path(early).exists() and Path(late).exists()):
            print(f"traj  skipping missing pair {early} {late}", flush=True)
            continue
        name = f"{Path(early).stem} vs {Path(late).stem}"
        sd_a = torch.load(early, map_location="cpu")
        sd_b = torch.load(late, map_location="cpu")
        live = {p: (unit_norms(sd_a, p) > DEAD_NORM) | (unit_norms(sd_b, p) > DEAD_NORM)
                for p in PERM_SIZES}
        perms = weight_matching(sd_a, sd_b)
        per, overall = identity_fraction(perms)
        per_live, overall_live = identity_fraction(perms, live)
        entry = {"pair": name, "identity": per, "identity_overall": overall,
                 "identity_live": per_live, "identity_live_overall": overall_live}
        print(f"traj  {name}: identity overall {overall:.3f} "
              f"(live only {overall_live:.3f}), per axis "
              f"{[f'{per[k]:.3f}' for k in sorted(per)]}", flush=True)

        if name in EVAL_PAIRS and args.episodes > 0:
            sd_b_aligned = apply_perms(sd_b, perms)
            curves = {"naive": [], "aligned": []}
            for lam in lambdas:
                _, w = eval_sd(interpolate(sd_a, sd_b, lam), env, args.episodes, device, args.seed)
                curves["naive"].append({"lam": lam, "win_rate": w})
                _, w = eval_sd(interpolate(sd_a, sd_b_aligned, lam), env, args.episodes, device, args.seed)
                curves["aligned"].append({"lam": lam, "win_rate": w})
            entry["curves"] = curves
            for kind in ("naive", "aligned"):
                wr = [f"{r['win_rate']:.2%}" for r in curves[kind]]
                print(f"      {kind:8s} win rates over lambda {lambdas}: {wr}", flush=True)
        report["traj"].append(entry)

    out_path = out_dir / "wsl_nulls_n5.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
