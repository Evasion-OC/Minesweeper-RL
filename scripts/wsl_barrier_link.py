"""The barrier link: does per-axis instability predict each axis's
contribution to the interpolation barrier?

For every pair: match with restarts, then measure the barrier with all axes
aligned and with each axis in turn left unaligned (wsl.barrier). The
contribution of an axis is how much worse the barrier gets without it. Two
curves, per the brief: Q-value MSE against the endpoints on a frozen probe
buffer (cheap, every pair) and greedy win rate (expensive, first
--winrate-pairs pairs). The probe buffer comes from a reference agent that
is in no population being compared.

If an E3 sweep JSON for the same population is given, prints the Spearman
of per-axis instability against per-axis barrier contribution, which is the
second half of the E3 primary analysis.

    python3 scripts/wsl_barrier_link.py --zoo "checkpoints/zoo/*_best.pt" \
        --e3 results/wsl_e3r_zoo16.json
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
from minesweeper.eval import evaluate  # noqa: E402
from minesweeper.models import DQN, infer_width, load_dqn, pick_device  # noqa: E402
from wsl.align import PERM_AXES, apply_perms, interpolate, weight_matching_restarts  # noqa: E402
from wsl.barrier import (barrier, collect_probe_states, leave_one_axis_perms,  # noqa: E402
                         q_mse_curve)

LAMS = [0.0, 0.25, 0.5, 0.75, 1.0]


def win_curve(sd_a, sd_b_variant, env, episodes, device, seed, model):
    out = []
    for lam in LAMS:
        model.load_state_dict(interpolate(sd_a, sd_b_variant, lam))
        model.eval()
        _, w = evaluate(model, env, episodes, device, seed=seed)
        out.append(w)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zoo", default="checkpoints/zoo/*_best.pt")
    p.add_argument("--pairs", type=int, default=0, help="0 = all pairs")
    p.add_argument("--winrate-pairs", type=int, default=15)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--probe-states", type=int, default=2048)
    p.add_argument("--reference", default="checkpoints/plain_seed0_long_best.pt",
                   help="agent for the probe buffer; outside every population")
    p.add_argument("--e3", default=None, help="matching E3 sweep JSON for the correlation")
    p.add_argument("--rows", type=int, default=8)
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--mines", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/wsl_barrier_link.json")
    args = p.parse_args()

    device = pick_device(args.device)
    env = MinesweeperEnv(n_rows=args.rows, n_cols=args.cols, num_mines=args.mines)
    ref = load_dqn(args.reference, device)
    states = collect_probe_states(env, args.probe_states, seed=args.seed,
                                  policy=ref, device=device)
    print(f"probe buffer: {states[0].shape[0]} states from {args.reference}", flush=True)

    paths = sorted(glob.glob(args.zoo))
    sds = {Path(q).stem: torch.load(q, map_location="cpu") for q in paths}
    model = DQN(width=infer_width(next(iter(sds.values())))).to(device)
    pairs = list(itertools.combinations(sorted(sds), 2))
    if args.pairs:
        pairs = pairs[:args.pairs]

    results = []
    for k, (a, b) in enumerate(pairs):
        sd_a, sd_b = sds[a], sds[b]
        perms, _, _ = weight_matching_restarts(sd_a, sd_b)

        def mse_curve(x, y, lams):
            return q_mse_curve(x, y, model, states, device, lams)

        entry = {"pair": f"{a} vs {b}", "a": a, "b": b, "q_mse": {}, "win": {}}
        cv = mse_curve(sd_a, apply_perms(sd_b, perms), LAMS)
        full = barrier(cv, LAMS, higher_is_better=False)
        full["curve"] = cv
        entry["q_mse"]["aligned"] = full
        for ax in PERM_AXES:
            held = apply_perms(sd_b, leave_one_axis_perms(perms, ax))
            cvh = mse_curve(sd_a, held, LAMS)
            bb = barrier(cvh, LAMS, higher_is_better=False)
            entry["q_mse"][ax] = {"barrier": bb["barrier"], "curve": cvh,
                                  "contribution": bb["barrier"] - full["barrier"]}
        if k < args.winrate_pairs and args.episodes > 0:
            wcv = win_curve(sd_a, apply_perms(sd_b, perms), env,
                            args.episodes, device, args.seed, model)
            wfull = barrier(wcv, LAMS)
            wfull["curve"] = wcv
            entry["win"]["aligned"] = wfull
            for ax in PERM_AXES:
                held = apply_perms(sd_b, leave_one_axis_perms(perms, ax))
                wcvh = win_curve(sd_a, held, env, args.episodes, device,
                                 args.seed, model)
                wb = barrier(wcvh, LAMS)
                entry["win"][ax] = {"barrier": wb["barrier"], "curve": wcvh,
                                    "contribution": wb["barrier"] - wfull["barrier"]}
        results.append(entry)
        contribs = " ".join(f"{ax}:{entry['q_mse'][ax]['contribution']:+.3f}"
                            for ax in PERM_AXES)
        print(f"pair {k + 1}/{len(pairs)} {a} vs {b}  q-mse contribution {contribs}",
              flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"config": vars(args), "pairs": results}, f, indent=2)
    print(f"wrote {out_path}", flush=True)

    if args.e3:
        with open(args.e3) as f:
            e3 = {e["pair"]: e for e in json.load(f)["pairs"]}
        for probe, get in (("churn", lambda e, ax: e["perturbation"][ax]["churn_mean"]),
                           ("lap_rel_gap", lambda e, ax: e["assignment_gap"][ax]["rel_gap"])):
            x, y = [], []
            for entry in results:
                if entry["pair"] not in e3:
                    continue
                for ax in PERM_AXES:
                    x.append(get(e3[entry["pair"]], ax))
                    y.append(entry["q_mse"][ax]["contribution"])
            rho, pv = sps.spearmanr(x, y)
            print(f"instability ({probe}) vs q-mse barrier contribution over "
                  f"{len(x)} (pair, axis) points: rho {rho:+.3f}  p {pv:.5f}", flush=True)


if __name__ == "__main__":
    main()
