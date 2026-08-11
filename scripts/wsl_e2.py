"""E2 alignment suite: the barrier table.

For every pair: three matching arms (naive, weight matching with restarts,
activation matching on the frozen probe buffer), each with and without
REPAIR. Barriers on two curves per the brief: greedy win rate over the
seeded board set, and Q-value MSE against the endpoints on the same probe
buffer. One barrier convention throughout (wsl.barrier).

REPAIR targets follow the arm's permutation: after aligning B, unit k of
the aligned network is B's unit perm[k], so its target statistics are
B's unit perm[k]'s statistics.

    python3 scripts/wsl_e2.py --episodes 200
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
from minesweeper.models import DQN, infer_width, load_dqn, pick_device  # noqa: E402
from wsl.actmatch import activation_matching  # noqa: E402
from wsl.activations import AXIS_MODULE  # noqa: E402
from wsl.align import apply_perms, interpolate, weight_matching_restarts  # noqa: E402
from wsl.barrier import barrier, collect_probe_states, q_mse_curve  # noqa: E402
from wsl.repair import HIDDEN, repair, unit_stats  # noqa: E402

LAMS = [0.0, 0.25, 0.5, 0.75, 1.0]
INTERIOR = [0.25, 0.5, 0.75]

W1PILOT = ",".join(f"checkpoints/zoo/zoo_seed{s}_best.pt" for s in range(7, 17))


def permute_stats(stats, perms):
    out = {}
    for ax, layer in AXIS_MODULE.items():
        idx = torch.as_tensor(perms[ax])
        out[layer] = (stats[layer][0][idx], stats[layer][1][idx])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zoo", default=W1PILOT, help="comma-separated globs")
    p.add_argument("--pairs", type=int, default=0)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--probe-states", type=int, default=2048)
    p.add_argument("--reference", default="checkpoints/plain_seed0_long_best.pt")
    p.add_argument("--rows", type=int, default=8)
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--mines", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/wsl_e2_table.json")
    args = p.parse_args()

    device = pick_device(args.device)
    env = MinesweeperEnv(n_rows=args.rows, n_cols=args.cols, num_mines=args.mines)
    ref = load_dqn(args.reference, device)
    states = collect_probe_states(env, args.probe_states, seed=args.seed,
                                  policy=ref, device=device)
    print(f"probe buffer: {states[0].shape[0]} states", flush=True)

    paths = sorted(q for pat in args.zoo.split(",") for q in glob.glob(pat))
    sds = {Path(q).stem: torch.load(q, map_location="cpu") for q in paths}
    width = infer_width(next(iter(sds.values())))

    def make_model(sd):
        m = DQN(width=width).to(device)
        m.load_state_dict(sd)
        m.eval()
        return m

    models = {k: make_model(sd) for k, sd in sds.items()}
    stats = {k: unit_stats(models[k], states, device, HIDDEN) for k in sds}
    eval_model = DQN(width=width).to(device)

    def win_at(sd):
        eval_model.load_state_dict(sd)
        eval_model.eval()
        _, w = evaluate(eval_model, env, args.episodes, device, seed=args.seed)
        return w

    pairs = list(itertools.combinations(sorted(sds), 2))
    if args.pairs:
        pairs = pairs[:args.pairs]

    results = []
    for k, (a, b) in enumerate(pairs):
        sd_a, sd_b = sds[a], sds[b]
        w_a, w_b = win_at(sd_a), win_at(sd_b)
        perms_w, _, _ = weight_matching_restarts(sd_a, sd_b)
        perms_act = activation_matching(models[a], models[b], states, device)
        identity = {ax: np.arange(len(perms_w[ax])) for ax in perms_w}
        arms = {"naive": identity, "weight": perms_w, "act": perms_act}

        entry = {"pair": f"{a} vs {b}", "a": a, "b": b, "endpoints": [w_a, w_b],
                 "win": {}, "q_mse": {}}
        for arm, perms in arms.items():
            sd_bv = apply_perms(sd_b, perms)
            stats_bv = permute_stats(stats[b], perms)
            raw_curve, rep_curve = [w_a], [w_a]
            for lam in INTERIOR:
                mid = interpolate(sd_a, sd_bv, lam)
                raw_curve.append(win_at(mid))
                fixed = repair(mid, stats[a], stats_bv, lam, make_model,
                               states, device)
                rep_curve.append(win_at(fixed))
            raw_curve.append(w_b)
            rep_curve.append(w_b)
            entry["win"][arm] = {"curve": raw_curve,
                                 "barrier": barrier(raw_curve, LAMS)}
            entry["win"][arm + "+repair"] = {"curve": rep_curve,
                                             "barrier": barrier(rep_curve, LAMS)}
            entry["q_mse"][arm] = q_mse_curve(sd_a, sd_bv, eval_model, states,
                                              device, LAMS)
        results.append(entry)
        mids = "  ".join(f"{arm}:{entry['win'][arm]['curve'][2]:.2%}"
                         f"/{entry['win'][arm + '+repair']['curve'][2]:.2%}"
                         for arm in arms)
        print(f"pair {k + 1}/{len(pairs)} {a} vs {b}  midpoint raw/repair  {mids}",
              flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"config": vars(args), "pairs": results}, f, indent=2)
    print(f"wrote {out_path}\n", flush=True)

    print(f"E2 barrier table over {len(results)} pairs "
          "(median win-rate rel barrier | mean midpoint win):")
    for arm in ("naive", "weight", "act"):
        for suffix in ("", "+repair"):
            key = arm + suffix
            rel = np.median([e["win"][key]["barrier"]["rel_barrier"] for e in results])
            mid = np.mean([e["win"][key]["curve"][2] for e in results])
            print(f"  {key:14s} rel barrier {rel:.3f}   midpoint {mid:.2%}", flush=True)


if __name__ == "__main__":
    main()
