"""N4: recompute the degeneracy statistics on activation covariances and ask
whether activations predict what weights do or do not.

Uses the same frozen probe buffer as the barrier link (same reference agent,
same seed, so the states are identical). For every checkpoint in the zoo and
its untrained frac000 partner population: activation-spectrum degeneracy per
axis. Then, against the corrected E3 sweep: (a) the per-pair between-layer
ranking support, side by side with the weight statistics; (b) the
within-axis Spearman, where the weight statistics were null.

    python3 scripts/wsl_n4.py --e3 results/wsl_e3r_zoo16.json
"""

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy import stats as sps  # noqa: E402

from minesweeper.env import MinesweeperEnv  # noqa: E402
from minesweeper.models import DQN, infer_width, load_dqn, pick_device  # noqa: E402
from wsl.activations import activation_degeneracy  # noqa: E402
from wsl.align import PERM_AXES  # noqa: E402
from wsl.barrier import collect_probe_states  # noqa: E402
from wsl.degeneracy import AXIS_LAYER  # noqa: E402


def rank_table(e3, stats_by_ckpt, label):
    """Per-pair between-layer ranking of a degeneracy statistic against the
    three probes, plus the within-axis Spearman."""
    axes = list(PERM_AXES)
    probes = {"churn": lambda e, ax: e["perturbation"][ax]["churn_mean"],
              "lap_rel_gap": lambda e, ax: e["assignment_gap"][ax]["rel_gap"],
              "disagree": lambda e, ax: e["restart"][ax]["disagree_frac"]}
    expect = {"churn": -1, "lap_rel_gap": +1, "disagree": -1}
    print(f"--- {label}: min_rel_gap, per-pair between-layer ranking "
          f"(n={len(e3['pairs'])})")
    for probe, get in probes.items():
        rhos = []
        for e in e3["pairs"]:
            x = [0.5 * (stats_by_ckpt[e["a"]][ax]["min_rel_gap"]
                        + stats_by_ckpt[e["b"]][ax]["min_rel_gap"]) for ax in axes]
            y = [get(e, ax) for ax in axes]
            rhos.append(sps.spearmanr(x, y)[0])
        rhos = np.array(rhos)
        exp = expect[probe]
        print(f"    vs {probe:12s} median rho {np.median(rhos):+.2f}  "
              f"right-signed {np.mean(np.sign(rhos) == exp):.0%} (predict {exp:+d})")
    print(f"--- {label}: within-axis Spearman (n=120 per axis)")
    for ax in axes:
        cells = []
        for probe, get in probes.items():
            x = [0.5 * (stats_by_ckpt[e["a"]][ax]["min_rel_gap"]
                        + stats_by_ckpt[e["b"]][ax]["min_rel_gap"])
                 for e in e3["pairs"]]
            y = [get(e, ax) for e in e3["pairs"]]
            rho, pv = sps.spearmanr(x, y)
            cells.append(f"{probe} {rho:+.2f} (p={pv:.3f})")
        print(f"    {ax}: " + "  ".join(cells))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zoo", default="checkpoints/zoo/*_best.pt")
    p.add_argument("--init-glob", default="checkpoints/zoo/*_frac000.pt")
    p.add_argument("--e3", default="results/wsl_e3r_zoo16.json")
    p.add_argument("--probe-states", type=int, default=2048)
    p.add_argument("--reference", default="checkpoints/plain_seed0_long_best.pt")
    p.add_argument("--rows", type=int, default=8)
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--mines", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/wsl_n4_activations.json")
    args = p.parse_args()

    device = pick_device(args.device)
    env = MinesweeperEnv(n_rows=args.rows, n_cols=args.cols, num_mines=args.mines)
    ref = load_dqn(args.reference, device)
    states = collect_probe_states(env, args.probe_states, seed=args.seed,
                                  policy=ref, device=device)
    print(f"probe buffer: {states[0].shape[0]} states", flush=True)

    stats = {}
    for pattern in (args.zoo, args.init_glob):
        for path in sorted(glob.glob(pattern)):
            sd = torch.load(path, map_location="cpu")
            model = DQN(width=infer_width(sd)).to(device)
            model.load_state_dict(sd)
            stats[Path(path).stem] = activation_degeneracy(model, states, device)
    for stem in sorted(stats):
        s = stats[stem]
        print(f"act-degeneracy {stem}: " + "  ".join(
            f"{ax}:H={s[ax]['entropy']:.3f},g={s[ax]['min_rel_gap']:.4f},"
            f"r={s[ax]['live_rank']}" for ax in PERM_AXES), flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"config": vars(args), "stats": stats}, f, indent=2)
    print(f"wrote {out_path}\n", flush=True)

    with open(args.e3) as f:
        e3 = json.load(f)
    weight_stats = {ck: {ax: e3["degeneracy"][ck][AXIS_LAYER[ax]] for ax in PERM_AXES}
                    for ck in e3["degeneracy"]}
    rank_table(e3, weight_stats, "WEIGHT spectra (baseline)")
    print()
    rank_table(e3, stats, "ACTIVATION spectra (N4)")


if __name__ == "__main__":
    main()
