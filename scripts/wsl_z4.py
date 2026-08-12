"""Z4 analysis: the supervised control's version of E3 and the barrier table.

Degeneracy per hidden layer for every final and init checkpoint; the
instability probes (perturbation churn with restarts, assignment gap on the
live submatrix) on a seeded subsample of pairs, since churn dominates cost;
test-accuracy barrier curves naive versus weight-aligned on every pair.
Prints the between-layer per-pair ranking, the within-axis Spearman, and
the barrier table.

    python3 scripts/wsl_z4.py
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
from torchvision import datasets, transforms  # noqa: E402

from minesweeper.models import pick_device  # noqa: E402
from wsl.barrier import barrier  # noqa: E402
from wsl.degeneracy import degeneracy_stats  # noqa: E402
from wsl.instability import lap_gap  # noqa: E402
from wsl.align import interpolate  # noqa: E402
from wsl.mlpzoo import (AXIS_LAYER_MLP, MLP, MLP_AXES, apply_perms_mlp,  # noqa: E402
                        axis_cost_matrix_mlp, live_masks_mlp,
                        weight_matching_mlp_restarts)
from wsl.spectra import layer_spectra  # noqa: E402

LAMS = [0.0, 0.25, 0.5, 0.75, 1.0]


def churn_probe(sd_a, sd_b, base, eps=1e-2, draws=10, restarts=5, rng_seed=0):
    live_a, live_b = live_masks_mlp(sd_a), live_masks_mlp(sd_b)
    gen = torch.Generator().manual_seed(rng_seed)
    churn = {ax: [] for ax in MLP_AXES}
    for _ in range(draws):
        sd_bp = {}
        for k, v in sd_b.items():
            noise = torch.randn(v.shape, generator=gen, dtype=v.dtype)
            n = noise.norm()
            if n > 0:
                noise *= eps * v.norm() / n
            sd_bp[k] = v + noise
        perms, _, _ = weight_matching_mlp_restarts(sd_a, sd_bp, n_restarts=restarts)
        for ax in MLP_AXES:
            el = live_a[ax] & live_b[ax][base[ax]]
            churn[ax].append(float(np.mean(perms[ax][el] != base[ax][el]))
                             if el.any() else 0.0)
    return {ax: float(np.mean(churn[ax])) for ax in MLP_AXES}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt-dir", default="checkpoints/z4")
    p.add_argument("--probe-pairs", type=int, default=60)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/wsl_z4.json")
    args = p.parse_args()

    device = pick_device(args.device)
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    test_ds = datasets.MNIST(args.data_dir, train=False, download=True, transform=tf)
    xs = torch.stack([test_ds[i][0] for i in range(len(test_ds))]).to(device)
    ys = torch.tensor([test_ds[i][1] for i in range(len(test_ds))])

    finals = {Path(q).stem: torch.load(q, map_location="cpu")
              for q in sorted(glob.glob(f"{args.ckpt_dir}/*_final.pt"))}
    inits = {Path(q).stem: torch.load(q, map_location="cpu")
             for q in sorted(glob.glob(f"{args.ckpt_dir}/*_init.pt"))}
    hidden = finals[next(iter(finals))]["fc1.weight"].shape[0]
    model = MLP(hidden=hidden).to(device)

    def test_acc(sd):
        model.load_state_dict(sd)
        model.eval()
        correct = 0
        with torch.no_grad():
            for i in range(0, xs.shape[0], 4096):
                pred = model(xs[i:i + 4096]).argmax(1).cpu()
                correct += (pred == ys[i:i + 4096]).sum().item()
        return correct / xs.shape[0]

    layers = list(AXIS_LAYER_MLP.values())
    deg = {}
    for pool in (finals, inits):
        for stem, sd in pool.items():
            deg[stem] = {ax: degeneracy_stats(layer_spectra(sd, [l])[l])
                         for ax, l in AXIS_LAYER_MLP.items()}
    print(f"degeneracy done for {len(deg)} checkpoints "
          f"({len(finals)} finals, {len(inits)} inits)", flush=True)

    pairs = list(itertools.combinations(sorted(finals), 2))
    rng = np.random.default_rng(args.seed)
    probe_idx = set(rng.choice(len(pairs), size=min(args.probe_pairs, len(pairs)),
                               replace=False).tolist())

    results = []
    for k, (a, b) in enumerate(pairs):
        sd_a, sd_b = finals[a], finals[b]
        perms, _, objs = weight_matching_mlp_restarts(sd_a, sd_b)
        sd_bal = apply_perms_mlp(sd_b, perms)
        naive = [test_acc(interpolate(sd_a, sd_b, lam)) for lam in LAMS]
        aligned = [test_acc(interpolate(sd_a, sd_bal, lam)) for lam in LAMS]
        entry = {"pair": f"{a} vs {b}", "a": a, "b": b,
                 "naive_acc": naive, "aligned_acc": aligned,
                 "naive_barrier": barrier(naive, LAMS),
                 "aligned_barrier": barrier(aligned, LAMS),
                 "objectives": {"min": min(objs), "max": max(objs)}}
        if k in probe_idx:
            live_a, live_b = live_masks_mlp(sd_a), live_masks_mlp(sd_b)
            gaps = {}
            for ax in MLP_AXES:
                C = axis_cost_matrix_mlp(sd_a, sd_b, ax, perms)
                C = C[np.ix_(live_a[ax], live_b[ax])]
                v1, v2 = lap_gap(C)
                gaps[ax] = {"rel_gap": (v1 - v2) / max(abs(v1), 1e-12)}
            entry["assignment_gap"] = gaps
            entry["churn"] = churn_probe(sd_a, sd_b, perms)
        results.append(entry)
        if (k + 1) % 25 == 0 or k in probe_idx:
            print(f"pair {k + 1}/{len(pairs)} {a} vs {b} naive mid "
                  f"{naive[2]:.3f} aligned mid {aligned[2]:.3f}"
                  + ("  [probed]" if k in probe_idx else ""), flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"config": vars(args), "degeneracy": deg, "pairs": results}, f,
                  indent=2)
    print(f"wrote {out_path}\n", flush=True)

    nb = np.median([e["naive_barrier"]["rel_barrier"] for e in results])
    ab = np.median([e["aligned_barrier"]["rel_barrier"] for e in results])
    nm = np.mean([e["naive_acc"][2] for e in results])
    am = np.mean([e["aligned_acc"][2] for e in results])
    print(f"Z4 barrier table over {len(results)} pairs: naive rel barrier {nb:.3f} "
          f"(midpoint acc {nm:.3f}), aligned {ab:.3f} (midpoint acc {am:.3f})",
          flush=True)

    probed = [e for e in results if "churn" in e]
    print(f"\nbetween-layer per-pair ranking over {len(probed)} probed pairs "
          "(min_rel_gap vs probe, 3 axes per pair):", flush=True)
    for probe, get, exp in (("churn", lambda e, ax: e["churn"][ax], -1),
                            ("rel_gap", lambda e, ax: e["assignment_gap"][ax]["rel_gap"], +1)):
        rhos = []
        for e in probed:
            x = [0.5 * (deg[e["a"]][ax]["min_rel_gap"] + deg[e["b"]][ax]["min_rel_gap"])
                 for ax in MLP_AXES]
            y = [get(e, ax) for ax in MLP_AXES]
            rhos.append(sps.spearmanr(x, y)[0])
        rhos = np.array(rhos)
        print(f"  min_rel_gap vs {probe:8s} median rho {np.median(rhos):+.2f}  "
              f"right-signed {np.mean(np.sign(rhos) == exp):.0%} (predict {exp:+d})",
              flush=True)
    print("\nwithin-axis Spearman over probed pairs:", flush=True)
    for ax in MLP_AXES:
        x = [0.5 * (deg[e["a"]][ax]["min_rel_gap"] + deg[e["b"]][ax]["min_rel_gap"])
             for e in probed]
        y = [e["churn"][ax] for e in probed]
        rho, pv = sps.spearmanr(x, y)
        print(f"  {ax}: min_rel_gap vs churn rho {rho:+.3f} (p={pv:.3f})", flush=True)


if __name__ == "__main__":
    main()
