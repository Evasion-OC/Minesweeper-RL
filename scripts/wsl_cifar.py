"""CIFAR width-arm analysis: the inversion's third family.

Per width multiplier: test-accuracy barriers (naive, aligned-with-restarts,
aligned + BN-reset REPAIR at interior points) over every pair; assignment
margins per axis at the converged matching on every pair; churn on a
subsample; degeneracy spectra for finals and inits; init-pair margins for
the order-statistics purge.

    python3 scripts/wsl_cifar.py --mult 0.25
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
from torchvision import datasets, transforms  # noqa: E402

from minesweeper.models import pick_device  # noqa: E402
from wsl.align import interpolate  # noqa: E402
from wsl.barrier import barrier  # noqa: E402
from wsl.degeneracy import degeneracy_stats  # noqa: E402
from wsl.instability import lap_gap  # noqa: E402
from wsl.cifarzoo import (AXIS_LAYER_CIFAR, CIFAR_AXES, CifarVGG,  # noqa: E402
                          apply_perms_cifar, axis_cost_matrix_cifar,
                          live_masks_cifar, reset_bn_stats,
                          weight_matching_cifar_restarts)
from wsl.spectra import layer_spectra  # noqa: E402

LAMS = [0.0, 0.25, 0.5, 0.75, 1.0]


def margins_for(sd_a, sd_b, perms):
    live_a, live_b = live_masks_cifar(sd_a), live_masks_cifar(sd_b)
    out = {}
    for ax in CIFAR_AXES:
        C = axis_cost_matrix_cifar(sd_a, sd_b, ax, perms)
        C = C[np.ix_(live_a[ax], live_b[ax])]
        v1, v2 = lap_gap(C)
        out[ax] = {"rel_gap": (v1 - v2) / max(abs(v1), 1e-12)}
    return out


def churn_for(sd_a, sd_b, base, eps=1e-2, draws=10, restarts=5, rng_seed=0):
    live_a, live_b = live_masks_cifar(sd_a), live_masks_cifar(sd_b)
    gen = torch.Generator().manual_seed(rng_seed)
    churn = {ax: [] for ax in CIFAR_AXES}
    for _ in range(draws):
        sd_bp = {}
        for k, v in sd_b.items():
            if not v.dtype.is_floating_point:
                sd_bp[k] = v.clone()
                continue
            noise = torch.randn(v.shape, generator=gen, dtype=v.dtype)
            n = noise.norm()
            if n > 0:
                noise *= eps * v.norm() / n
            sd_bp[k] = v + noise
        perms, _, _ = weight_matching_cifar_restarts(sd_a, sd_bp, n_restarts=restarts)
        for ax in CIFAR_AXES:
            el = live_a[ax] & live_b[ax][base[ax]]
            churn[ax].append(float(np.mean(perms[ax][el] != base[ax][el]))
                             if el.any() else 0.0)
    return {ax: float(np.mean(churn[ax])) for ax in CIFAR_AXES}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mult", type=float, required=True)
    p.add_argument("--churn-pairs", type=int, default=10)
    p.add_argument("--init-pairs", type=int, default=15)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    tag = str(args.mult).replace(".", "p")
    ckpt_dir = f"checkpoints/cifar_m{tag}"
    out_path = Path(args.out or f"results/wsl_cifar_m{tag}.json")
    device = pick_device(args.device)

    norm = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2470, 0.2435, 0.2616))
    test_tf = transforms.Compose([transforms.ToTensor(), norm])
    test_ds = datasets.CIFAR10(args.data_dir, train=False, download=True,
                               transform=test_tf)
    xs = torch.stack([test_ds[i][0] for i in range(len(test_ds))])
    ys = torch.tensor([test_ds[i][1] for i in range(len(test_ds))])
    train_ds = datasets.CIFAR10(args.data_dir, train=True, download=True,
                                transform=test_tf)
    bn_loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True,
                                            generator=torch.Generator().manual_seed(0))

    finals = {Path(q).stem: torch.load(q, map_location="cpu")
              for q in sorted(glob.glob(f"{ckpt_dir}/*_final.pt"))}
    inits = {Path(q).stem: torch.load(q, map_location="cpu")
             for q in sorted(glob.glob(f"{ckpt_dir}/*_init.pt"))}
    model = CifarVGG(mult=args.mult).to(device)

    def test_acc(sd):
        model.load_state_dict(sd)
        model.eval()
        correct = 0
        with torch.no_grad():
            for i in range(0, xs.shape[0], 2048):
                pred = model(xs[i:i + 2048].to(device)).argmax(1).cpu()
                correct += (pred == ys[i:i + 2048]).sum().item()
        return correct / xs.shape[0]

    def test_acc_bnreset(sd):
        model.load_state_dict(sd)
        reset_bn_stats(model, bn_loader, device)
        correct = 0
        with torch.no_grad():
            for i in range(0, xs.shape[0], 2048):
                pred = model(xs[i:i + 2048].to(device)).argmax(1).cpu()
                correct += (pred == ys[i:i + 2048]).sum().item()
        return correct / xs.shape[0]

    layers = list(AXIS_LAYER_CIFAR.values())
    deg = {}
    for pool in (finals, inits):
        for stem, sd in pool.items():
            spec = layer_spectra(sd, layers)
            deg[stem] = {ax: degeneracy_stats(spec[l])
                         for ax, l in AXIS_LAYER_CIFAR.items()}

    pairs = list(itertools.combinations(sorted(finals), 2))
    rng = np.random.default_rng(args.seed)
    churn_idx = set(rng.choice(len(pairs), size=min(args.churn_pairs, len(pairs)),
                               replace=False).tolist())
    results = []
    for k, (a, b) in enumerate(pairs):
        sd_a, sd_b = finals[a], finals[b]
        perms, _, objs = weight_matching_cifar_restarts(sd_a, sd_b)
        sd_bal = apply_perms_cifar(sd_b, perms)
        naive = [test_acc(interpolate(sd_a, sd_b, lam)) for lam in LAMS]
        aligned = [test_acc(interpolate(sd_a, sd_bal, lam)) for lam in LAMS]
        repair = [aligned[0]] + [test_acc_bnreset(interpolate(sd_a, sd_bal, lam))
                                 for lam in (0.25, 0.5, 0.75)] + [aligned[-1]]
        entry = {"pair": f"{a} vs {b}", "a": a, "b": b,
                 "naive_acc": naive, "aligned_acc": aligned, "repair_acc": repair,
                 "naive_barrier": barrier(naive, LAMS),
                 "aligned_barrier": barrier(aligned, LAMS),
                 "repair_barrier": barrier(repair, LAMS),
                 "assignment_gap": margins_for(sd_a, sd_b, perms),
                 "objectives": {"min": min(objs), "max": max(objs)}}
        if k in churn_idx:
            entry["churn"] = churn_for(sd_a, sd_b, perms)
        results.append(entry)
        print(f"pair {k + 1}/{len(pairs)} {a} vs {b} naive mid {naive[2]:.3f} "
              f"aligned mid {aligned[2]:.3f} repair mid {repair[2]:.3f}"
              + ("  [churned]" if k in churn_idx else ""), flush=True)

    init_pairs = list(itertools.combinations(sorted(inits), 2))
    idx = rng.choice(len(init_pairs), size=min(args.init_pairs, len(init_pairs)),
                     replace=False)
    init_margins = {ax: [] for ax in CIFAR_AXES}
    for i in idx:
        a, b = init_pairs[i]
        perms, _, _ = weight_matching_cifar_restarts(inits[a], inits[b])
        m = margins_for(inits[a], inits[b], perms)
        for ax in CIFAR_AXES:
            init_margins[ax].append(m[ax]["rel_gap"])
    init_margins = {ax: float(np.median(init_margins[ax])) for ax in CIFAR_AXES}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"config": vars(args), "degeneracy": deg, "pairs": results,
                   "init_margins": init_margins}, f, indent=2)
    print(f"wrote {out_path}\n", flush=True)

    nb = np.median([e["naive_barrier"]["rel_barrier"] for e in results])
    ab = np.median([e["aligned_barrier"]["rel_barrier"] for e in results])
    rb = np.median([e["repair_barrier"]["rel_barrier"] for e in results])
    marg = np.median([e["assignment_gap"][ax]["rel_gap"]
                      for e in results for ax in CIFAR_AXES])
    print(f"mult {args.mult}: naive {nb:.3f}  aligned {ab:.3f}  +BNreset {rb:.3f}  "
          f"median margin {marg:.2e}  init margin {np.median(list(init_margins.values())):.2e}",
          flush=True)


if __name__ == "__main__":
    main()
