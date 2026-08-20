"""E6 payoff: diagnostic-guided mixed-precision quantization.

Allocation rule, fixed in advance with no tuning: within each network, rank
the quantizable layers by the endpoint min-relative-gap; the sharper half
(fragile, by E6's measured sign) gets int8, the more degenerate half
(robust) gets int4. Arms: uniform int8, guided, anti (the allocation
swapped: the adversarial control), uniform int4. All per-tensor, whole
network quantized at once, BatchNorm fp32 as everywhere. Layers with no
defined gap (rank-one output heads) get int8 in guided and anti alike, so
the two arms differ only where the diagnostic speaks. Yardsticks as in E6:
normalised frozen-buffer Q-MSE (RL), test-accuracy drop (supervised);
average bits reported weighted by parameter count.

    python3 scripts/wsl_e6_mixed.py --family cifar
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from wsl.degeneracy import degeneracy_stats  # noqa: E402
from wsl.quantize import fake_quantize  # noqa: E402
from wsl.spectra import layer_spectra  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wsl_e6 import load_family  # noqa: E402


def quantize_alloc(sd, alloc):
    out = dict(sd)
    for name, bits in alloc.items():
        out[name] = fake_quantize(sd[name], bits, per_channel=False)
    return out


def allocations(sd, layers):
    spectra = layer_spectra(sd, layers)
    gaps = {n: degeneracy_stats(spectra[n])["min_rel_gap"] for n in layers}
    ranked = [n for n in layers if np.isfinite(gaps[n])]
    ranked.sort(key=lambda n: gaps[n], reverse=True)  # sharpest first
    n_sharp = len(ranked) // 2
    sharp, degen = set(ranked[:n_sharp]), set(ranked[n_sharp:])
    heads = [n for n in layers if not np.isfinite(gaps[n])]
    guided = {**{n: 8 for n in sharp}, **{n: 4 for n in degen},
              **{n: 8 for n in heads}}
    anti = {**{n: 4 for n in sharp}, **{n: 8 for n in degen},
            **{n: 8 for n in heads}}
    return {"uniform8": {n: 8 for n in layers}, "guided": guided,
            "anti": anti, "uniform4": {n: 4 for n in layers}}


def avg_bits(sd, alloc):
    tot = sum(sd[n].numel() for n in alloc)
    return sum(bits * sd[n].numel() for n, bits in alloc.items()) / tot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", required=True, choices=("dqn", "mlp", "cifar"))
    p.add_argument("--data-dir", default="data")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    # reuse wsl_e6's family loading and metric machinery by importing its
    # main-scope pieces: simplest is to duplicate the small metric closures
    import glob

    from minesweeper.env import MinesweeperEnv
    from minesweeper.models import DQN, infer_width, load_dqn, pick_device
    from wsl.barrier import collect_probe_states, q_values
    from wsl.cifarzoo import CifarVGG
    from wsl.mlpzoo import MLP

    device = pick_device(args.device)
    pats, layers = load_family(args.family)
    paths = sorted(q for pat in pats for q in glob.glob(pat))
    out_path = Path(args.out or f"results/wsl_e6_mixed_{args.family}.json")

    if args.family == "dqn":
        env = MinesweeperEnv(n_rows=8, n_cols=8, num_mines=10)
        ref = load_dqn("checkpoints/plain_seed0_long_best.pt", device)
        states = collect_probe_states(env, 2048, seed=args.seed, policy=ref,
                                      device=device)

        def make_model(sd):
            m = DQN(width=infer_width(sd)).to(device)
            m.load_state_dict(sd)
            m.eval()
            return m

        def degradation(sd, model, base):
            q = q_values(model, sd, states, device)
            return float(((q - base) ** 2).mean() / base.var())

        def baseline(sd, model):
            return q_values(model, sd, states, device)
    else:
        from torchvision import datasets, transforms
        if args.family == "mlp":
            tf = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))])
            ds = datasets.MNIST(args.data_dir, train=False, download=True, transform=tf)
        else:
            tf = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                          (0.2470, 0.2435, 0.2616))])
            ds = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=tf)
        xs = torch.stack([ds[i][0] for i in range(len(ds))])
        ys = torch.tensor([ds[i][1] for i in range(len(ds))])

        def make_model(sd):
            if args.family == "mlp":
                m = MLP(hidden=sd["fc1.weight"].shape[0]).to(device)
            else:
                m = CifarVGG(mult=sd["conv1.weight"].shape[0] / 64).to(device)
            m.load_state_dict(sd)
            m.eval()
            return m

        def acc(sd, model):
            model.load_state_dict(sd)
            model.eval()
            correct = 0
            with torch.no_grad():
                for i in range(0, xs.shape[0], 2048):
                    pred = model(xs[i:i + 2048].to(device)).argmax(1).cpu()
                    correct += (pred == ys[i:i + 2048]).sum().item()
            return correct / xs.shape[0]

        def degradation(sd, model, base):
            return float(base - acc(sd, model))

        def baseline(sd, model):
            return acc(sd, model)

    results = {}
    for k, path in enumerate(paths):
        sd = torch.load(path, map_location="cpu")
        model = make_model(sd)
        base = baseline(sd, model)
        allocs = allocations(sd, layers)
        entry = {}
        for arm, alloc in allocs.items():
            entry[arm] = {"degradation": degradation(quantize_alloc(sd, alloc),
                                                     model, base),
                          "avg_bits": avg_bits(sd, alloc)}
        results[Path(path).stem] = entry
        if (k + 1) % 20 == 0 or k == len(paths) - 1:
            print(f"{k + 1}/{len(paths)} done", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)
    print(f"wrote {out_path}", flush=True)

    for arm in ("uniform8", "guided", "anti", "uniform4"):
        d = np.mean([r[arm]["degradation"] for r in results.values()])
        b = np.mean([r[arm]["avg_bits"] for r in results.values()])
        print(f"  {arm:9s} mean degradation {d:.4f} at {b:.2f} avg bits", flush=True)


if __name__ == "__main__":
    main()
