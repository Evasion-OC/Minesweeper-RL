"""E6 sweep: per-layer quantization sensitivity across the existing zoos.

For every checkpoint and quantizable layer: fake-quantize that layer alone
under each scheme and measure degradation on the family's yardstick
(normalised frozen-buffer Q-MSE for the RL zoos; test-accuracy drop for
MNIST and CIFAR). Alongside each layer: the E3 degeneracy statistic and the
dynamic-range covariate. No new training; the zoos are bought twice.

    python3 scripts/wsl_e6.py --family mlp
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
from minesweeper.models import DQN, infer_width, load_dqn, pick_device  # noqa: E402
from wsl.barrier import collect_probe_states, q_values  # noqa: E402
from wsl.cifarzoo import CifarVGG  # noqa: E402
from wsl.degeneracy import degeneracy_stats  # noqa: E402
from wsl.mlpzoo import MLP  # noqa: E402
from wsl.quantize import SCHEMES, dynamic_range, quantize_layer  # noqa: E402
from wsl.spectra import layer_spectra  # noqa: E402

DQN_LAYERS = ["conv1.weight", "conv2.weight", "conv3.weight",
              "advantage_conv.weight", "value_fc1.weight", "value_fc2.weight"]
MLP_LAYERS = ["fc1.weight", "fc2.weight", "fc3.weight", "out.weight"]
CIFAR_LAYERS = [f"conv{i}.weight" for i in range(1, 7)] + ["fc.weight"]


def load_family(family):
    if family == "dqn":
        pats = ["checkpoints/zoo/zoo_seed*_best.pt",
                "checkpoints/z3/z3_w2_*_best.pt", "checkpoints/z3/z3_w4_*_best.pt"]
        return pats, DQN_LAYERS
    if family == "mlp":
        return ["checkpoints/z4/*_final.pt", "checkpoints/z4w32/*_final.pt",
                "checkpoints/z4w64/*_final.pt", "checkpoints/z4w128/*_final.pt"], MLP_LAYERS
    if family == "cifar":
        return ["checkpoints/cifar_m0p25/*_final.pt", "checkpoints/cifar_m0p5/*_final.pt",
                "checkpoints/cifar_m1p0/*_final.pt"], CIFAR_LAYERS
    raise ValueError(family)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", required=True, choices=("dqn", "mlp", "cifar"))
    p.add_argument("--probe-states", type=int, default=2048)
    p.add_argument("--reference", default="checkpoints/plain_seed0_long_best.pt")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = pick_device(args.device)
    pats, layers = load_family(args.family)
    paths = sorted(q for pat in pats for q in glob.glob(pat))
    out_path = Path(args.out or f"results/wsl_e6_{args.family}.json")

    if args.family == "dqn":
        env = MinesweeperEnv(n_rows=8, n_cols=8, num_mines=10)
        ref = load_dqn(args.reference, device)
        states = collect_probe_states(env, args.probe_states, seed=args.seed,
                                      policy=ref, device=device)

        def make_model(sd):
            m = DQN(width=infer_width(sd)).to(device)
            m.load_state_dict(sd)
            m.eval()
            return m

        def metric_fn(sd, model):
            return q_values(model, sd, states, device)

        def sensitivity(base, quant):
            return float(((quant - base) ** 2).mean() / base.var())
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

        def metric_fn(sd, model):
            model.load_state_dict(sd)
            model.eval()
            correct = 0
            with torch.no_grad():
                for i in range(0, xs.shape[0], 2048):
                    pred = model(xs[i:i + 2048].to(device)).argmax(1).cpu()
                    correct += (pred == ys[i:i + 2048]).sum().item()
            return correct / xs.shape[0]

        def sensitivity(base, quant):
            return float(base - quant)

    results = {}
    for k, path in enumerate(paths):
        sd = torch.load(path, map_location="cpu")
        stem = Path(path).stem
        model = make_model(sd)
        base = metric_fn(sd, model)
        spectra = layer_spectra(sd, layers)
        entry = {"layers": {}}
        if args.family != "dqn":
            entry["fp32_acc"] = base
        for name in layers:
            stats = degeneracy_stats(spectra[name])
            row = {"min_rel_gap": stats["min_rel_gap"],
                   "entropy": stats["entropy"],
                   "drange": dynamic_range(sd[name]),
                   "schemes": {}}
            for scheme in SCHEMES:
                sq = quantize_layer(sd, name, scheme)
                row["schemes"][scheme] = sensitivity(base, metric_fn(sq, model))
            entry["layers"][name] = row
        results[stem] = entry
        if (k + 1) % 10 == 0 or k == len(paths) - 1:
            print(f"{k + 1}/{len(paths)} checkpoints done", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
