"""E6 confirmatory arm: win-rate sensitivity for the base-width DQN zoo,
int4 per-tensor only, 1000 seeded episodes per configuration, with the
fp32 baseline evaluated on the same episodes. Binomial SE at 1000 episodes
and p near 0.1 is about 0.9 percentage points; that is the noise floor.

    python3 scripts/wsl_e6_winrate.py
"""

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from minesweeper.env import MinesweeperEnv  # noqa: E402
from minesweeper.eval import evaluate  # noqa: E402
from minesweeper.models import DQN, infer_width, pick_device  # noqa: E402
from wsl.quantize import quantize_layer  # noqa: E402

LAYERS = ["conv1.weight", "conv2.weight", "conv3.weight",
          "advantage_conv.weight", "value_fc1.weight", "value_fc2.weight"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zoo", default="checkpoints/zoo/zoo_seed*_best.pt")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--scheme", default="int4_tensor")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/wsl_e6_winrate.json")
    args = p.parse_args()

    device = pick_device(args.device)
    env = MinesweeperEnv(n_rows=8, n_cols=8, num_mines=10)
    paths = sorted(glob.glob(args.zoo))
    model = None
    results = {}
    for k, path in enumerate(paths):
        sd = torch.load(path, map_location="cpu")
        if model is None:
            model = DQN(width=infer_width(sd)).to(device)

        def win(s):
            model.load_state_dict(s)
            model.eval()
            _, w = evaluate(model, env, args.episodes, device, seed=args.seed)
            return w

        stem = Path(path).stem
        base = win(sd)
        entry = {"fp32_win": base, "layers": {}}
        for name in LAYERS:
            entry["layers"][name] = win(quantize_layer(sd, name, args.scheme))
        results[stem] = entry
        print(f"{k + 1}/{len(paths)} {stem}: fp32 {base:.3f}  " +
              " ".join(f"{n.split('.')[0]}:{entry['layers'][n]:.3f}"
                       for n in LAYERS), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
