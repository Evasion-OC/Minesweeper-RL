"""Train the Z4 supervised control zoo: the Ainsworth MLP recipe on MNIST
(three hidden layers of 512, Adam 1e-3), many seeds. Saves the
initialisation (the untrained control) and the final network per seed.

    python3 scripts/train_mlp_zoo.py --seeds $(seq 1 30)
"""

import argparse
import csv
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torchvision import datasets, transforms  # noqa: E402

from minesweeper.models import pick_device  # noqa: E402
from wsl.mlpzoo import MLP  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 31)))
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="checkpoints/z4")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = pick_device(args.device)
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST(args.data_dir, train=True, download=True, transform=tf)
    test_ds = datasets.MNIST(args.data_dir, train=False, download=True, transform=tf)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        t0 = time.time()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = MLP(hidden=args.hidden).to(device)
        torch.save(model.state_dict(), out_dir / f"z4_seed{seed}_init.pt")
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=args.batch, shuffle=True,
            generator=torch.Generator().manual_seed(seed))
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=2048)

        rows = []
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = F.cross_entropy(model(x), y)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for x, y in test_loader:
                    pred = model(x.to(device)).argmax(1).cpu()
                    correct += (pred == y).sum().item()
                    total += y.numel()
            rows.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                         "test_acc": correct / total})
        torch.save(model.state_dict(), out_dir / f"z4_seed{seed}_final.pt")
        with open(out_dir / f"z4_seed{seed}_log.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[z4_seed{seed}] done in {time.time() - t0:.0f}s: "
              f"test acc {rows[-1]['test_acc']:.4f}", flush=True)


if __name__ == "__main__":
    main()
