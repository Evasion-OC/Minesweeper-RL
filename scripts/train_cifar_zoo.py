"""Train the CIFAR-10 width-arm zoos: VGG-style nets with BatchNorm at a
given width multiplier, many seeds, saving the initialisation per seed.

    python3 scripts/train_cifar_zoo.py --mult 1.0 --seeds $(seq 1 8)
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
from wsl.cifarzoo import CifarVGG  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 9)))
    p.add_argument("--mult", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = pick_device(args.device)
    tag = str(args.mult).replace(".", "p")
    out_dir = Path(args.out_dir or f"checkpoints/cifar_m{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    norm = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2470, 0.2435, 0.2616))
    train_tf = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(), norm])
    test_tf = transforms.Compose([transforms.ToTensor(), norm])
    train_ds = datasets.CIFAR10(args.data_dir, train=True, download=True,
                                transform=train_tf)
    test_ds = datasets.CIFAR10(args.data_dir, train=False, download=True,
                               transform=test_tf)

    for seed in args.seeds:
        t0 = time.time()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = CifarVGG(mult=args.mult).to(device)
        name = f"cifar_m{tag}_seed{seed}"
        torch.save(model.state_dict(), out_dir / f"{name}_init.pt")
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                              weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=args.batch, shuffle=True, num_workers=2,
            generator=torch.Generator().manual_seed(seed))
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1024,
                                                  num_workers=2)
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
            sched.step()
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for x, y in test_loader:
                    pred = model(x.to(device)).argmax(1).cpu()
                    correct += (pred == y).sum().item()
                    total += y.numel()
            rows.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                         "test_acc": correct / total})
            if epoch % 6 == 0:
                print(f"[{name}] epoch {epoch}/{args.epochs} "
                      f"loss {rows[-1]['train_loss']:.3f} "
                      f"acc {rows[-1]['test_acc']:.4f}", flush=True)
        torch.save(model.state_dict(), out_dir / f"{name}_final.pt")
        with open(out_dir / f"{name}_log.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[{name}] done in {time.time() - t0:.0f}s: "
              f"test acc {rows[-1]['test_acc']:.4f}", flush=True)


if __name__ == "__main__":
    main()
