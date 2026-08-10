"""Train a seed zoo of plain DQNs for the weight-space analysis.

Each agent: same board, same hyperparameters, different seed. Modest budget -
the zoo needs competent, independently trained agents, not record-setters.

    python3 scripts/train_zoo.py --seeds 1 2 3 4 5 6 7 8 --episodes 800
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minesweeper.train import train  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8])
    p.add_argument("--rows", type=int, default=8)
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--mines", type=int, default=10)
    p.add_argument("--episodes", type=int, default=800)
    p.add_argument("--eval-episodes", type=int, default=200)
    p.add_argument("--epsilon-decay-steps", type=int, default=15_000)
    p.add_argument("--learning-starts", type=int, default=500)
    p.add_argument("--target-sync-steps", type=int, default=2_000)
    p.add_argument("--milestones", default="0,0.25,0.5,0.75",
                   help="training fractions to checkpoint (empty to disable)")
    p.add_argument("--width", type=int, default=1,
                   help="channel multiplier, for the width sweep")
    p.add_argument("--prefix", default="zoo", help="run-name prefix")
    p.add_argument("--out-dir", default="checkpoints/zoo")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    fracs = tuple(float(v) for v in args.milestones.split(",") if v != "")

    for seed in args.seeds:
        train(n_rows=args.rows, n_cols=args.cols, num_mines=args.mines,
              variant="plain", width=args.width, seed=seed,
              episodes_per_batch=args.episodes, num_batches=1,
              eval_episodes=args.eval_episodes,
              epsilon_decay_steps=args.epsilon_decay_steps,
              learning_starts=args.learning_starts,
              target_sync_steps=args.target_sync_steps,
              out_dir=args.out_dir, run_name=f"{args.prefix}_seed{seed}",
              device=args.device, milestone_fracs=fracs)


if __name__ == "__main__":
    main()
