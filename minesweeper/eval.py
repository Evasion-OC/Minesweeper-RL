"""Evaluation: greedy play with a small tie-breaking epsilon, optional D4 TTA,
and cross-size zero-shot evaluation of a single checkpoint.
"""

import argparse
import random

import numpy as np
import torch

from .env import MinesweeperEnv
from .models import pick_device, load_dqn
from . import d4


def evaluate(model, env, episodes, device, eval_epsilon=0.01, tta=False, seed=None):
    """Returns (avg_reward, win_rate). With tta=True, Q values are averaged
    over the D4 orbit of the board."""
    if episodes <= 0:
        return 0.0, 0.0
    rng_state = None
    if seed is not None:
        # reproducible boards without leaking reseeded global RNGs to the caller
        rng_state = (random.getstate(), np.random.get_state())
        random.seed(seed)
        np.random.seed(seed)
    was_training = model.training
    model.eval()
    total_reward, win_count = 0.0, 0
    max_steps = max(2 * env.n_rows * env.n_cols, 200)
    try:
        for _ in range(episodes):
            spatial, scalars = env.reset()
            done = False
            episode_reward = 0.0
            won = False
            for _ in range(max_steps):
                if done:
                    break
                valid = env.valid_actions()
                if not valid:
                    break
                if random.random() < eval_epsilon:
                    action = random.choice(valid)
                else:
                    sp = torch.from_numpy(spatial).unsqueeze(0).to(device)
                    sc = torch.from_numpy(scalars).unsqueeze(0).to(device)
                    with torch.no_grad():
                        q = d4.tta_q_values(model, sp, sc) if tta else model(sp, sc)
                    action = valid[q[0, valid].argmax().item()]
                next_state, reward, done, info = env.step(action)
                spatial, scalars = next_state
                episode_reward += reward
                if info.get('reason') == 'win':
                    won = True
            total_reward += episode_reward
            win_count += int(won)
    finally:
        if was_training:
            model.train()
        if rng_state is not None:
            random.setstate(rng_state[0])
            np.random.set_state(rng_state[1])
    return total_reward / episodes, win_count / episodes


def cross_size_eval(model, sizes, episodes, device, tta=False, seed=0):
    """Zero-shot: one checkpoint, evaluated on several board sizes.
    sizes: list of (rows, cols, mines). Returns list of result dicts."""
    results = []
    for rows, cols, mines in sizes:
        env = MinesweeperEnv(n_rows=rows, n_cols=cols, num_mines=mines)
        avg_r, win = evaluate(model, env, episodes, device, tta=tta, seed=seed)
        results.append({"rows": rows, "cols": cols, "mines": mines,
                        "avg_reward": avg_r, "win_rate": win, "tta": tta})
        print(f"{rows}x{cols}/{mines} mines  tta={tta}  "
              f"avg_reward={avg_r:.2f}  win_rate={win:.2%}", flush=True)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--sizes", default="8x8x10,16x16x40,16x30x99",
                   help="comma-separated rowsxcolsxmines")
    p.add_argument("--tta", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    device = pick_device(args.device)
    model = load_dqn(args.checkpoint, device)
    sizes = [tuple(int(v) for v in s.split("x")) for s in args.sizes.split(",")]
    cross_size_eval(model, sizes, args.episodes, device, tta=args.tta, seed=args.seed)


if __name__ == "__main__":
    main()
