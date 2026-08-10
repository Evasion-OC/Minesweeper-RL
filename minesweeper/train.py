"""Headless Double+Dueling DQN training, ported from the monolith.

Same algorithm and default hyperparameters as train_in_batches in
Minesweeper_Solver.py: prioritized replay, Huber loss, env-step-paced
epsilon/beta/target sync, per-batch eval with early stopping. Additions:
explicit board size, seeding, model variant selection, checkpoint dir,
CSV metrics log.
"""

import argparse
import csv
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .env import MinesweeperEnv
from .models import DQN, pick_device
from .equivariant import EquivariantDQN
from .replay import PrioritizedReplayBuffer
from .eval import evaluate


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_model(variant, width=1):
    if variant == "plain":
        return DQN(width=width)
    if variant == "equivariant":
        if width != 1:
            raise ValueError("width scaling is implemented for the plain variant only")
        return EquivariantDQN()
    raise ValueError(f"unknown variant {variant!r}")


def train(n_rows=8, n_cols=8, num_mines=10,
          variant="plain",
          width=1,
          seed=0,
          episodes_per_batch=1000,
          num_batches=2,
          eval_episodes=200,
          batch_size=512,
          gamma=0.99,
          learning_rate=1e-4,
          epsilon_start=1.0,
          epsilon_end=0.1,
          epsilon_decay_steps=80_000,
          learning_starts=1_000,
          target_sync_steps=10_000,
          beta_start=0.4,
          beta_end=0.5,
          replay_capacity=100_000,
          early_stop_patience=3,
          out_dir="checkpoints",
          run_name=None,
          device=None,
          log_every=100,
          milestone_fracs=()):
    if num_batches < 1:
        raise ValueError(f"num_batches must be >= 1, got {num_batches}")
    device = pick_device(device)
    set_seed(seed)  # the original monolith runs were unseeded; package runs are reproducible
    os.makedirs(out_dir, exist_ok=True)
    run_name = run_name or f"{variant}_r{n_rows}x{n_cols}m{num_mines}_seed{seed}"
    csv_path = os.path.join(out_dir, f"{run_name}_log.csv")

    env = MinesweeperEnv(n_rows=n_rows, n_cols=n_cols, num_mines=num_mines)
    policy_net = make_model(variant, width=width).to(device)
    target_net = make_model(variant, width=width).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate, weight_decay=1e-5)
    replay_buffer = PrioritizedReplayBuffer(capacity=replay_capacity, alpha=0.6)

    best_eval_win_rate = -1.0
    best_path = None
    bad_batches = 0
    env_steps = 0
    epsilon = epsilon_start
    max_steps = max(2 * n_rows * n_cols, 200)

    log_rows = []

    # checkpoints at fixed training fractions (0 = initialisation): pure
    # saves with no RNG draws, so the training stream is identical with or
    # without them. Under early stopping, later milestones are never reached.
    total_episodes = episodes_per_batch * num_batches
    milestones = {}
    for frac in sorted(milestone_fracs):
        milestones.setdefault(int(round(frac * total_episodes)), frac)

    def save_milestone(frac):
        path = os.path.join(out_dir, f"{run_name}_frac{int(round(frac * 100)):03d}.pt")
        torch.save(policy_net.state_dict(), path)

    if 0 in milestones:
        save_milestone(milestones.pop(0))

    for batch_idx in range(1, num_batches + 1):
        batch_rewards, batch_wins = [], []
        for episode in range(episodes_per_batch):
            spatial, scalars = env.reset()
            done = False
            total_reward = 0.0
            won = False

            for _ in range(max_steps):
                if done:
                    break
                env_steps += 1
                progress = min(1.0, env_steps / max(1, epsilon_decay_steps))
                epsilon = epsilon_start + (epsilon_end - epsilon_start) * progress

                valid = env.valid_actions()
                if not valid:
                    break

                if env_steps < learning_starts or random.random() < epsilon:
                    action = random.choice(valid)
                else:
                    sp = torch.from_numpy(spatial).unsqueeze(0).to(device)
                    sc = torch.from_numpy(scalars).unsqueeze(0).to(device)
                    with torch.no_grad():
                        q = policy_net(sp, sc)
                    action = valid[q[0, valid].argmax().item()]

                next_state, reward, done, info = env.step(action)
                next_spatial, next_scalars = next_state
                total_reward += reward
                if info.get('reason') == 'win':
                    won = True
                replay_buffer.push((spatial, scalars), action, reward,
                                   (next_spatial, next_scalars), done)
                spatial, scalars = next_spatial, next_scalars

                if env_steps >= learning_starts and len(replay_buffer) >= batch_size:
                    beta = beta_start + (beta_end - beta_start) * progress
                    samples, indices, weights = replay_buffer.sample(batch_size, beta=beta)
                    states_b, actions_b, rewards_b, next_states_b, dones_b = zip(*samples)
                    sp_t = torch.from_numpy(np.stack([s[0] for s in states_b])).to(device)
                    sc_t = torch.from_numpy(np.stack([s[1] for s in states_b])).to(device)
                    nsp_t = torch.from_numpy(np.stack([s[0] for s in next_states_b])).to(device)
                    nsc_t = torch.from_numpy(np.stack([s[1] for s in next_states_b])).to(device)
                    actions_t = torch.tensor(actions_b, dtype=torch.long, device=device)
                    rewards_t = torch.tensor(rewards_b, dtype=torch.float32, device=device)
                    dones_t = torch.tensor(dones_b, dtype=torch.float32, device=device)
                    weights_t = torch.tensor(np.asarray(weights), dtype=torch.float32, device=device)

                    q_pred = policy_net(sp_t, sc_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        next_actions = policy_net(nsp_t, nsc_t).argmax(1, keepdim=True)
                        next_q = target_net(nsp_t, nsc_t).gather(1, next_actions).squeeze(1)
                        td_target = rewards_t + gamma * next_q * (1.0 - dones_t)
                    td_errors = q_pred - td_target
                    huber = F.smooth_l1_loss(q_pred, td_target, reduction='none')
                    loss = (weights_t * huber).mean()

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)
                    optimizer.step()

                    replay_buffer.update_priorities(
                        indices, td_errors.abs().detach().cpu().numpy() + 1e-6)

                if env_steps % target_sync_steps == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            batch_rewards.append(total_reward)
            batch_wins.append(1 if won else 0)
            if (episode + 1) % log_every == 0:
                print(f"[{run_name}] batch {batch_idx} ep {episode + 1}/{episodes_per_batch} "
                      f"avg_r={np.mean(batch_rewards[-log_every:]):.2f} "
                      f"win={np.mean(batch_wins[-log_every:]):.2%} "
                      f"eps={epsilon:.3f} steps={env_steps}", flush=True)

            done_episodes = (batch_idx - 1) * episodes_per_batch + episode + 1
            if done_episodes in milestones:
                save_milestone(milestones.pop(done_episodes))

        avg_reward, win_rate = evaluate(policy_net, env, eval_episodes, device)
        log_rows.append({"batch": batch_idx, "train_avg_reward": float(np.mean(batch_rewards)),
                         "train_win_rate": float(np.mean(batch_wins)),
                         "eval_avg_reward": avg_reward, "eval_win_rate": win_rate,
                         "env_steps": env_steps, "epsilon": epsilon})
        print(f"[{run_name}] batch {batch_idx} EVAL avg_r={avg_reward:.2f} win={win_rate:.2%}", flush=True)

        ckpt = os.path.join(out_dir, f"{run_name}_batch{batch_idx}.pt")
        torch.save(policy_net.state_dict(), ckpt)
        if win_rate > best_eval_win_rate:
            best_eval_win_rate = win_rate
            best_path = os.path.join(out_dir, f"{run_name}_best.pt")
            torch.save(policy_net.state_dict(), best_path)
            bad_batches = 0
        else:
            bad_batches += 1
        if bad_batches >= early_stop_patience:
            print(f"[{run_name}] early stop at batch {batch_idx}", flush=True)
            if best_path is not None:  # parity with the monolith: leave the live net at its best
                policy_net.load_state_dict(torch.load(best_path, map_location=device))
            break

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"[{run_name}] done. best eval win rate {best_eval_win_rate:.2%} -> {best_path}", flush=True)
    return best_path, best_eval_win_rate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, default=8)
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--mines", type=int, default=10)
    p.add_argument("--variant", choices=["plain", "equivariant"], default="plain")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes-per-batch", type=int, default=1000)
    p.add_argument("--num-batches", type=int, default=2)
    p.add_argument("--eval-episodes", type=int, default=200)
    p.add_argument("--epsilon-decay-steps", type=int, default=80_000)
    p.add_argument("--learning-starts", type=int, default=1_000)
    p.add_argument("--target-sync-steps", type=int, default=10_000)
    p.add_argument("--out-dir", default="checkpoints")
    p.add_argument("--run-name", default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    train(n_rows=args.rows, n_cols=args.cols, num_mines=args.mines,
          variant=args.variant, seed=args.seed,
          episodes_per_batch=args.episodes_per_batch,
          num_batches=args.num_batches, eval_episodes=args.eval_episodes,
          epsilon_decay_steps=args.epsilon_decay_steps,
          learning_starts=args.learning_starts,
          target_sync_steps=args.target_sync_steps,
          out_dir=args.out_dir, run_name=args.run_name, device=args.device)


if __name__ == "__main__":
    main()
