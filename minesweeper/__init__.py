"""Headless Minesweeper RL package.

Extracted from Minesweeper_Solver.py so the environment, models, and training
loop can be used without the Qt GUI. The GUI remains the entry point for
interactive play; everything here is importable and scriptable.
"""

from .board import (
    generate_random_board,
    compute_clues,
    create_board,
    build_adjacency,
    bfs_expand,
    build_state,
    DIFFICULTY_PARAMS,
)
from .env import MinesweeperEnv
from .models import DQN, pick_device, load_dqn
from .replay import ReplayBuffer, PrioritizedReplayBuffer
