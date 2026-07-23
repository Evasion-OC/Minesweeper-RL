"""Board mechanics: generation, clues, adjacency, flood fill, RL state encoding.

Pure functions over a board of dicts {'isMine', 'clue', 'covered', 'flagged'},
identical semantics to the GUI monolith so checkpoints transfer.
"""

import random
from collections import deque

import numpy as np

DIFFICULTY_PARAMS = {
    'beginner': (8, 8, 10),
    'intermediate': (16, 16, 40),
    'expert': (16, 30, 99),
    'extreme': (25, 50, 375),
}

_DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def generate_random_board(n_rows, n_cols, num_mines, first_move=None):
    cells = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    if first_move:
        exset = {first_move}
        fr, fc = first_move
        for dr, dc in _DIRS:
            rr, cc = fr + dr, fc + dc
            if 0 <= rr < n_rows and 0 <= cc < n_cols:
                exset.add((rr, cc))
        available = list(set(cells) - exset)
    else:
        available = cells[:]
    if num_mines > len(available):
        raise ValueError("Too many mines for the available cells.")
    mines = set(random.sample(available, num_mines))
    return [['*' if (r, c) in mines else '.' for c in range(n_cols)]
            for r in range(n_rows)]


def compute_clues(grid):
    n_rows, n_cols = len(grid), len(grid[0])
    for r in range(n_rows):
        for c in range(n_cols):
            if grid[r][c] == '*':
                continue
            count = 0
            for dr, dc in _DIRS:
                rr, cc = r + dr, c + dc
                if 0 <= rr < n_rows and 0 <= cc < n_cols and grid[rr][cc] == '*':
                    count += 1
            grid[r][c] = str(count)


def create_board(grid):
    board = []
    for row in grid:
        rowdata = []
        for ch in row:
            if ch == '*':
                rowdata.append({'isMine': True, 'clue': -1, 'covered': True, 'flagged': False})
            else:
                rowdata.append({'isMine': False, 'clue': int(ch), 'covered': True, 'flagged': False})
        board.append(rowdata)
    return board


def build_adjacency(board):
    n_rows, n_cols = len(board), len(board[0])
    adj = []
    for r in range(n_rows):
        row = []
        for c in range(n_cols):
            neighbors = []
            for dr, dc in _DIRS:
                rr, cc = r + dr, c + dc
                if 0 <= rr < n_rows and 0 <= cc < n_cols:
                    neighbors.append((rr, cc))
            row.append(neighbors)
        adj.append(row)
    return adj


def bfs_expand(board, r, c, adj):
    """Flood-fill from a 0-clue cell. Caller has already revealed (r, c)."""
    if board[r][c]['clue'] != 0:
        return 0
    revealed_count = 0
    queue = deque([(r, c)])
    visited = set()
    while queue:
        curr_r, curr_c = queue.popleft()
        if (curr_r, curr_c) in visited:
            continue
        visited.add((curr_r, curr_c))
        cell = board[curr_r][curr_c]
        if cell['covered'] and not cell['flagged']:
            cell['covered'] = False
            revealed_count += 1
        if cell['clue'] == 0 and not cell['flagged']:
            for nr, nc in adj[curr_r][curr_c]:
                if (nr, nc) not in visited:
                    queue.append((nr, nc))
    return revealed_count


def build_state(board, n_rows, n_cols, num_mines):
    """Canonical RL state, shared by training and GUI inference.

    Returns (spatial, scalars):
      spatial[0]: covered mask  (1 if covered, 0 otherwise)
      spatial[1]: clue / 8      (0..1 if revealed, 0 if covered)
      scalars: [covered_ratio, mine_density]
    """
    spatial = np.zeros((2, n_rows, n_cols), dtype=np.float32)
    covered_count = 0
    for r in range(n_rows):
        for c in range(n_cols):
            cell = board[r][c]
            if cell['covered']:
                spatial[0, r, c] = 1.0
                covered_count += 1
            else:
                spatial[1, r, c] = cell['clue'] / 8.0
    cells = max(n_rows * n_cols, 1)
    scalars = np.array([covered_count / cells, num_mines / cells], dtype=np.float32)
    return spatial, scalars
