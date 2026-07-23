"""Reveal-only Minesweeper environment with guaranteed-safe first click.

Same dynamics and reward as the GUI monolith's MinesweeperEnv, but headless
and constructible from explicit (n_rows, n_cols, num_mines) so the same code
runs any board size.
"""

from .board import (
    DIFFICULTY_PARAMS,
    generate_random_board,
    compute_clues,
    create_board,
    build_adjacency,
    bfs_expand,
    build_state,
)


class MinesweeperEnv:
    """Action space is n_rows * n_cols reveal indices; no flag action (a flag
    action creates a degenerate "flag everything" attractor for risk-averse
    Q functions).

    Reward (bounded):
      step penalty           -0.05
      reveal safe (per cell)  +1.0
      cascade per BFS cell    +0.2
      hit a mine (terminal)  -10.0
      win (terminal)         +20.0
    """

    def __init__(self, difficulty=None, n_rows=None, n_cols=None, num_mines=None):
        if difficulty is not None:
            # unlike the monolith, an unknown preset raises instead of silently
            # defaulting to beginner
            self.n_rows, self.n_cols, self.num_mines = DIFFICULTY_PARAMS[difficulty]
        else:
            if None in (n_rows, n_cols, num_mines):
                raise ValueError("Give either a difficulty preset or all of n_rows, n_cols, num_mines.")
            self.n_rows, self.n_cols, self.num_mines = n_rows, n_cols, num_mines
        self.first_reveal_done = False
        self.reset()

    def reset(self):
        grid = generate_random_board(self.n_rows, self.n_cols, self.num_mines)
        compute_clues(grid)
        self.board = create_board(grid)
        self.adj = build_adjacency(self.board)
        self.done = False
        self.step_count = 0
        self.first_reveal_done = False
        return self._get_state()

    def _get_state(self):
        return build_state(self.board, self.n_rows, self.n_cols, self.num_mines)

    def _ensure_safe_first_reveal(self, r, c):
        grid = generate_random_board(self.n_rows, self.n_cols, self.num_mines, first_move=(r, c))
        compute_clues(grid)
        self.board = create_board(grid)
        self.adj = build_adjacency(self.board)
        self.first_reveal_done = True

    def valid_actions(self):
        acts = []
        for i in range(self.n_rows * self.n_cols):
            r, c = divmod(i, self.n_cols)
            if self.board[r][c]['covered']:
                acts.append(i)
        return acts

    def step(self, action):
        r, c = divmod(action, self.n_cols)
        cell = self.board[r][c]

        if not cell['covered']:
            self.step_count += 1
            return self._get_state(), -0.5, self.done, {'reason': 'invalid_move'}

        if not self.first_reveal_done:
            self._ensure_safe_first_reveal(r, c)
            cell = self.board[r][c]

        self.step_count += 1
        info = {}
        reward = -0.05

        cell['covered'] = False
        if cell['isMine']:
            self.done = True
            reward += -10.0
            info['reason'] = 'hit_mine'
        else:
            reward += 1.0
            if cell['clue'] == 0:
                cascade = bfs_expand(self.board, r, c, self.adj)
                reward += 0.2 * cascade

        if self.check_win():
            self.done = True
            reward += 20.0
            info['reason'] = 'win'

        return self._get_state(), reward, self.done, info

    def check_win(self):
        return all(cell['covered'] == cell['isMine'] for row in self.board for cell in row)
