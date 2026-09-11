# Minesweeper-RL

A Minesweeper agent studied through the symmetries of the game.

Two solvers over the same board: a fully convolutional dueling DQN that
reads the board as a two-plane image and outputs one Q value per cell, and a
deductive constraint solver (SAT/SMT) that uses symmetry reduction over
the dihedral group D4 via Burnside's lemma. The learned side carries the
same group structure as the deductive side: D4 test-time augmentation, a
hand-rolled p4m group-equivariant variant of the network, and the stuff in
`wsl/`, where I look at what independently trained agents have in common
once you line their neurons up.

Originally my BSc final project at the University of Leeds; the symmetry and
`wsl/` work came later.

## The network

```
spatial (B, 2, H, W)  covered mask, clue/8
        │  3×3 conv ×3 (32, 64, 64)
        ├── advantage: 1×1 conv → per-cell Q-map (B, H·W)   translation-equivariant
        └── value: global avg pool + [covered_ratio, mine_density] → MLP
Q = value + (advantage − mean)
```

No layer has a size-dependent weight shape, so one set of weights runs on
any board size. The equivariant variant (`minesweeper/equivariant.py`)
replaces the convolutions with D4 group convolutions (lifting + group conv,
group pooling before the heads), making the Q-map equivariant to rotations
and reflections of the board *by construction*. Numerical tests in
`tests/test_equivariance.py` verify this on all 8 group elements.

## Results

Shipped checkpoint (`best_model_batch5_*.pt`, trained on 8×8/10), evaluated
for 400 episodes per cell, greedy play with ε = 0.01. TTA means averaging
Q-maps over the board's symmetry orbit with the same weights and no
retraining: all 8 D4 elements on square boards, the 4 axis-preserving
elements on the non-square 16×30 board.

| board (zero-shot except 8×8) | win rate | win rate + D4 TTA | avg reward | avg reward + TTA |
|---|---|---|---|---|
| 8×8, 10 mines        | 50.5% | **70.5%** | 24.6 | **34.4** |
| 16×16, 40 mines      | 1.3%  | **16.0%** | 20.2 | **48.0** |
| 16×30, 99 mines      | 0.0%  | 0.0%      | 7.4  | **10.9** |

The 16×16 and 16×30 rows use the 8×8 weights unchanged, so the fully
convolutional design really does transfer across board sizes without
retraining. Respecting the game's symmetry group at inference adds 20 points
of win rate on the training size and takes the 16×16 zero-shot win rate from
1.3% to 16%, about 12×. On 16×30 neither variant wins games, though TTA
still raises the average reward from 7.4 to 10.9.

### Where to put the symmetry

Plain FCN vs the same weights with D4 TTA vs the p4m-equivariant network,
all trained for the same 3,000-episode budget on 8×8/10 (500 eval episodes
per cell; `scripts/run_ablation.py`):

| model (matched budget) | 8×8 win rate | 16×16 zero-shot win rate |
|---|---|---|
| plain FCN              | 15.0% | 0.0% |
| plain FCN + D4 TTA     | 15.4% | 0.0% |
| p4m-equivariant        | **34.0%** | 0.6% |

The two symmetry mechanisms do different jobs. Orbit-averaging at inference
helps a well-trained model (the +20 points above) but does almost nothing
for an undertrained one; symmetrising a weak Q function just averages its
noise. Building the symmetry into the architecture pays during training
instead: at the same budget the equivariant network more than doubles the
plain network's win rate, and it also beats a plain run trained 2.5 times
longer (20.0% at 7,500 episodes, same eval protocol).

## wsl/

This part treats the trained agents themselves as data. The architecture's
neuron-permutation symmetry group is a product of four symmetric groups
(over the three conv-channel axes and the value-MLP hidden axis);
`wsl/align.py` implements Git Re-Basin style weight matching over it,
and matching a permuted copy of a network recovers it exactly
(`tests/test_align.py`).

Linear interpolation between independently trained agents (6-agent seed
zoo; 4 pairs, each pairing the seed-1 agent with a different partner;
200 eval episodes per point):

<p align="center"><img src="results/wsl_barriers.png" width="700"></p>

Naive interpolation collapses at the midpoint (anchor agent 12.5% win,
partners 7.5% to 12%, naive midpoint mean 0.9%). Permutation alignment roughly
quadruples the midpoint mean (3.9%) and lies above the naive curve at every
interior lambda in the 4-pair average, though individual pairs vary. It does
not close the barrier: these independently trained agents are not
linearly mode-connected even after alignment, which matches what people
find for small networks trained from scratch.

<p align="center"><img src="results/wsl_spectra.png" width="900"></p>

The per-layer singular value spectra are consistent across seeds.
Two layers (conv3, value_fc1) carry a near-flat bulk: long runs of
nearly equal singular values. This is the regime where
truncation-based spectral methods are perturbation-sensitive, since the
subspace attached to a cluster of near-equal singular values is only
determined up to rotation within the cluster.

Reproduce: `scripts/train_zoo.py`, then `scripts/wsl_report.py`.

### From observation to measurement

That near-flat-spectrum observation grew into a study of when permutation
alignment can be trusted at all, now written up as a workshop paper (in
preparation). The zoos grew to 156 networks: 36 DQNs across three channel
widths plus 120 MNIST MLPs across four hidden widths, every population with
untrained controls, and the suite in `wsl/` measures alignment reliability
from the endpoint weights alone.

<p align="center"><img src="results/figure1.png" width="900"></p>

The short version of what came out:

- You can tell in advance which layers weight matching aligns reliably.
  A layer's spectral gap and the assignment problem's best-versus-second-best
  margin rank the four axes correctly in 95% of pairs (permutation test
  p < 1e-4). The value head sits at margins of 1e-13, the numerical tie
  floor, where the matching is provably arbitrary; the dueling head's
  argmax-invariance means that arbitrariness is behaviourally inert, and
  aligning only the convolutions reproduces full alignment exactly.
- Identifiability and mergeability move in opposite directions. The same
  pipeline closes the barrier on wide MNIST MLPs and leaves it open here at
  every width, and sweeping MLP width shows why: the barrier closes exactly
  as the matching margins collapse. Wide networks merge because their unit
  correspondences stop being unique; these RL agents keep well-determined
  correspondences whose best permutation isn't enough.
- Single-run weight matching fails silently more often than not. On
  instances with a known exact solution, one coordinate-descent run recovers
  the true permutation 40% of the time; two restarts recover all of them,
  and five capture everything on real pairs. Every alignment number here
  uses best-of-ten (`wsl.align.weight_matching_restarts`).

The dead-unit tie classes, the exact non-uniqueness characterisation, and
the full null battery (untrained zoos, noise-matched controls, activation
covariance) are in the paper's appendix; every mathematical claim is also a
unit test (`tests/test_proposition.py`).

## What's where

```
minesweeper/         headless package: board, env, models (width-parameterised), replay, train, eval, d4, equivariant
wsl/                 alignment (weight + activation matching, restarts), spectra, degeneracy and
                     instability diagnostics, barriers, REPAIR, the MNIST MLP zoo machinery
scripts/             train_zoo, train_mlp_zoo, run_ablation, wsl_report, wsl_nulls, wsl_e2, wsl_e3,
                     wsl_e3_analysis, wsl_barrier_link, wsl_partial, wsl_z4, wsl_figure1
tests/               D4 group axioms, equivariance, alignment recovery, width scaling, diagnostics,
                     barrier convention, REPAIR, the proposition's claims
Minesweeper_Solver.py   the original PyQt GUI app (play, watch the agent, deductive solvers)
Graph_Pattern_Generation2.py   pattern mining for the deductive solver
```

## Run

```bash
python3 -m pytest tests/ -q                      # group + equivariance + alignment tests
python3 -m minesweeper.train --rows 8 --cols 8 --mines 10 \
    --episodes-per-batch 1500 --num-batches 2 \
    --epsilon-decay-steps 15000 --learning-starts 500 --target-sync-steps 2000
python3 -m minesweeper.eval <checkpoint> --episodes 400 --sizes 8x8x10,16x16x40,16x30x99
python3 -m minesweeper.eval <checkpoint> --episodes 400 --sizes 8x8x10,16x16x40,16x30x99 --tta
python3 scripts/train_zoo.py --seeds 1 2 3 4 5 6 --episodes 2500
python3 scripts/wsl_report.py --zoo "checkpoints/zoo/*_best.pt" --pairs 4
python3 scripts/wsl_nulls.py                     # pipeline sanity: self-match + same-run matches
python3 scripts/wsl_e3.py                        # degeneracy + instability sweep over a zoo
python3 scripts/train_mlp_zoo.py --hidden 512    # the MNIST control zoo
python3 scripts/wsl_figure1.py                   # the paper figure, from committed results
python3 Minesweeper_Solver.py                    # the GUI
```

Training defaults mirror the original project; the flags above use a faster
exploration schedule sized for short runs. PyTorch with CUDA, Apple-Silicon
MPS, or CPU.

<p align="center"><img src="minesweeper_training_plot.png" width="600"></p>
