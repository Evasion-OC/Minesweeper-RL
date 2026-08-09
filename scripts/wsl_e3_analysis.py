"""Cross-population analysis of E3 sweep JSONs: does degeneracy predict
instability once populations with different degeneracy ranges are pooled?

Each input JSON is one population (a matched set of checkpoints swept
pairwise by scripts/wsl_e3.py). This script builds long-form rows
(population, pair, axis, degeneracy statistics, instability probes) and
prints four tables:

  1. population x axis medians: how training moves both quantities;
  2. within-axis Spearman pooled across populations: the claim's test,
     with degeneracy given real range;
  3. within-axis Spearman inside each population: the restricted-range grid;
  4. the fully pooled correlation, for continuity with the driver preview.

Optionally writes the rows as CSV, which is the one figure's input.

    python3 scripts/wsl_e3_analysis.py results/wsl_e3_zoo16.json \
        results/wsl_e3_frac0*.json --csv results/wsl_e3_rows.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from scipy import stats as sps  # noqa: E402

from wsl.degeneracy import AXIS_LAYER  # noqa: E402

STATS = ("entropy", "min_rel_gap")
PROBES = ("churn", "lap_rel_gap", "disagree")


def load_rows(paths):
    rows = []
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        pop = Path(path).stem.replace("wsl_e3_", "")
        deg = data["degeneracy"]
        for entry in data["pairs"]:
            for ax, layer in AXIS_LAYER.items():
                da, db = deg[entry["a"]][layer], deg[entry["b"]][layer]
                rows.append({
                    "population": pop, "pair": entry["pair"], "axis": ax,
                    "entropy_a": da["entropy"], "entropy_b": db["entropy"],
                    "min_rel_gap_a": da["min_rel_gap"], "min_rel_gap_b": db["min_rel_gap"],
                    "entropy": 0.5 * (da["entropy"] + db["entropy"]),
                    "min_rel_gap": 0.5 * (da["min_rel_gap"] + db["min_rel_gap"]),
                    "churn": entry["perturbation"][ax]["churn_mean"],
                    "lap_rel_gap": entry["assignment_gap"][ax]["rel_gap"],
                    "disagree": entry["restart"][ax]["disagree_frac"],
                })
    return rows


def spearman_cells(rows, stat):
    cells = []
    for probe in PROBES:
        x = np.array([r[stat] for r in rows])
        y = np.array([r[probe] for r in rows])
        rho, p = sps.spearmanr(x, y)
        cells.append(f"{rho:+.3f} (p={p:.4f})")
    return cells


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsons", nargs="+")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    rows = load_rows(args.jsons)
    pops = sorted({r["population"] for r in rows})
    axes = list(AXIS_LAYER)

    print(f"populations: {', '.join(pops)}  ({len(rows)} rows)\n")

    print("1. population x axis medians (entropy | min_rel_gap || churn | lap_rel_gap | disagree)")
    for pop in pops:
        for ax in axes:
            sub = [r for r in rows if r["population"] == pop and r["axis"] == ax]
            med = {k: float(np.median([r[k] for r in sub])) for k in STATS + PROBES}
            print(f"  {pop:10s} {ax}: {med['entropy']:.3f} | {med['min_rel_gap']:.5f} || "
                  f"{med['churn']:.3f} | {med['lap_rel_gap']:.2e} | {med['disagree']:.3f}")

    print("\n2. within-axis Spearman, pooled across populations "
          f"(n per axis = {len(rows) // len(axes)})")
    header = f"  {'axis':4s} {'stat':12s} " + " ".join(f"{p:>18s}" for p in PROBES)
    print(header)
    for ax in axes:
        sub = [r for r in rows if r["axis"] == ax]
        for stat in STATS:
            cells = spearman_cells(sub, stat)
            print(f"  {ax:4s} {stat:12s} " + " ".join(f"{c:>18s}" for c in cells))

    print("\n3. within-axis Spearman inside each population (min_rel_gap only, rho)")
    print(f"  {'population':10s} " + " ".join(f"{ax:>8s}" for ax in axes))
    for pop in pops:
        cells = []
        for ax in axes:
            sub = [r for r in rows if r["population"] == pop and r["axis"] == ax]
            rho, _ = sps.spearmanr([r["min_rel_gap"] for r in sub],
                                   [r["churn"] for r in sub])
            cells.append(f"{rho:+.2f}")
        print(f"  {pop:10s} " + " ".join(f"{c:>8s}" for c in cells))

    print(f"\n4. fully pooled ({len(rows)} rows)")
    for stat in STATS:
        cells = spearman_cells(rows, stat)
        print(f"  {stat:12s} " + " ".join(f"{c:>18s}" for c in cells))

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
