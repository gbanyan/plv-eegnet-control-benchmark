"""Per-dataset Spearman association sensitivity analyses.

Outputs:
- figures/fig5_bootstrap_rho.png/.pdf  -- forest plot of per-dataset rho with 95% CIs
- results/subject_level_association_analysis.json

Run from project root:
    .venv/bin/python -m scripts.bootstrap_and_mixed
"""
from __future__ import annotations

import json
import itertools
from pathlib import Path

import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType, not Type 3
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Use the final three-condition data plan; superseded robustness families are
# intentionally not dependencies of this analysis.
from scripts.final_data import DATASETS, FD, load_dataset

def bootstrap_spearman(x, y, n_boot=2000, seed=0):
    """Return a percentile CI, redrawing rather than imputing undefined rhos."""
    rng = np.random.default_rng(seed)
    n = len(x)
    rhos = np.empty(n_boot)
    i = 0
    invalid = 0
    while i < n_boot:
        idx = rng.integers(0, n, n)
        xi, yi = x[idx], y[idx]
        if np.std(xi) == 0 or np.std(yi) == 0:
            invalid += 1
            continue
        rhos[i] = stats.spearmanr(xi, yi).statistic
        i += 1
    point = stats.spearmanr(x, y).statistic
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    return point, lo, hi, rhos, invalid


def permutation_spearman_p(x, y, *, seed=0, n_resamples=100_000):
    """Two-sided label-permutation p-value; enumerate exactly when n <= 9."""
    xr = stats.rankdata(x).astype(float)
    yr = stats.rankdata(y).astype(float)
    xr -= xr.mean()
    yr -= yr.mean()
    denom = np.linalg.norm(xr) * np.linalg.norm(yr)
    observed = float(np.dot(xr, yr) / denom)
    threshold = abs(observed) - 1e-15
    n = len(x)
    if n <= 9:
        extreme = 0
        total = 0
        for permutation in itertools.permutations(yr.tolist()):
            value = float(np.dot(xr, permutation) / denom)
            extreme += abs(value) >= threshold
            total += 1
        return extreme / total, "exact", total

    rng = np.random.default_rng(seed)
    extreme = 0
    remaining = n_resamples
    while remaining:
        size = min(10_000, remaining)
        permuted = np.vstack([rng.permutation(yr) for _ in range(size)])
        values = permuted @ xr / denom
        extreme += int(np.count_nonzero(np.abs(values) >= threshold))
        remaining -= size
    return (extreme + 1) / (n_resamples + 1), "monte_carlo", n_resamples


def holm_adjust(p_values):
    """Return Holm step-down adjusted p-values in original order."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted_sorted = np.maximum.accumulate(
        (len(p_values) - np.arange(len(p_values))) * p_values[order]
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def main():
    datasets = [load_dataset(name) for name in DATASETS.keys()]

    # ---- Per-dataset Spearman ρ with bootstrap 95% CIs --------------------
    print("=" * 78)
    print("Per-dataset Spearman ρ(baseline, Δ_PLV) with 95% bootstrap CIs (n_boot=2000)")
    print("=" * 78)
    rows = []
    association_rows = []
    rng_seed = 7
    for ds in datasets:
        x = ds["baseline"]
        y = ds["real"] - ds["zero"]
        bootstrap_seed = rng_seed
        rho, lo, hi, rhos, invalid = bootstrap_spearman(
            x, y, n_boot=2000, seed=bootstrap_seed
        )
        permutation_seed = 70_000 + rng_seed
        p_perm, p_method, p_resamples = permutation_spearman_p(
            x, y, seed=permutation_seed
        )
        rng_seed += 1
        rows.append((ds["name"], len(x), rho, lo, hi))
        print(f"  {ds['name']:14s}  n={len(x):3d}  ρ={rho:+.3f}  "
              f"95% CI [{lo:+.3f}, {hi:+.3f}]  p_perm={p_perm:.6g}")
        association_rows.append({
            "dataset": ds["name"],
            "n_subjects": len(x),
            "spearman_rho": float(rho),
            "bootstrap_percentile_95ci": [float(lo), float(hi)],
            "bootstrap_resamples": 2000,
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_invalid_resamples_redrawn": invalid,
            "permutation_p_two_sided": float(p_perm),
            "permutation_method": p_method,
            "permutation_resamples_or_enumerations": p_resamples,
            "permutation_seed": None if p_method == "exact" else permutation_seed,
        })

    adjusted = holm_adjust([
        row["permutation_p_two_sided"] for row in association_rows
    ])
    for row, p_adjusted in zip(association_rows, adjusted):
        row["permutation_p_holm_four_datasets"] = float(p_adjusted)

    association_output = Path("results/subject_level_association_analysis.json")
    association_output.write_text(json.dumps({
        "analysis": "exploratory per-dataset baseline-accuracy association",
        "intervals": "pointwise and unadjusted",
        "permutation_p_values": (
            "unadjusted values plus a Holm sensitivity across four datasets"
        ),
        "datasets": association_rows,
    }, indent=2) + "\n")
    print(f"Wrote association artifact: {association_output}")

    # ---- Forest plot ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ys = np.arange(len(rows))[::-1]   # top-to-bottom
    for y, (name, n, rho, lo, hi) in zip(ys, rows):
        ax.errorbar(rho, y, xerr=[[rho - lo], [hi - rho]],
                    fmt="o", color="#d05656", ms=8, mfc="white",
                    mec="#d05656", mew=1.7, capsize=4, elinewidth=1.4)
        ax.text(0.99, y, f"n={n}", transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=9, color="#666",
                family="monospace")
    ax.axvline(0, color="#999", lw=0.8, ls="--")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel(r"Spearman $\rho$  (baseline acc.  vs.  $\Delta_{PLV}$ = real − zero)")
    ax.set_xlim(-1, 1)
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_title("Per-dataset Spearman ρ with 95% bootstrap CIs",
                 fontsize=11)
    fig.tight_layout()
    out = FD / "fig5_bootstrap_rho"
    fig.savefig(f"{out}.png", dpi=180, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote forest plot: {out}.png / .pdf")

if __name__ == "__main__":
    main()
