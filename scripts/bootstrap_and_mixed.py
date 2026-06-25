"""Bootstrap CIs for per-dataset Spearman rho + mixed-effects regression.

Outputs:
- figures/fig5_bootstrap_rho.png/.pdf  -- forest plot of per-dataset rho with 95% CIs
- (stdout) mixed-effects fit table for Δ_PLV ~ baseline + (baseline|dataset)

Run from project root:
    .venv/bin/python -m scripts.bootstrap_and_mixed
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Re-use the dataset spec from build_figures.
from scripts.build_figures import DATASETS, RD, FD, load_dataset, _per_subject

import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd


def bootstrap_spearman(x, y, n_boot=2000, seed=0) -> tuple[float, float, float]:
    """Returns (point, lo, hi) percentile-CI for Spearman rho."""
    rng = np.random.default_rng(seed)
    n = len(x)
    rhos = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        # Skip pathological resamples where one variable has zero variance.
        xi, yi = x[idx], y[idx]
        if np.std(xi) == 0 or np.std(yi) == 0:
            rhos[i] = 0.0
            continue
        rhos[i] = stats.spearmanr(xi, yi).statistic
    point = stats.spearmanr(x, y).statistic
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    return point, lo, hi, rhos


def main():
    datasets = [load_dataset(name) for name in DATASETS.keys()]

    # ---- Per-dataset Spearman ρ with bootstrap 95% CIs --------------------
    print("=" * 78)
    print("Per-dataset Spearman ρ(baseline, Δ_PLV) with 95% bootstrap CIs (n_boot=2000)")
    print("=" * 78)
    rows = []
    rng_seed = 7
    for ds in datasets:
        x = ds["baseline"]
        y = ds["real"] - ds["zero"]
        rho, lo, hi, rhos = bootstrap_spearman(x, y, n_boot=2000, seed=rng_seed)
        rng_seed += 1
        rows.append((ds["name"], len(x), rho, lo, hi))
        print(f"  {ds['name']:14s}  n={len(x):3d}  ρ={rho:+.3f}  "
              f"95% CI [{lo:+.3f}, {hi:+.3f}]")

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

    # ---- Mixed-effects regression -----------------------------------------
    rec = []
    for ds in datasets:
        for sid, b, d in zip(ds["sids"], ds["baseline"], ds["real"] - ds["zero"]):
            rec.append({"dataset": ds["name"], "subject": int(sid),
                        "baseline": float(b), "delta": float(d)})
    df = pd.DataFrame(rec)
    print("\n" + "=" * 78)
    print(f"Pooled DataFrame: n_obs={len(df)}, datasets={df['dataset'].nunique()}")
    print(df.groupby("dataset")[["baseline", "delta"]]
            .agg(["mean", "std", "count"]).round(3))

    # Model 1: random intercept by dataset
    print("\n" + "-" * 78)
    print("Model 1: Δ_PLV ~ baseline + (1|dataset)   (random intercept)")
    m1 = smf.mixedlm("delta ~ baseline", data=df, groups="dataset").fit(reml=True)
    print(m1.summary())

    # Model 2: random intercept + random slope by dataset
    print("\n" + "-" * 78)
    print("Model 2: Δ_PLV ~ baseline + (baseline|dataset)   (random intercept + slope)")
    try:
        m2 = smf.mixedlm("delta ~ baseline", data=df, groups="dataset",
                         re_formula="~baseline").fit(reml=True, method="lbfgs")
        print(m2.summary())
    except Exception as e:
        print(f"  (random-slope model failed to converge: {e})")
        m2 = None

    # Per-dataset OLS slopes for comparison
    print("\n" + "-" * 78)
    print("Per-dataset OLS slopes (baseline -> Δ_PLV)")
    for ds_name, sub in df.groupby("dataset"):
        if len(sub) < 3:
            print(f"  {ds_name:14s}  (skipped, n<3)")
            continue
        x = sm.add_constant(sub["baseline"])
        ols = sm.OLS(sub["delta"], x).fit()
        slope = ols.params["baseline"]
        se = ols.bse["baseline"]
        ci = ols.conf_int().loc["baseline"].values
        p = ols.pvalues["baseline"]
        print(f"  {ds_name:14s}  n={len(sub):3d}  slope={slope:+.3f}  "
              f"SE={se:.3f}  95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]  p={p:.4f}")


if __name__ == "__main__":
    main()
