"""Build the cross-dataset figures from the per-subject result JSONs.

Outputs PNG + PDF per figure into figures/. Run from project root:
    .venv/bin/python -m scripts.build_figures
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

PROJECT = Path(__file__).resolve().parents[1]
RD = PROJECT / "results"
FD = PROJECT / "figures"
FD.mkdir(parents=True, exist_ok=True)

# Per-dataset metadata + JSON-file plan. None for a seed means "this dataset
# has no per-seed file naming for this seed (legacy)".
# All datasets now use the proper-split rerun (3 seeds × 4 conditions
# per dataset = 12 JSONs each, 48 total). The `propersplit` suffix
# distinguishes the new files from the legacy single-set runs.
DATASETS = {
    "BCI 2a": {
        "n_channels": 22, "n_classes": 4, "n_pairs": 231,
        "baseline_file_seeds": {s: f"bci2a_eegnet_seed{s}_300ep_propersplit.json" for s in (42, 43, 44)},
        "seeds": {
            "real":    {s: f"bci2a_plv_eegnet_seed{s}_300ep_propersplit.json" for s in (42, 43, 44)},
            "zero":    {s: f"bci2a_plv_eegnet_zero_seed{s}_300ep_propersplit.json" for s in (42, 43, 44)},
            "shuffle": {s: f"bci2a_plv_eegnet_shuffle_pairs_seed{s}_300ep_propersplit.json" for s in (42, 43, 44)},
        },
    },
    "BCI 2b": {
        "n_channels": 3, "n_classes": 2, "n_pairs": 3,
        "baseline_file_seeds": {s: f"bci2b_eegnet_seed{s}_300ep_propersplit.json" for s in (42, 43, 44)},
        "seeds": {
            "real":    {s: f"bci2b_plv_eegnet_seed{s}_300ep_propersplit.json" for s in (42, 43, 44)},
            "zero":    {s: f"bci2b_plv_eegnet_zero_seed{s}_300ep_propersplit.json" for s in (42, 43, 44)},
            "shuffle": {s: f"bci2b_plv_eegnet_shuffle_pairs_seed{s}_300ep_propersplit.json" for s in (42, 43, 44)},
        },
    },
    "Lee2019": {
        "n_channels": 62, "n_classes": 2, "n_pairs": 1891,
        "baseline_file_seeds": {s: f"lee2019_eegnet_seed{s}_200ep_propersplit.json" for s in (42, 43, 44)},
        "seeds": {
            "real":    {s: f"lee2019_plv_eegnet_seed{s}_200ep_propersplit.json" for s in (42, 43, 44)},
            "zero":    {s: f"lee2019_plv_eegnet_zero_seed{s}_200ep_propersplit.json" for s in (42, 43, 44)},
            "shuffle": {s: f"lee2019_plv_eegnet_shuffle_pairs_seed{s}_200ep_propersplit.json" for s in (42, 43, 44)},
        },
    },
    "Schirrmeister": {
        "n_channels": 128, "n_classes": 4, "n_pairs": 8128,
        "baseline_file_seeds": {s: f"schirrmeister_eegnet_seed{s}_200ep_propersplit.json" for s in (42, 43, 44)},
        "seeds": {
            "real":    {s: f"schirrmeister_plv_eegnet_seed{s}_200ep_propersplit.json" for s in (42, 43, 44)},
            "zero":    {s: f"schirrmeister_plv_eegnet_zero_seed{s}_200ep_propersplit.json" for s in (42, 43, 44)},
            "shuffle": {s: f"schirrmeister_plv_eegnet_shuffle_pairs_seed{s}_200ep_propersplit.json" for s in (42, 43, 44)},
        },
    },
}

PALETTE = {
    "baseline": "#444",
    "zero":     "#7aa6d6",   # blue-grey
    "shuffle":  "#f3a953",   # warm orange
    "real":     "#d05656",   # red — the main effect
}


def _load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _per_subject(d: dict) -> dict[int, float]:
    return {s["subject"]: s["best_val_acc"] for s in d["per_subject"]}


def load_dataset(name: str) -> dict:
    """Returns dict with per-subject mean (across seeds) for each condition.

    All conditions, including the eegnet baseline, are averaged across
    the available seeds.
    """
    meta = DATASETS[name]
    # Baseline (averaged across seeds, like the other conditions).
    base_per_seed = []
    sids = None
    for seed, fname in meta["baseline_file_seeds"].items():
        d = _load(RD / fname)
        ps = _per_subject(d)
        if sids is None:
            sids = sorted(ps)
        base_per_seed.append(np.array([ps[s] for s in sids]))
    base_stack = np.stack(base_per_seed, axis=0)
    out = {"name": name, "meta": meta, "sids": sids,
           "baseline": base_stack.mean(axis=0),
           "baseline_n_seeds": base_stack.shape[0]}
    for cond, seed_files in meta["seeds"].items():
        per_seed_arrs = []
        for seed, fname in seed_files.items():
            ps = _per_subject(_load(RD / fname))
            per_seed_arrs.append(np.array([ps[s] for s in sids]))
        stack = np.stack(per_seed_arrs, axis=0)
        out[cond] = stack.mean(axis=0)
        out[cond + "_n_seeds"] = stack.shape[0]
    return out


# ----------------------------------------------------------------------------
# Figure 1 — headline cross-dataset scatter (4 panels: baseline vs Δ_PLV)
# ----------------------------------------------------------------------------
def fig_headline_scatter(datasets: list[dict]) -> Path:
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharey=True)
    for ax, ds in zip(axes, datasets):
        x = ds["baseline"]
        y = ds["real"] - ds["zero"]    # Δ_PLV = real - zero (paired)
        rho, p = stats.spearmanr(x, y)
        ax.scatter(x, y, s=46, c=PALETTE["real"], edgecolors="white",
                   linewidths=0.8, zorder=3)
        # Linear fit overlay (visual aid only)
        if len(x) >= 3:
            xs = np.linspace(x.min(), x.max(), 50)
            slope, intercept = np.polyfit(x, y, 1)
            ax.plot(xs, slope*xs + intercept, color="#333",
                    lw=1.2, alpha=0.55, zorder=2)
        ax.axhline(0, color="#999", lw=0.8, ls="--", zorder=1)
        meta = ds["meta"]
        title = f"{ds['name']}\n{meta['n_channels']}ch · {meta['n_classes']}-cls · n={len(x)}"
        ax.set_title(title, fontsize=10)
        ax.text(0.04, 0.95, f"ρ = {rho:+.2f}\np = {p:.3f}",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=9, family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="#bbb", lw=0.7))
        ax.set_xlabel("EEGNet baseline accuracy")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(r"$\Delta_{PLV}$  (real − zero)")
    fig.suptitle("Cross-dataset: subject-stratified PLV effect "
                 "(channel-count hypothesis fails at Schirrmeister)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    out = FD / "fig1_headline_scatter"
    fig.savefig(f"{out}.png", dpi=180, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------
# Figure 2 — 4-dataset paired ablation bar chart (the structure-specific harm)
# ----------------------------------------------------------------------------
def fig_ablation_bars(datasets: list[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4.2))
    n = len(datasets)
    width = 0.2
    xs = np.arange(n)
    conds = [("baseline", "EEGNet baseline"),
             ("zero",     "PLV-EEGNet (zero)"),
             ("shuffle",  "PLV-EEGNet (shuffled-pair)"),
             ("real",     "PLV-EEGNet (real)")]
    for i, (key, label) in enumerate(conds):
        means = [ds[key].mean() for ds in datasets]
        sds = [ds[key].std() for ds in datasets]
        ax.bar(xs + (i - 1.5)*width, means, width=width, yerr=sds,
               color=PALETTE[key], edgecolor="white", linewidth=0.7,
               capsize=2.5, label=label, error_kw=dict(elinewidth=0.9, alpha=0.7))
    ax.set_xticks(xs)
    ax.set_xticklabels([ds["name"] for ds in datasets])
    ax.set_ylabel("Best val accuracy (mean ± SD across subjects)")
    ax.set_ylim(0.4, 1.0)
    ax.axhline(0.25, color="#bbb", lw=0.8, ls=":")  # 4-class chance
    ax.axhline(0.50, color="#bbb", lw=0.8, ls=":")  # 2-class chance
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(ncol=2, frameon=False, loc="upper right", fontsize=9)
    ax.set_title("Paired ablations: structure-specific harm of real PLV "
                 "(real ≪ shuffle ≈ zero ≈ baseline on 2 of 4 datasets)",
                 fontsize=11)
    fig.tight_layout()
    out = FD / "fig2_ablation_bars"
    fig.savefig(f"{out}.png", dpi=180, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------
# Figure 3 — Schirrmeister band-match retest (ruling out the alternative)
# ----------------------------------------------------------------------------
def fig_schirr_bandmatch() -> Path:
    """Schirrmeister band-match figure (proper-split, 3-seeds for the
    main 3 conditions; single-seed for the notch controls which we already
    showed are no-ops on mu/beta — gamma's small −1.7 pp notch effect is
    surfaced in Table III)."""
    multi = {
        "eegnet @ 8-30 input":   [f"schirrmeister_eegnet_seed{s}_200ep_propersplit.json" for s in (42,43,44)],
        "eegnet @ 1-120 input":  [f"schirrmeister_eegnet_bb1-120_seed{s}_200ep_propersplit.json" for s in (42,43,44)],
        "PLV(8-30) @ 1-120":     [f"schirrmeister_plv_eegnet_bb1-120_phase8-30_seed{s}_200ep_propersplit.json" for s in (42,43,44)],
        "PLV(70-120) @ 1-120":   [f"schirrmeister_plv_eegnet_bb1-120_phase70-120_seed{s}_200ep_propersplit.json" for s in (42,43,44)],
    }
    single = {
        "PLV(8-30) + 50 Hz notch":   "schirrmeister_plv_eegnet_bb1-120_notch50_phase8-30_seed42_200ep_propersplit.json",
        "PLV(70-120) + 50 Hz notch": "schirrmeister_plv_eegnet_bb1-120_notch50_phase70-120_seed42_200ep_propersplit.json",
    }
    means, sds, labels = [], [], []
    for label, files in multi.items():
        # Average per-subject across seeds, then compute mean/sd over subjects
        per_subj = {}
        for fname in files:
            d = _load(RD / fname)
            for s in d["per_subject"]:
                per_subj.setdefault(s["subject"], []).append(s["best_val_acc"])
        sids = sorted(per_subj)
        arr = np.array([np.mean(per_subj[s]) for s in sids])
        means.append(arr.mean())
        sds.append(arr.std())
        labels.append(label)
    for label, fname in single.items():
        d = _load(RD / fname)
        means.append(d["aggregate"]["mean_best"])
        sds.append(d["aggregate"]["std_best"])
        labels.append(label)
    # Reorder for visual comparison: 2 baselines, then mu/beta pair, then gamma pair
    order_idx = [0, 1, 2, 4, 3, 5]
    means = [means[i] for i in order_idx]
    sds = [sds[i] for i in order_idx]
    labels = [labels[i] for i in order_idx]
    fig, ax = plt.subplots(figsize=(10, 4.0))
    colors = ["#444", "#444", "#7aa6d6", "#7aa6d6", "#d05656", "#d05656"]
    bars = ax.bar(range(len(means)), means, yerr=sds,
                  color=colors, edgecolor="white", linewidth=0.7,
                  capsize=3, error_kw=dict(elinewidth=0.9, alpha=0.7))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("Best val accuracy (mean ± SD)")
    ax.set_ylim(0.4, 1.0)
    ax.axhline(0.25, color="#bbb", lw=0.8, ls=":")
    # Annotate with values
    for i, (m, sd) in enumerate(zip(means, sds)):
        ax.text(i, m + sd + 0.012, f"{m:.3f}", ha="center", va="bottom",
                fontsize=8.5, family="monospace")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_title("Schirrmeister band-match retest: high-gamma PLV does not "
                 "rescue, line-noise notch is a no-op",
                 fontsize=11)
    fig.tight_layout()
    out = FD / "fig3_schirr_bandmatch"
    fig.savefig(f"{out}.png", dpi=180, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------
# Figure 4 — Schirrmeister bimodal gamma response (notched vs un-notched)
# ----------------------------------------------------------------------------
def fig_schirr_bimodal() -> Path:
    """Bimodal subject scatter — 3-seed averaged for un-notched, single-seed
    for notched (notch effect is small and well-characterized in Table III)."""
    def avg_seeds(files):
        per_subj = {}
        for fname in files:
            d = _load(RD / fname)
            for s in d["per_subject"]:
                per_subj.setdefault(s["subject"], []).append(s["best_val_acc"])
        return {k: np.mean(v) for k, v in per_subj.items()}
    base  = avg_seeds([f"schirrmeister_eegnet_bb1-120_seed{s}_200ep_propersplit.json" for s in (42,43,44)])
    mb_un = avg_seeds([f"schirrmeister_plv_eegnet_bb1-120_phase8-30_seed{s}_200ep_propersplit.json" for s in (42,43,44)])
    g_un  = avg_seeds([f"schirrmeister_plv_eegnet_bb1-120_phase70-120_seed{s}_200ep_propersplit.json" for s in (42,43,44)])
    mb_n  = _per_subject(_load(RD / "schirrmeister_plv_eegnet_bb1-120_notch50_phase8-30_seed42_200ep_propersplit.json"))
    g_n   = _per_subject(_load(RD / "schirrmeister_plv_eegnet_bb1-120_notch50_phase70-120_seed42_200ep_propersplit.json"))
    sids = sorted(base)

    delta_un = np.array([g_un[s] - mb_un[s] for s in sids])
    delta_n  = np.array([g_n[s]  - mb_n[s]  for s in sids])
    order = np.argsort(delta_un)
    sids_o = [sids[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 4.0))
    x = np.arange(len(sids_o))
    ax.bar(x - 0.18, delta_un[order], width=0.36, color="#9bb6d6",
           edgecolor="white", linewidth=0.6, label="un-notched")
    ax.bar(x + 0.18, delta_n[order], width=0.36, color="#d05656",
           edgecolor="white", linewidth=0.6, label="+50 Hz notch")
    ax.axhline(0, color="#666", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{s:02d}" for s in sids_o], fontsize=9)
    ax.set_ylabel(r"$\Delta$  PLV(70-120) − PLV(8-30)")
    ax.set_xlabel("subject (sorted by un-notched Δ)")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_title("Schirrmeister: gamma-vs-mu/beta PLV per-subject Δ is bimodal "
                 "and reproducible across notched/un-notched runs",
                 fontsize=10.5)
    fig.tight_layout()
    out = FD / "fig4_schirr_bimodal"
    fig.savefig(f"{out}.png", dpi=180, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    datasets = [load_dataset(name) for name in
                ("BCI 2a", "BCI 2b", "Lee2019", "Schirrmeister")]
    print("Figure 1:", fig_headline_scatter(datasets))
    print("Figure 2:", fig_ablation_bars(datasets))
    print("Figure 3:", fig_schirr_bandmatch())
    print("Figure 4:", fig_schirr_bimodal())


if __name__ == "__main__":
    main()
