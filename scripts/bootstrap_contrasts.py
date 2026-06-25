"""Per-seed contrast values + subject-bootstrap CIs for primary contrasts.

Addresses two reviewer concerns:
1. "3 seeds is thin" — by reporting each seed's contrast separately, reader
   can judge seed-to-seed variance directly.
2. "Δ has no CI" — by adding subject-level bootstrap 95% CI on the
   seed-averaged Δ, alongside the existing paired-t and Wilcoxon p-values.

Covers Lee2019 and Schirrmeister (the high-channel datasets where the
primary contrasts live), and includes wPLI when its results JSONs are
available (otherwise skips with a notice).

Run from project root:
    .venv/bin/python -m scripts.bootstrap_contrasts > paper/contrast_cis.md
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

R = Path("results")
SEEDS = (42, 43, 44)


def load_per_seed(stem: str, suffix: str, dataset_subjects: int) -> np.ndarray | None:
    """Return [n_seeds, n_subjects] array of eval-at-best-inner-val accuracies.

    Returns None if any seed's JSON is missing.
    """
    rows = []
    for s in SEEDS:
        p = R / f"{stem}_plv_eegnet{suffix}_seed{s}_200ep_propersplit.json"
        if not p.exists():
            return None
        d = json.loads(p.read_text())
        per = sorted(d["per_subject"], key=lambda r: r["subject"])
        if len(per) != dataset_subjects:
            return None
        rows.append([r["best_val_acc"] for r in per])
    return np.array(rows)


def load_baseline_per_seed(stem: str, dataset_subjects: int) -> np.ndarray | None:
    rows = []
    for s in SEEDS:
        p = R / f"{stem}_eegnet_seed{s}_200ep_propersplit.json"
        if not p.exists():
            return None
        d = json.loads(p.read_text())
        per = sorted(d["per_subject"], key=lambda r: r["subject"])
        if len(per) != dataset_subjects:
            return None
        rows.append([r["best_val_acc"] for r in per])
    return np.array(rows)


def bootstrap_subject_ci(
    diffs: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Subject-level bootstrap percentile 95% CI on the mean of paired diffs."""
    rng = np.random.default_rng(seed)
    n = len(diffs)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        means[i] = diffs[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return diffs.mean(), lo, hi


def fmt_p(p: float) -> str:
    if math.isnan(p):
        return "—"
    if p < 0.0001:
        return "<0.0001"
    if p < 0.001:
        return f"{p:.4f}"
    return f"{p:.3f}"


def report_contrast(name: str, a_seeds: np.ndarray, b_seeds: np.ndarray, *,
                    boot_seed: int) -> None:
    """a_seeds, b_seeds: [n_seeds, n_subjects]. Reports per-seed Δ then
    seed-averaged Δ with paired-t, Wilcoxon, and bootstrap 95% CI."""
    n_seeds, n_subj = a_seeds.shape
    # Per-seed Δ (subject-mean of pairwise diffs).
    per_seed_delta = (a_seeds - b_seeds).mean(axis=1) * 100  # in pp
    # Seed-averaged per-subject pairwise diff -> primary contrast.
    diffs = (a_seeds.mean(axis=0) - b_seeds.mean(axis=0))
    delta_pp = diffs.mean() * 100
    t_stat, p_t = stats.ttest_rel(a_seeds.mean(axis=0), b_seeds.mean(axis=0))
    w_stat, p_w = stats.wilcoxon(diffs, alternative="two-sided",
                                 zero_method="wilcox", method="auto")
    pt, lo, hi = bootstrap_subject_ci(diffs * 100, n_boot=2000, seed=boot_seed)
    print(f"  {name}")
    print(f"    per-seed Δ (s42, s43, s44):  "
          f"{per_seed_delta[0]:+.2f}, {per_seed_delta[1]:+.2f}, "
          f"{per_seed_delta[2]:+.2f} pp  "
          f"(seed range = {per_seed_delta.max()-per_seed_delta.min():.2f} pp)")
    print(f"    seed-avg Δ = {delta_pp:+.2f} pp  "
          f"95% CI [{lo:+.2f}, {hi:+.2f}]  "
          f"paired-t p = {fmt_p(p_t)}  Wilcoxon p = {fmt_p(p_w)}")


def main() -> None:
    print("# Per-seed contrast values + subject-bootstrap 95% CIs\n")
    print("Per-seed Δ shows the contrast computed within each seed independently "
          "(variance across seeds reflects training-init noise).\n"
          "Seed-avg Δ is the primary reported metric (per-subject seed-mean, "
          "then paired across subjects).\n"
          "Bootstrap 95% CI is subject-level (n_boot=2000).\n")
    boot_counter = 11
    for ds_label, stem, n_subj in [
        ("Lee2019",       "lee2019",       54),
        ("Schirrmeister", "schirrmeister", 14),
    ]:
        print(f"\n## {ds_label} (n={n_subj})\n")
        baseline = load_baseline_per_seed(stem, n_subj)
        zero     = load_per_seed(stem, "_zero",          n_subj)
        shuffle  = load_per_seed(stem, "_shuffle_pairs", n_subj)
        real     = load_per_seed(stem, "",               n_subj)
        pli      = load_per_seed(stem, "_pli",           n_subj)
        wpli     = load_per_seed(stem, "_wpli",          n_subj)
        if any(a is None for a in (baseline, zero, shuffle, real, pli)):
            print(f"  [skip] missing required JSONs for {ds_label}")
            continue
        contrasts = [
            ("real − shuffle (primary structure-specific contrast)", real, shuffle),
            ("real − zero    (overhead+structure contrast)",         real, zero),
            ("real − base    (naive with-PLV vs. without-PLV)",      real, baseline),
            ("PLI − zero     (volume-conduction-resistant variant)", pli,  zero),
            ("PLI − shuffle",                                        pli,  shuffle),
            ("PLI − real(PLV)",                                      pli,  real),
        ]
        if wpli is not None:
            contrasts += [
                ("wPLI − zero    (debiased VC-resistant variant)",   wpli, zero),
                ("wPLI − shuffle",                                   wpli, shuffle),
                ("wPLI − real(PLV)",                                 wpli, real),
                ("wPLI − PLI     (wPLI vs PLI head-to-head)",        wpli, pli),
            ]
        else:
            print("  *(wPLI JSONs not yet available; wPLI rows skipped)*\n")
        for name, a, b in contrasts:
            report_contrast(name, a, b, boot_seed=boot_counter)
            boot_counter += 1


if __name__ == "__main__":
    main()
