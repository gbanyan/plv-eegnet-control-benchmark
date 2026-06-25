"""Analyze the two reviewer-requested validation experiments.

#1 Fixed-permutation shuffle: does the (real - shuffle) gap survive when the
   shuffle permutation is FIXED per trial across epochs (i.e. the per-epoch
   reshuffling stochastic-regularizer is removed)? If the fixed gap matches the
   per-epoch gap, the gap is a same-marginal pair-position effect, not a
   reshuffling artifact.

#2 Lee2019 bottleneck: does the (real - shuffle) gap survive a P->16 bottleneck
   on the second high-channel dataset? If so, the effect is not dimensionality
   (p >> n) driven.

Uses the same per-subject loader and seed-averaging as the paper tables.
Run on gblinux: .venv/bin/python -m scripts.analyze_new_exps
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from scripts.build_figures import RD, _load, _per_subject

SEEDS = (42, 43, 44)


def avg_per_subject(fmt):
    """Mean per-subject eval-at-best-inner-val across seeds; None if any file missing."""
    arrs, sids = [], None
    for s in SEEDS:
        path = RD / fmt.format(seed=s)
        if not path.exists():
            return None, None
        ps = _per_subject(_load(path))
        if sids is None:
            sids = sorted(ps)
        arrs.append(np.array([ps[i] for i in sids]))
    return np.stack(arrs).mean(axis=0), sids


def contrast(a, b, label):
    if a is None or b is None:
        print(f"  {label}: <pending — result files not all present yet>")
        return
    diff = a - b
    t, p = stats.ttest_rel(a, b)
    try:
        w = stats.wilcoxon(a, b).pvalue
    except ValueError:
        w = float("nan")
    print(f"  {label}: {100*diff.mean():+.2f} pp  "
          f"(t={t:+.2f}, p={p:.4f}, wilcoxon={w:.4f}, n={len(diff)})")


def main():
    print("\n=== #1  Fixed-permutation shuffle (stochastic-regularizer control) ===")
    for ds in ("lee2019", "schirrmeister"):
        real, _ = avg_per_subject(ds + "_plv_eegnet_seed{seed}_200ep_propersplit.json")
        shuf, _ = avg_per_subject(ds + "_plv_eegnet_shuffle_pairs_seed{seed}_200ep_propersplit.json")
        fix, _ = avg_per_subject(ds + "_plv_eegnet_shuffle_pairs_fixed_seed{seed}_200ep_propersplit.json")
        print(f"\n[{ds}]")
        contrast(real, shuf, "real - shuffle (per-epoch, published)")
        contrast(real, fix,  "real - shuffle_FIXED (epoch-stable)  ")

    print("\n=== #2  Lee2019 bottleneck P=1891 -> 16 (dimensionality control) ===")
    realbn, _ = avg_per_subject("lee2019_plv_eegnet_bn16_seed{seed}_200ep_propersplit.json")
    shufbn, _ = avg_per_subject("lee2019_plv_eegnet_shuffle_pairs_bn16_seed{seed}_200ep_propersplit.json")
    zerobn, _ = avg_per_subject("lee2019_plv_eegnet_zero_bn16_seed{seed}_200ep_propersplit.json")
    print()
    contrast(realbn, shufbn, "real(bn16) - shuffle(bn16)")
    contrast(realbn, zerobn, "real(bn16) - zero(bn16)   ")
    print("\n(reference: non-bottleneck Lee2019 real-shuffle = -2.80 pp, p=0.011)")


if __name__ == "__main__":
    main()
