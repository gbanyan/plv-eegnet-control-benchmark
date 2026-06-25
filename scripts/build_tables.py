"""Build Tables I, II, III for the J-BHI paper draft.

Outputs Markdown-formatted tables to stdout, ready to paste into
paper/draft.md. Run from project root:
    .venv/bin/python -m scripts.build_tables > paper/tables.md
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from scripts.build_figures import DATASETS, RD, load_dataset, _load


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def paired_t(diffs):
    n = len(diffs)
    m = statistics.mean(diffs)
    sd = statistics.stdev(diffs) if n > 1 else 0.0
    if sd == 0:
        return m, sd, float("nan"), float("nan")
    t = m / (sd / math.sqrt(n))
    try:
        from scipy import stats
        p = stats.t.sf(abs(t), n - 1) * 2
    except Exception:
        p = float("nan")
    return m, sd, t, p


def wilcoxon_p(diffs):
    """Two-sided Wilcoxon signed-rank p (sensitivity check vs. paired t).

    Returns NaN if scipy is unavailable, n<6, or all diffs are zero.
    """
    nonzero = [d for d in diffs if d != 0]
    if len(nonzero) < 6:
        return float("nan")
    try:
        from scipy import stats
        # Use exact mode where possible; falls back to approx for larger n.
        return stats.wilcoxon(diffs, alternative="two-sided",
                              zero_method="wilcox", method="auto").pvalue
    except Exception:
        return float("nan")


def fmt_p(p):
    if p is None or math.isnan(p):
        return "—"
    if p < 0.0001:
        return "<0.0001"
    if p < 0.001:
        return f"{p:.4f}"
    return f"{p:.3f}"


# ----------------------------------------------------------------------------
# Table I — Datasets summary
# ----------------------------------------------------------------------------
def table_I():
    print("\n## Table I — Datasets summary\n")
    print("| Property | BCI 2a | BCI 2b | Lee2019 | Schirrmeister 2017 |")
    print("|---|---|---|---|---|")
    rows = [
        ("MOABB code",            "BNCI2014_001", "BNCI2014_004", "Lee2019_MI", "Schirrmeister2017"),
        ("Lab",                   "Graz BCI",     "Graz BCI",     "Korea Univ. (OpenBMI)", "Freiburg"),
        ("Classes",               "4 (L/R/feet/tongue)", "2 (L/R)", "2 (L/R)",   "4 (L/R/feet/rest)"),
        ("Subjects",              "9", "9", "54", "14"),
        ("Channels (EEG)",        "22", "3 (C3, Cz, C4)", "62", "128"),
        ("PLV pairs",             "231", "3", "1891", "8128"),
        ("Native sample rate",    "250 Hz", "250 Hz", "1000 Hz → 250", "500 Hz → 250"),
        ("Trial window",          "4 s (1000 samp)", "4 s (1000 samp)", "4 s (1000 samp)", "4 s (1000 samp)"),
        ("Sessions / subject",    "2 (T,E)", "5 (T+E concat)", "2 (sess. 0,1)", "1 (built-in T,E)"),
        ("T-set trials / subject *(stratified-split 80/20 → inner-train/inner-val)*",
            "288", "≈400", "≈200 (sess. 0)", "320–897"),
        ("E-set trials / subject *(held-out; epoch chosen by inner-val)*",
            "288", "≈320", "≈200 (sess. 1)", "160"),
        ("Chance accuracy",       "0.25", "0.50", "0.50", "0.25"),
    ]
    for r in rows:
        print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |")


# ----------------------------------------------------------------------------
# Table II — 4-dataset paired ablation
# ----------------------------------------------------------------------------
def table_II():
    print("\n## Table II — Four-dataset paired ablation contrasts\n")
    datasets = [load_dataset(name) for name in DATASETS.keys()]

    # Header for per-condition means
    print("**Per-condition means (eval-at-best-inner-val accuracy, mean ± SD):**\n")
    print("| Dataset | n | EEGNet baseline | Zero-PLV | Shuffled-pair PLV | Real PLV |")
    print("|---|---|---|---|---|---|")
    for ds in datasets:
        n = len(ds["sids"])
        line = f"| {ds['name']} | {n} |"
        for cond in ("baseline", "zero", "shuffle", "real"):
            arr = ds[cond]
            m = arr.mean()
            sd = arr.std()
            line += f" {m:.3f} ± {sd:.3f} |"
        print(line)

    # Combined paired-t + Wilcoxon: cell shows Δ (p_t / p_W).
    print("\n**Paired contrasts (Δ in percentage points; cell shows "
          "Δ (*p_t* / *p_W*), with *p_t* paired *t*-test, *p_W* Wilcoxon "
          "signed-rank, both two-sided):**\n")
    print("| Dataset | n | Δ(zero − base) | Δ(shuffle − zero) | **Δ(real − shuffle)** | **Δ(real − zero)** |")
    print("|---|---|---|---|---|---|")
    for ds in datasets:
        n = len(ds["sids"])
        line = f"| {ds['name']} | {n} |"
        contrasts = [
            (ds["zero"] - ds["baseline"]),
            (ds["shuffle"] - ds["zero"]),
            (ds["real"] - ds["shuffle"]),
            (ds["real"] - ds["zero"]),
        ]
        for diffs in contrasts:
            m, sd, t, p = paired_t(diffs.tolist())
            p_w = wilcoxon_p(diffs.tolist())
            line += f" {100*m:+.2f} pp ({fmt_p(p)}/{fmt_p(p_w)}) |"
        print(line)
    print("\n*Bold contrasts are the central methodological tests: "
          "(real−shuffle) isolates pair-indexed PLV structure from "
          "same-marginal noise injection; (real−zero) isolates "
          "pair-indexed PLV structure from architectural overhead. "
          "Neither contrast by itself isolates neural phase-coupling "
          "from joint-distribution and feature-stability effects "
          "(see Sec. III-D, V-A in draft). Paired *t* assumes "
          "approximate normality of paired differences; Wilcoxon "
          "signed-rank is reported as a non-parametric sensitivity "
          "check appropriate at small *n*.*")


# ----------------------------------------------------------------------------
# Table III — Schirrmeister band-match
# ----------------------------------------------------------------------------
def table_III():
    print("\n## Table III — Schirrmeister band-match retest contrasts\n")
    # Proper-split + 3 seeds for the main 3 broadband-input conditions;
    # single-seed retained for the notch controls (notch is independently
    # established as a no-op below).
    multi_seed = {
        "EEGNet @ 8-30 Hz input":     [f"schirrmeister_eegnet_seed{s}_200ep_propersplit.json" for s in (42,43,44)],
        "EEGNet @ 1-120 Hz input":    [f"schirrmeister_eegnet_bb1-120_seed{s}_200ep_propersplit.json" for s in (42,43,44)],
        "PLV(8-30) @ 1-120 Hz":       [f"schirrmeister_plv_eegnet_bb1-120_phase8-30_seed{s}_200ep_propersplit.json" for s in (42,43,44)],
        "PLV(70-120) @ 1-120 Hz":     [f"schirrmeister_plv_eegnet_bb1-120_phase70-120_seed{s}_200ep_propersplit.json" for s in (42,43,44)],
    }
    single_seed = {
        "PLV(8-30) @ 1-120 Hz + 50 Hz notch":   "schirrmeister_plv_eegnet_bb1-120_notch50_phase8-30_seed42_200ep_propersplit.json",
        "PLV(70-120) @ 1-120 Hz + 50 Hz notch": "schirrmeister_plv_eegnet_bb1-120_notch50_phase70-120_seed42_200ep_propersplit.json",
    }
    print("**Per-condition means (n=14 subjects):**\n")
    print("| Configuration | Mean | SD | n_seeds |")
    print("|---|---|---|---|")
    per_subj = {}
    for label, files in multi_seed.items():
        # average across seeds per subject
        vals = {}
        for fname in files:
            d = _load(RD / fname)
            for s in d["per_subject"]:
                vals.setdefault(s["subject"], []).append(s["best_val_acc"])
        avg = {k: sum(v)/len(v) for k, v in vals.items()}
        per_subj[label] = avg
        sids = sorted(avg)
        arr = [avg[s] for s in sids]
        m = sum(arr)/len(arr)
        sd = (sum((x-m)**2 for x in arr)/(len(arr)-1))**0.5
        print(f"| {label} | {m:.4f} | {sd:.4f} | {len(files)} |")
    for label, fname in single_seed.items():
        d = _load(RD / fname)
        agg = d["aggregate"]
        per_subj[label] = {s["subject"]: s["best_val_acc"] for s in d["per_subject"]}
        print(f"| {label} | {agg['mean_best']:.4f} | {agg['std_best']:.4f} | 1 |")

    # Paired contrasts
    print("\n**Paired contrasts (Δ in percentage points, paired *t*-test, n=14):**\n")
    print("| Contrast | Δ (pp) | t | p |")
    print("|---|---|---|---|")
    sids = sorted(next(iter(per_subj.values())).keys())
    pairs = [
        ("PLV(70-120) − PLV(8-30) at 1-120 input  (key band-match contrast)",
         "PLV(70-120) @ 1-120 Hz", "PLV(8-30) @ 1-120 Hz"),
        ("PLV(70-120) notched − un-notched  (line-noise control)",
         "PLV(70-120) @ 1-120 Hz + 50 Hz notch", "PLV(70-120) @ 1-120 Hz"),
        ("PLV(8-30) notched − un-notched  (line-noise control)",
         "PLV(8-30) @ 1-120 Hz + 50 Hz notch", "PLV(8-30) @ 1-120 Hz"),
        ("PLV(70-120) − EEGNet baseline at 1-120 input",
         "PLV(70-120) @ 1-120 Hz", "EEGNet @ 1-120 Hz input"),
        ("PLV(8-30) − EEGNet baseline at 1-120 input",
         "PLV(8-30) @ 1-120 Hz", "EEGNet @ 1-120 Hz input"),
        ("EEGNet broadband − EEGNet 8-30 input",
         "EEGNet @ 1-120 Hz input", "EEGNet @ 8-30 Hz input"),
    ]
    for label, a, b in pairs:
        diffs = [per_subj[a][s] - per_subj[b][s] for s in sids]
        m, sd, t, p = paired_t(diffs)
        print(f"| {label} | {100*m:+.2f} | {t:+.3f} | {fmt_p(p)} |")


def table_IV():
    """Volume-conduction sensitivity via zero-lag-resistant variants
    (PLI and wPLI), high-channel datasets."""
    print("\n## Table IV — Volume-conduction sensitivity check via "
          "zero-lag-resistant variants (PLI, wPLI), 3 seeds averaged\n")
    print("**Paired contrasts (Δ in percentage points; paired *t*-test, "
          "two-sided; Wilcoxon signed-rank as non-parametric sensitivity):**\n")
    print("| Dataset | n | Contrast | Δ (pp) | paired *t* | "
          "paired-*t* *p* | Wilcoxon *p* |")
    print("|---|---|---|---|---|---|---|")
    seeds = (42, 43, 44)
    datasets = [
        ("Lee2019", "lee2019"),
        ("Schirrmeister", "schirrmeister"),
    ]
    for ds_label, stem in datasets:
        # Load per-subject seed-averaged values for each condition.
        def load_avg(suffix):
            vals = {}
            for s in seeds:
                d = _load(RD / f"{stem}_plv_eegnet{suffix}_seed{s}_200ep_propersplit.json")
                for r in d["per_subject"]:
                    vals.setdefault(r["subject"], []).append(r["best_val_acc"])
            sids = sorted(vals)
            return sids, [sum(vals[s]) / len(vals[s]) for s in sids]
        sids_real, real    = load_avg("")
        sids_zero, zero    = load_avg("_zero")
        sids_shuf, shuffle = load_avg("_shuffle_pairs")
        sids_pli,  pli     = load_avg("_pli")
        sids_wpli, wpli    = load_avg("_wpli")
        assert sids_real == sids_zero == sids_shuf == sids_pli == sids_wpli, \
            f"subject mismatch in {ds_label}"
        n = len(sids_real)
        contrasts = [
            ("real (PLV) − zero", real, zero),
            ("PLI  − zero",       pli,  zero),
            ("PLI  − shuffle",    pli,  shuffle),
            ("PLI  − real (PLV)", pli,  real),
            ("wPLI − zero",       wpli, zero),
            ("wPLI − shuffle",    wpli, shuffle),
            ("wPLI − real (PLV)", wpli, real),
            ("wPLI − PLI  (head-to-head)", wpli, pli),
        ]
        for label, a, b in contrasts:
            diffs = [ai - bi for ai, bi in zip(a, b)]
            m, sd, t, p = paired_t(diffs)
            p_w = wilcoxon_p(diffs)
            print(f"| {ds_label} | {n} | {label} | {100*m:+.2f} | "
                  f"{t:+.2f} | {fmt_p(p)} | {fmt_p(p_w)} |")
    print("\n*Substituting PLI or wPLI for PLV in the auxiliary head "
          "removes the exact zero-lag component of the coupling estimate "
          "(PLI by sign-thresholding; wPLI by debiased magnitude weighting "
          "[Vinck et al. 2011]). The harm direction is preserved and "
          "amplified across both datasets, both metrics, and both "
          "paired-*t* and Wilcoxon, ruling out a zero-lag-only "
          "volume-conduction explanation. Critically, the wPLI − PLI "
          "head-to-head contrast is null on both datasets (Lee2019 "
          "+0.83 pp, n.s.; Schirrmeister +0.00 pp, p=1.000), so the "
          "harm cannot be attributed to PLI's lower information content "
          "(sign-thresholding) — wPLI carries magnitude information yet "
          "produces the same harm pattern. Residual VC channels not "
          "addressed by sign-thresholding (near-zero-lag, broadband "
          "common-source, short-distance spatial leakage) require "
          "CSD/Laplacian preprocessing or distance-stratified pair "
          "removal as further sensitivity work.*")


def table_V():
    """Schirrmeister CSD volume-conduction sensitivity (preprocessing-side
    test, complement to metric-side PLI/wPLI in Table IV)."""
    print("\n## Table V — Schirrmeister CSD/Laplacian sensitivity check "
          "(n=14, 3 seeds averaged)\n")
    print("**Per-condition means (eval-at-best-inner-val accuracy, "
          "mean ± SD; raw vs CSD-preprocessed input):**\n")
    print("| Condition | raw mean ± SD | CSD mean ± SD |")
    print("|---|---|---|")
    seeds = (42, 43, 44)
    def load_seed_avg(stem, model):
        vals = {}
        for s in seeds:
            d = _load(RD / f"{stem}_{model}_seed{s}_200ep_propersplit.json")
            for r in d["per_subject"]:
                vals.setdefault(r["subject"], []).append(r["best_val_acc"])
        sids = sorted(vals)
        return sids, [sum(vals[s]) / len(vals[s]) for s in sids]
    rows = []
    for cond_label, model in [
        ("EEGNet baseline",        "eegnet"),
        ("Zero-PLV",               "plv_eegnet_zero"),
        ("Shuffled-pair PLV",      "plv_eegnet_shuffle_pairs"),
        ("Real PLV",               "plv_eegnet"),
    ]:
        sids_raw, raw = load_seed_avg("schirrmeister", model)
        sids_csd, csd = load_seed_avg("schirrmeister_csd", model)
        assert sids_raw == sids_csd, f"subject mismatch for {model}"
        import statistics as _s
        raw_m = _s.mean(raw); raw_sd = _s.stdev(raw)
        csd_m = _s.mean(csd); csd_sd = _s.stdev(csd)
        print(f"| {cond_label} | {100*raw_m:.2f} ± {100*raw_sd:.2f} | "
              f"{100*csd_m:.2f} ± {100*csd_sd:.2f} |")
        rows.append((cond_label, raw, csd))
    print("\n**Within-CSD paired contrasts (Δ in pp; paired *t*-test, "
          "two-sided; Wilcoxon signed-rank as non-parametric sensitivity):**\n")
    print("| Contrast | Δ (pp) | paired *t* | paired-*t* *p* | Wilcoxon *p* |")
    print("|---|---|---|---|---|")
    csd_by = {label: csd for label, raw, csd in rows}
    contrasts_within = [
        ("real (CSD) − shuffle (CSD)  (primary structure-specific)",
            "Real PLV", "Shuffled-pair PLV"),
        ("real (CSD) − zero (CSD)",      "Real PLV", "Zero-PLV"),
        ("shuffle (CSD) − zero (CSD)",   "Shuffled-pair PLV", "Zero-PLV"),
        ("zero (CSD) − baseline (CSD)  (arch overhead)",
            "Zero-PLV", "EEGNet baseline"),
        ("real (CSD) − baseline (CSD)  (naive)",
            "Real PLV", "EEGNet baseline"),
    ]
    for label, a, b in contrasts_within:
        diffs = [ai - bi for ai, bi in zip(csd_by[a], csd_by[b])]
        m, sd, t, p = paired_t(diffs)
        p_w = wilcoxon_p(diffs)
        print(f"| {label} | {100*m:+.2f} | {t:+.2f} | {fmt_p(p)} | {fmt_p(p_w)} |")
    print("\n**Harm attenuation by CSD (raw contrast vs CSD contrast):**\n")
    print("| Contrast | raw Δ (pp) | CSD Δ (pp) | attenuation |")
    print("|---|---|---|---|")
    raw_by = {label: raw for label, raw, csd in rows}
    for label, a, b in contrasts_within[:3]:
        raw_diffs = [ai - bi for ai, bi in zip(raw_by[a], raw_by[b])]
        csd_diffs = [ai - bi for ai, bi in zip(csd_by[a], csd_by[b])]
        rm, _, _, _ = paired_t(raw_diffs)
        cm, _, _, _ = paired_t(csd_diffs)
        if abs(rm) > 1e-9:
            atten = 100.0 * (1.0 - abs(cm) / abs(rm))
        else:
            atten = float("nan")
        print(f"| {label} | {100*rm:+.2f} | {100*cm:+.2f} | "
              f"{atten:.0f}% absorbed by CSD |")
    print("\n*CSD (spherical-spline current source density, Perrin et al. "
          "1989, applied via mne.preprocessing.compute_current_source_density "
          "before the dataset-level 8-30 Hz bandpass) is a spatial high-pass "
          "preprocessing step that attenuates volume-conduction by reweighting "
          "each channel against its surface Laplacian. Unlike the PLI/wPLI "
          "checks in Table IV (metric-side, leaving the input signal "
          "unchanged), CSD modifies the input signal seen by both the EEGNet "
          "trunk and the PLV head; a same-marginal harm that survives CSD "
          "therefore cannot be a pure volume-conduction artifact even after "
          "near-zero-lag and short-distance common-source attenuation. The "
          "(real − shuffle) and (real − zero) contrasts both shrink under CSD "
          "(by ~46% and ~39% respectively) but remain highly significant, "
          "indicating that VC plausibly accounts for roughly half of the "
          "Schirrmeister harm while the residual half is non-VC. The "
          "architectural-overhead null (zero ≈ baseline) is preserved under "
          "CSD. Lee2019 CSD is left for future work due to MOABB cache "
          "constraints.*")


def table_VI():
    """Schirrmeister bottleneck (PLV head 8128->16->concat) sensitivity
    test for the dimensionality-driven harm hypothesis."""
    print("\n## Table VI — Schirrmeister bottleneck sensitivity check "
          "(PLV head $P\\!=\\!8128 \\to 16$ projection, n=14, 3 seeds averaged)\n")
    print("**Per-condition means (eval-at-best-inner-val accuracy, "
          "mean ± SD; raw vs bottleneck-projected PLV head):**\n")
    print("| Condition | raw mean ± SD (P=8128) | bottleneck mean ± SD (P=16) |")
    print("|---|---|---|")
    seeds = (42, 43, 44)
    def load_seed_avg(stem, model):
        vals = {}
        for s in seeds:
            d = _load(RD / f"{stem}_{model}_seed{s}_200ep_propersplit.json")
            for r in d["per_subject"]:
                vals.setdefault(r["subject"], []).append(r["best_val_acc"])
        sids = sorted(vals)
        return sids, [sum(vals[s]) / len(vals[s]) for s in sids]
    rows = []
    pairs_for_bn = [
        ("EEGNet baseline (no PLV head)", "eegnet", None),
        ("Zero-PLV",                       "plv_eegnet_zero",          "plv_eegnet_zero_bn16"),
        ("Shuffled-pair PLV",              "plv_eegnet_shuffle_pairs", "plv_eegnet_shuffle_pairs_bn16"),
        ("Real PLV",                       "plv_eegnet",               "plv_eegnet_bn16"),
    ]
    import statistics as _s
    for cond_label, model_raw, model_bn in pairs_for_bn:
        _, raw = load_seed_avg("schirrmeister", model_raw)
        rm, rs = _s.mean(raw), _s.stdev(raw)
        if model_bn is None:
            print(f"| {cond_label} | {100*rm:.2f} ± {100*rs:.2f} | — (no PLV head to project) |")
            rows.append((cond_label, raw, None))
        else:
            _, bn = load_seed_avg("schirrmeister", model_bn)
            bm, bs = _s.mean(bn), _s.stdev(bn)
            print(f"| {cond_label} | {100*rm:.2f} ± {100*rs:.2f} | {100*bm:.2f} ± {100*bs:.2f} |")
            rows.append((cond_label, raw, bn))
    print("\n**Within-bottleneck paired contrasts (Δ in pp; paired *t*-test, "
          "two-sided; Wilcoxon signed-rank as non-parametric sensitivity):**\n")
    print("| Contrast | Δ (pp) | paired *t* | paired-*t* *p* | Wilcoxon *p* |")
    print("|---|---|---|---|---|")
    bn_by = {label: bn for label, raw, bn in rows if bn is not None}
    bn_by["EEGNet baseline (no PLV head)"] = rows[0][1]  # baseline = raw EEGNet, no PLV head
    bn_contrasts = [
        ("real_bn − shuffle_bn  (primary structure-specific)",
            "Real PLV", "Shuffled-pair PLV"),
        ("real_bn − zero_bn",                "Real PLV", "Zero-PLV"),
        ("shuffle_bn − zero_bn",             "Shuffled-pair PLV", "Zero-PLV"),
        ("zero_bn − baseline  (arch overhead)",
            "Zero-PLV", "EEGNet baseline (no PLV head)"),
    ]
    for label, a, b in bn_contrasts:
        diffs = [ai - bi for ai, bi in zip(bn_by[a], bn_by[b])]
        m, sd, t, p = paired_t(diffs)
        p_w = wilcoxon_p(diffs)
        print(f"| {label} | {100*m:+.2f} | {t:+.2f} | {fmt_p(p)} | {fmt_p(p_w)} |")
    print("\n**Bottleneck attenuation (does compression remove the harm?):**\n")
    print("| Contrast | raw Δ (pp) | bottleneck Δ (pp) | net change |")
    print("|---|---|---|---|")
    raw_by = {label: raw for label, raw, bn in rows}
    for label, a, b in bn_contrasts[:3]:
        raw_diffs = [ai - bi for ai, bi in zip(raw_by[a], raw_by[b])]
        bn_diffs  = [ai - bi for ai, bi in zip(bn_by[a],  bn_by[b])]
        rm, _, _, _ = paired_t(raw_diffs)
        bm, _, _, _ = paired_t(bn_diffs)
        print(f"| {label} | {100*rm:+.2f} | {100*bm:+.2f} | "
              f"{100*(bm-rm):+.2f} pp |")
    print("\n*The PLV head is augmented with a `Linear(P, 16)` projection "
          "after BatchNorm and before concatenation with the EEGNet trunk's "
          "flat embedding, compressing the 8128-pair vector to a 16-dim "
          "representation matched to the trunk's output dimension. Tests the alternative hypothesis "
          "that the (real − shuffle) harm is driven by the FC layer overfitting "
          "an 8128-feature vector with $\\sim 500$ training trials per subject "
          "($p \\gg n$ regime), with shuffle's per-trial randomization acting as "
          "implicit regularization. Result: **the (real − shuffle) and (real − "
          "zero) harms survive bottleneck compression essentially unchanged** "
          "(real_bn $\\approx$ real_raw, $p\\!=\\!0.68$; the (real − shuffle) "
          "contrast is even slightly stronger under bottleneck because shuffle's "
          "residual marginal-distribution effect vanishes — shuffle_bn − zero_bn "
          "drops from $-2.22$ pp ($p\\!=\\!0.003$) to $-0.15$ pp (n.s.)). "
          "The structure-specific harm is therefore not a dimensionality "
          "artifact: a 16-d compressed representation of real PLV reproduces "
          "the same harm as the raw 8128-d vector. A residual marginal-"
          "distribution effect previously visible in the raw shuffle-zero "
          "contrast appears to have been a high-dim FC-layer artifact rather "
          "than a true marginal-distribution effect.*")


def main():
    table_I()
    table_II()
    table_III()
    table_IV()
    table_V()
    table_VI()


if __name__ == "__main__":
    main()
