# Targeted DSP controls: analysis plan

Specified on 2026-08-24 after the global-fixed and no-normalization runs were
launched but before any complete targeted result JSON existed.

Datasets: Lee2019 and HGD. Conditions use seeds 42--44 and the existing
train/inner-validation/evaluation protocol.

Planned subject-paired contrasts after averaging the three seeds:

1. global-fixed permutation minus unpermuted real PLV;
2. resampled-training minus trial-fixed training without PLV normalization;
3. the existing BatchNorm resampled-minus-fixed schedule effect;
4. difference-in-differences: no-normalization schedule effect minus the
   BatchNorm schedule effect.

Each contrast reports its mean in percentage points, paired Cohen's dz, a
10,000-resample subject-bootstrap percentile 95% CI, a two-sided paired t-test,
and a two-sided Wilcoxon signed-rank sensitivity. The two dataset-level
global-fixed-minus-real tests form one Holm family for each test type. The two
normalization-interaction tests form a second Holm family for each test type.
No equivalence margin is specified; failure to reject is not evidence of
equivalence. Seed-level schedule effects are descriptive because each seed also
changes the split and initialization.
