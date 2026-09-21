"""Rebuild the constrained-bank table from its versioned result snapshot."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path('results/strength_classifier_20260921')
    data = json.loads((root / 'summary.json').read_text())
    assert data['complete'] and data['fits'] == 140
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Checkpoint & Fixed & Resampled & $R-F$ & 95\% CI \\", r"\midrule"]
    for rule, label in [('primary', 'Validation loss'), ('accuracy', 'Validation accuracy'), ('fixed_budget', 'Fixed budget')]:
        cell = data['cells'][rule]; result = data['contrasts'][rule]
        lo, hi = result['model_95ci_pp']
        lines.append(f"{label} & {cell['fixed']:.2f} & {cell['resampled']:.2f} & +{result['mean_difference_pp']:.2f} & [{lo:.2f}, {hi:.2f}] " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    table = Path('paper/revision_2026-09/tables/strength.tex')
    table.write_text('\n'.join(lines) + '\n')
    paths = [root/'summary.json', root/'verified_accuracy_grid.json', root/'training_seal.json', root/'E_bank_seal.json', root/'E_bank_QA.json', table]
    (root/'asset_ledger.json').write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, indent=2)+'\n')


if __name__ == '__main__':
    main()
