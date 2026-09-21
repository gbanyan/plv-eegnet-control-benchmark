"""Recompute reported numerical summaries from verified evaluation records."""
import json
from pathlib import Path
import numpy as np
from scripts.summarize_revision import make_summary, contrast_statistics

ROOT = Path(__file__).resolve().parent

def read(path):
    return json.loads((ROOT / path).read_text())

def close(a, b, path="root"):
    if isinstance(a, dict):
        assert set(a) == set(b), path
        for k in a:
            close(a[k], b[k], path + "/" + k)
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            close(x, y, path + "/" + str(i))
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        assert np.isclose(a, b, atol=1e-10, rtol=1e-10), (path, a, b)
    else:
        assert a == b, (path, a, b)

def main():
    summary = make_summary(read("configs/dsp_revision_20260919.json"),
                           [ROOT / "runs/dsp-revision-collected" / host for host in ["gblinux", "gblab146"]])
    expected = read("results/revision_manuscript_20260921/main_summary.json")
    for key in ["cells", "contrasts", "batches", "evaluated_jobs", "manifest_sha256"]:
        close(summary[key], expected[key], "main/" + key)
    print("main: all cells, contrasts, family corrections and counts match")
    for name, folder, expected_path, subjects, replicates in [
        ("gcn_hgd", "gcn-revision-collected", "results/revision_manuscript_20260921/schirrmeister_gcn.json", 14, range(2001, 2006)),
        ("gcn_lee", "gcn-lee-revision-collected", "results/revision_manuscript_20260921/lee2019_gcn.json", 54, range(2001, 2006)),
        ("strength", "strength-classifier-v1-20260921", "results/strength_classifier_20260921/summary.json", 14, range(1001, 1006)),
    ]:
        expected = read(expected_path)
        records = {}
        for p in (ROOT / "runs" / folder / "jobs").glob("*/evaluation.json"):
            record = json.loads(p.read_text());job = record["job"]
            records[job["condition"], job["subject"], job["replicate"]] = record
        conditions = ["fixed", "resampled"] if name == "strength" else ["fixed", "resampled", "real", "baseline"]
        for rule in ["primary", "accuracy", "fixed_budget"]:
            values = {c: np.array([[records[c, s, r]["metrics"][rule]["fixed_accuracy"]
                                   for r in replicates] for s in range(1, subjects + 1)]) for c in conditions}
            close({c: float(v.mean() * 100) for c, v in values.items()}, expected["cells"][rule], name + "/cells/" + rule)
            pairs = [("schedule", "resampled", "fixed")] if name == "strength" else [
                ("schedule", "resampled", "fixed"), ("real_minus_identity", "real", "baseline"),
                ("real_minus_fixed", "real", "fixed"), ("real_minus_resampled", "real", "resampled")]
            for label, a, b in pairs:
                actual = contrast_statistics(values[a] - values[b])
                reference = expected["contrasts"][rule if name == "strength" else rule + ":" + label]
                close(actual, {k: reference[k] for k in actual}, name + "/" + rule + "/" + label)
        print(name + ": all checkpoint means and contrasts match")
    print("Numeric tolerance: atol=rtol=1e-10; generic GCN/strength TOST fields do not declare equivalence endpoints.")

if __name__ == "__main__":
    main()
