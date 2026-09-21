"""Verify archived evaluations without raw EEG or model checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text())

def check(condition, message):
    if not condition:
        raise ValueError(message)

def main():
    report = {"checkpoint_files_verified": False, "raw_EEG_verified": False,
              "full_surrogate_banks_verified": False, "studies": {}}
    studies = [
        ("main", "dsp_revision_20260919.json", ["dsp-revision-collected/gblinux", "dsp-revision-collected/gblab146"]),
        ("gcn_hgd", "gcn_revision_v1_20260920.json", ["gcn-revision-collected"]),
        ("gcn_lee", "gcn_lee_revision_v1_20260920.json", ["gcn-lee-revision-collected"]),
        ("strength", "strength_classifier_v1_20260921.json", ["strength-classifier-v1-20260921"]),
    ]
    for name, config, folders in studies:
        manifest = read(ROOT / "configs" / config)
        mh = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        for path, expected in manifest["source_sha256"].items():
            check(sha(ROOT / path) == expected, "frozen source: " + path)
        records = {}
        seals = {}
        for folder in folders:
            root = ROOT / "runs" / folder
            for path in (root / "seals").glob("*.json"):
                seals[sha(path)] = read(path)
            for path in (root / "jobs").glob("*/evaluation.json"):
                record = read(path)
                check(record["job"]["id"] not in records, "duplicate evaluated job")
                records[record["job"]["id"]] = (path, record)
        expected = {job["id"] for job in manifest["jobs"]}
        check(set(records) == expected, name + ": incomplete or foreign grid")
        fixed_metrics = 0
        mc_metrics = 0
        for identifier, (path, record) in records.items():
            directory = path.parent
            check(record["manifest_sha256"] == mh, identifier + ": manifest")
            th = sha(directory / "train.json")
            check(th == record["training_record_sha256"], identifier + ": training record")
            train = read(directory / "train.json")
            check(train["E_set_accessed"] is False, identifier + ": training boundary")
            check(sha(directory / "curves.json") == train["curves_sha256"], identifier + ": curves")
            check(record["seal_sha256"] in seals, identifier + ": missing seal")
            seal = seals[record["seal_sha256"]]
            check(seal["records"][identifier] == th, identifier + ": sealed record")
            check(seal["manifest_sha256"] == mh, identifier + ": sealed manifest")
            check(sha(directory / "predictions.npz") == record["predictions_sha256"], identifier + ": predictions")
            with np.load(directory / "predictions.npz", allow_pickle=False) as arrays:
                labels = arrays["labels"]
                check(len(labels) == record["evaluation_trials"], identifier + ": trial count")
                for rule, metrics in record["metrics"].items():
                    probabilities = arrays[rule + "_fixed_probabilities"]
                    check(np.isfinite(probabilities).all(), identifier + ": nonfinite probabilities")
                    accuracy = np.mean(probabilities.argmax(-1) == labels)
                    check(np.isclose(accuracy, metrics["fixed_accuracy"], atol=1e-12, rtol=0), identifier + ": accuracy")
                    fixed_metrics += 1
                    if "mc_mean_accuracy" in metrics:
                        draws = arrays[rule + "_mc_probabilities"]
                        acc = (draws.argmax(-1) == labels).mean(-1)
                        checks = {"mc_mean_accuracy": acc.mean(), "mc_sd_accuracy": acc.std(ddof=1),
                                  "mc_standard_error": acc.std(ddof=1) / np.sqrt(len(acc)),
                                  "probability_ensemble_accuracy": np.mean(draws.mean(0).argmax(-1) == labels)}
                        for key, value in checks.items():
                            check(np.isclose(value, metrics[key], atol=1e-7, rtol=0), identifier + ": " + key)
                            mc_metrics += 1
        report["studies"][name] = {"fits": len(records), "fixed_metrics_recomputed": fixed_metrics,
                                     "mc_metrics_recomputed": mc_metrics, "source_hashes": len(manifest["source_sha256"])}
    report["bank_record_hashes"] = {}
    for partition, folder, seal_path in [
        ("T", "strength-bank-T-v3-all-20260921", "results/strength_v3_refinement_20260921/T_bank_seal.json"),
        ("E", "strength-bank-E-v3-20260921", "results/strength_classifier_20260921/E_bank_seal.json"),
    ]:
        seal = read(ROOT / seal_path)
        for key, record in seal["records"].items():
            check(sha(ROOT / "runs" / folder / key / "record.json") == record["record_sha256"], partition + ": bank record " + key)
        report["bank_record_hashes"][partition] = len(seal["records"])
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
