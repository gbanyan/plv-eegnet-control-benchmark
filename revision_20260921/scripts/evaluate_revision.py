"""Evaluate only fully sealed, immutable revision training batches."""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.revision_manifest import jobs_for_batch, load_manifest, manifest_hash
from scripts.train_revision import TrainingCache, feature_mask
from src.datasets.bci2a import BCI2aNPZ, trial_id_offset
from src.revision import FeatureControl, RevisionModel, atomic_json, plv_features, sha256_file, stream_seed
from src.train_all_subjects import DATASET_INFO


def checked_probabilities(logits):
    probability = logits.softmax(1)
    if not torch.isfinite(probability).all().item():
        raise RuntimeError("nonfinite evaluation probabilities")
    return probability


def validate_seal(manifest, batch, output):
    path = output / "seals" / f"{batch}.json"
    seal = json.loads(path.read_text())
    if seal["manifest_sha256"] != manifest_hash(manifest):
        raise RuntimeError("seal belongs to another manifest")
    jobs = jobs_for_batch(manifest, batch)
    if set(seal["records"]) != {j["id"] for j in jobs}:
        raise RuntimeError("seal does not cover the complete declared batch")
    for job in jobs:
        directory = output / "jobs" / job["id"]
        record = directory / "train.json"
        if sha256_file(record) != seal["records"][job["id"]]:
            raise RuntimeError("training record changed after seal")
        data = json.loads(record.read_text())
        for item in data["checkpoints"].values():
            if sha256_file(directory / item["file"]) != item["sha256"]:
                raise RuntimeError("selected checkpoint changed after seal")
    return seal, sha256_file(path)


class EvaluationCache:
    def __init__(self, root, device):
        self.root, self.device = root, device
        self.training = TrainingCache(root, device)
        self.key, self.values = None, {}

    def load(self, dataset, subject):
        if self.key == (dataset, subject):
            return
        self.values.clear()
        if self.key is not None:
            del self.x, self.y
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        self.training.load(dataset, subject)
        spec = DATASET_INFO[dataset]
        path = self.root / dataset / spec["filename_fmt"].format(sid=subject, suffix="E")
        self.e_hash = sha256_file(path)
        data = BCI2aNPZ(path)
        if data.X.shape[1:] != (spec["n_channels"], spec["n_samples"]):
            raise ValueError("invalid evaluation shape")
        self.x, self.y = data.X.to(self.device), data.y.to(self.device)
        self.ids = torch.arange(len(data)) + trial_id_offset(subject, evaluation=True)
        self.key = (dataset, subject)

    def features(self, condition):
        key = (tuple(condition["band"]), condition.get("edge", False))
        if key not in self.values:
            self.values[key] = torch.cat([plv_features(x, key[0], edge=key[1])
                                          for x in self.x.split(64)])
        full = self.values[key]
        mask = feature_mask(condition, self.key[0], full.shape[1])
        return full[:, mask.to(self.device)]


@torch.no_grad()
def evaluate_one(job, manifest, cache, output, seal_hash):
    directory = output / "jobs" / job["id"]
    destination = directory / "evaluation.json"
    protocol_hash = manifest_hash(manifest)
    if destination.exists():
        old = json.loads(destination.read_text())
        if old["manifest_sha256"] != protocol_hash or old["seal_sha256"] != seal_hash:
            raise RuntimeError("evaluation provenance mismatch")
        if sha256_file(directory / "predictions.npz") != old["predictions_sha256"]:
            raise RuntimeError("prediction archive changed")
        return old
    record = json.loads((directory / "train.json").read_text())
    cache.load(job["dataset"], job["subject"])
    if cache.training.data_hash != record["training_data_sha256"]:
        raise RuntimeError("training cache changed since fitting")
    condition = manifest["conditions"][job["condition"]]
    values = cache.features(condition)
    training_values, _ = cache.training.features(condition)
    train_ids = torch.tensor(record["training_indices"], dtype=torch.long)
    control = FeatureControl(values, cache.ids, torch.empty(0, dtype=torch.long), condition,
                             record["seeds"], donor_values=training_values[train_ids],
                             donor_ids=cache.training.ids[train_ids])
    model = RevisionModel(DATASET_INFO[job["dataset"]], condition, record["seeds"]["init"]).to(cache.device)
    arrays = {"labels": cache.y.cpu().numpy(), "trial_ids": cache.ids.numpy()}
    metrics = {}
    start = time.monotonic()
    for rule, artifact in record["checkpoints"].items():
        checkpoint = torch.load(directory / artifact["file"], map_location=cache.device, weights_only=True)
        if checkpoint["manifest_sha256"] != protocol_hash:
            raise RuntimeError("checkpoint manifest differs")
        model.load_state_dict(checkpoint["model"])
        model.eval()
        embedding = torch.cat([model._eegnet_features(x) for x in cache.x.split(64)])
        probability = checked_probabilities(model.from_embedding(embedding, control.fixed))
        arrays[f"{rule}_fixed_probabilities"] = probability.cpu().numpy()
        metrics[rule] = {"epoch": artifact["epoch"],
                         "fixed_accuracy": int((probability.argmax(1) == cache.y).sum()) / len(cache.y)}
        if rule == "primary" and job["condition"] in manifest["evaluation"]["mc_conditions"]:
            draws = []
            for draw in range(manifest["evaluation"]["mc_draws"]):
                generator = torch.Generator().manual_seed(stream_seed(
                    "revision-v1-inference", job["dataset"], job["subject"], job["replicate"], draw))
                permutations = torch.stack([torch.randperm(values.shape[1], generator=generator)
                                            for _ in range(len(values))]).to(cache.device)
                permuted = values.gather(1, permutations)
                draws.append(checked_probabilities(model.from_embedding(embedding, permuted)).cpu().numpy())
            draws = np.stack(draws)
            accuracy = (draws.argmax(2) == arrays["labels"][None, :]).mean(1)
            arrays["primary_mc_probabilities"] = draws
            metrics[rule].update({"mc_draw_accuracies": accuracy.tolist(),
                                 "mc_mean_accuracy": float(accuracy.mean()),
                                 "mc_sd_accuracy": float(accuracy.std(ddof=1)),
                                 "mc_standard_error": float(accuracy.std(ddof=1) / np.sqrt(len(accuracy))),
                                 "probability_ensemble_accuracy": float((draws.mean(0).argmax(1) == arrays["labels"]).mean())})
    prediction_path = directory / "predictions.npz"
    temporary = prediction_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(prediction_path)
    result = {"job": job, "manifest_sha256": protocol_hash, "seal_sha256": seal_hash,
              "training_record_sha256": sha256_file(directory / "train.json"),
              "evaluation_data_sha256": cache.e_hash,
              "predictions_sha256": sha256_file(prediction_path), "metrics": metrics,
              "evaluation_trials": len(cache.y), "elapsed_seconds": time.monotonic() - start}
    atomic_json(destination, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory-fraction", type=float, required=True)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    _, seal_hash = validate_seal(manifest, args.batch, args.output)
    # Validation of every seal/checkpoint precedes the first E-file access.
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    cache = EvaluationCache(args.data_root, torch.device("cuda"))
    jobs = jobs_for_batch(manifest, args.batch)
    for number, job in enumerate(jobs):
        evaluate_one(job, manifest, cache, args.output, seal_hash)
        atomic_json(args.output / "progress.json", {"stage": "evaluation", "batch": args.batch,
                    "job": job["id"], "completed_in_batch": number + 1, "batch_jobs": len(jobs),
                    "updated_unix_time": time.time()})
        print(json.dumps({"evaluated": job["id"], "completed": number + 1, "total": len(jobs)}), flush=True)
    atomic_json(args.output / "completed_batches" / f"{args.batch}.json",
                {"batch": args.batch, "manifest_sha256": manifest_hash(manifest),
                 "seal_sha256": seal_hash, "jobs": len(jobs), "completed_unix_time": time.time()})


if __name__ == "__main__":
    main()
