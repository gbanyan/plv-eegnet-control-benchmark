"""Train frozen revision batches using T files only; seal before evaluation."""
from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts.revision_manifest import jobs_for_batch, load_manifest, manifest_hash
from src.datasets.bci2a import BCI2aNPZ, trial_id_offset
from src.revision import (CheckpointSelector, FeatureControl, FrozenStandardization,
                          RevisionModel, atomic_json, plv_features, sha256_file,
                          stream_seed, tensor_hash)
from src.train_all_subjects import DATASET_INFO, _stratified_split


def state_hash(state):
    import hashlib
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def seeds_for(job):
    return {name: stream_seed("revision-v1", job["dataset"], job["subject"], job["replicate"], name)
            for name in ("init", "order", "dropout", "fixed", "permutation", "refresh",
                         "donor", "donor_fixed", "global", "split")}


def feature_mask(condition, dataset, full_size):
    if "dimension" not in condition:
        return torch.arange(full_size)
    order = torch.randperm(full_size, generator=torch.Generator().manual_seed(
        stream_seed("revision-v1-mask", dataset, condition["mask"])))
    return order[:condition["dimension"]]


class TrainingCache:
    def __init__(self, data_root: Path, device):
        self.data_root, self.device = data_root, device
        self.key = None
        self.values = {}

    def load(self, dataset, subject):
        if self.key == (dataset, subject):
            return
        self.values.clear()
        if self.key is not None:
            del self.x, self.y, self.ids
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        spec = DATASET_INFO[dataset]
        # This training loader deliberately constructs T paths only.
        self.path = self.data_root / dataset / spec["filename_fmt"].format(sid=subject, suffix="T")
        self.data_hash = sha256_file(self.path)
        data = BCI2aNPZ(self.path)
        if data.X.shape[1:] != (spec["n_channels"], spec["n_samples"]):
            raise ValueError(f"incorrect training shape: {self.path}")
        if data.y.min() < 0 or data.y.max() >= spec["n_classes"]:
            raise ValueError("invalid training labels")
        self.x, self.y = data.X.to(self.device), data.y.to(self.device)
        self.ids = torch.arange(len(data)) + trial_id_offset(subject, evaluation=False)
        self.key = (dataset, subject)

    def features(self, condition):
        key = (tuple(condition["band"]), condition.get("edge", False))
        if key not in self.values:
            self.values[key] = torch.cat([
                plv_features(x, key[0], edge=key[1]) for x in self.x.split(64)
            ])
        full = self.values[key]
        mask = feature_mask(condition, self.key[0], full.shape[1])
        return full[:, mask.to(self.device)], mask


def batches(indices, size):
    parts = list(indices.split(size))
    if len(parts) > 1 and len(parts[-1]) == 1:
        parts[-2] = torch.cat((parts[-2], parts[-1]))
        parts.pop()
    return parts


def train_one(job, manifest, cache, output: Path, progress=None):
    protocol_hash = manifest_hash(manifest)
    directory = output / "jobs" / job["id"]
    record_path = directory / "train.json"
    if record_path.exists():
        old = json.loads(record_path.read_text())
        if old["manifest_sha256"] != protocol_hash:
            raise RuntimeError("existing result belongs to a different manifest")
        for item in old["checkpoints"].values():
            if sha256_file(directory / item["file"]) != item["sha256"]:
                raise RuntimeError("completed checkpoint hash mismatch")
        return old
    directory.mkdir(parents=True, exist_ok=True)
    cache.load(job["dataset"], job["subject"])
    settings = manifest["training"]
    condition = manifest["conditions"][job["condition"]]
    seeds = seeds_for(job)
    indices, validation = _stratified_split(cache.y.cpu().numpy(),
                                            settings["inner_validation_fraction"], seeds["split"])
    train_ids = torch.tensor(indices, dtype=torch.long)
    validation_ids = torch.tensor(validation, dtype=torch.long)
    values, mask = cache.features(condition)
    control = FeatureControl(values, cache.ids, train_ids, condition, seeds)
    model = RevisionModel(DATASET_INFO[job["dataset"]], condition, seeds["init"]).to(cache.device)
    if isinstance(model.plv_norm, FrozenStandardization):
        mean, variance = control.calibration()
        model.plv_norm.mean.copy_(mean)
        model.plv_norm.variance.copy_(variance)
    initial_backbone_hash = state_hash(model.eegnet.state_dict())
    initial_classifier_hash = state_hash(model.classifier.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    order_rng = torch.Generator().manual_seed(seeds["order"])
    selector = CheckpointSelector(minimum=settings["minimum_epochs"], patience=settings["patience"],
                                  min_delta=settings["min_delta"])
    fixed_budget = settings["fixed_budget"][job["dataset"]]
    snapshots, history = {}, []
    if cache.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    for epoch in range(1, settings["maximum_epochs"] + 1):
        # Dropout starts from its own per-epoch stream; feature/control generation
        # and CPU order generation never consume that stream.
        torch.manual_seed(stream_seed(seeds["dropout"], epoch))
        control.begin_epoch(epoch)
        order = train_ids[torch.randperm(len(train_ids), generator=order_rng)]
        model.train()
        train_loss = torch.zeros((), device=cache.device)
        for ids in batches(order, settings["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            logits = model(cache.x[ids], control.current[ids])
            loss = F.cross_entropy(logits, cache.y[ids])
            loss.backward()
            optimizer.step()
            train_loss += loss.detach() * len(ids)
        model.eval()
        val_loss = torch.zeros((), device=cache.device)
        val_correct = torch.zeros((), device=cache.device, dtype=torch.long)
        with torch.no_grad():
            for ids in validation_ids.split(settings["batch_size"]):
                logits = model(cache.x[ids], control.fixed[ids])
                val_loss += F.cross_entropy(logits, cache.y[ids], reduction="sum")
                val_correct += (logits.argmax(1) == cache.y[ids]).sum()
        tl, vl, va = float(train_loss / len(train_ids)), float(val_loss / len(validation_ids)), float(val_correct / len(validation_ids))
        if not np.isfinite(tl):
            raise RuntimeError("nonfinite training loss")
        primary, accuracy = selector.update(epoch, vl, va)
        history.append({"epoch": epoch, "train_loss": tl, "val_loss": vl, "val_accuracy": va})
        for name, save in (("primary", primary), ("accuracy", accuracy), ("fixed_budget", epoch == fixed_budget)):
            if save:
                snapshots[name] = {"epoch": epoch, "model": {k: v.detach().cpu().clone()
                                                          for k, v in model.state_dict().items()}}
        if progress and (epoch == 1 or epoch % 25 == 0):
            progress(job, epoch, selector.stop_epoch)
        if selector.finished(epoch, fixed_budget, settings["maximum_epochs"]):
            break
    if set(snapshots) != {"primary", "accuracy", "fixed_budget"}:
        raise RuntimeError("missing prespecified checkpoint")
    checkpoints = {}
    for name, snapshot in snapshots.items():
        path = directory / f"{name}.pt"
        temporary = path.with_suffix(".pt.tmp")
        torch.save({**snapshot, "job": job, "manifest_sha256": protocol_hash}, temporary)
        temporary.replace(path)
        checkpoints[name] = {"file": path.name, "epoch": snapshot["epoch"], "sha256": sha256_file(path)}
    atomic_json(directory / "curves.json", {"manifest_sha256": protocol_hash, "epochs": history})
    record = {
        "job": job, "manifest_sha256": protocol_hash, "condition": condition,
        "host": platform.node(), "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name() if cache.device.type == "cuda" else "cpu",
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "training_data_sha256": cache.data_hash,
        "training_indices": indices, "validation_indices": validation, "seeds": seeds,
        "feature_mask_sha256": tensor_hash(mask), "fixed_features_sha256": tensor_hash(control.fixed),
        "final_training_features_sha256": tensor_hash(control.current),
        "initial_backbone_sha256": initial_backbone_hash,
        "initial_classifier_sha256": initial_classifier_hash,
        "control_redraw_count": control.redraw_count,
        "epochs_completed": epoch, "virtual_stop_epoch": selector.stop_epoch,
        "hit_maximum_without_stop": selector.stop_epoch is None,
        "elapsed_seconds": time.monotonic() - start,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20 if cache.device.type == "cuda" else None,
        "checkpoints": checkpoints, "curves_sha256": sha256_file(directory / "curves.json"),
        "E_set_accessed": False,
    }
    # Completion marker written only after all artifacts have their checksums.
    atomic_json(record_path, record)
    return record


def seal_batch(manifest, batch, output):
    records = {}
    for job in jobs_for_batch(manifest, batch):
        directory = output / "jobs" / job["id"]
        path = directory / "train.json"
        record = json.loads(path.read_text())
        if record["manifest_sha256"] != manifest_hash(manifest) or record["E_set_accessed"]:
            raise RuntimeError("invalid training completion record")
        for artifact in record["checkpoints"].values():
            if sha256_file(directory / artifact["file"]) != artifact["sha256"]:
                raise RuntimeError("checkpoint changed before seal")
        records[job["id"]] = sha256_file(path)
    if not records:
        raise RuntimeError("empty batch cannot be sealed")
    seal = {"batch": batch, "manifest_sha256": manifest_hash(manifest), "records": records,
            "sealed_unix_time": time.time(), "status": "all_training_complete_before_E_access"}
    seal_path = output / "seals" / f"{batch}.json"
    if seal_path.exists():
        old = json.loads(seal_path.read_text())
        if old["records"] != records or old["manifest_sha256"] != seal["manifest_sha256"]:
            raise RuntimeError("attempt to change sealed batch")
        return old
    atomic_json(seal_path, seal)
    return seal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory-fraction", type=float, required=True)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    cache = TrainingCache(args.data_root, torch.device("cuda"))
    jobs = jobs_for_batch(manifest, args.batch)
    completed = 0

    def progress(job, epoch, stop):
        atomic_json(args.output / "progress.json", {"stage": "training", "batch": args.batch,
                    "job": job["id"], "completed_in_batch": completed, "batch_jobs": len(jobs),
                    "epoch": epoch, "virtual_stop_epoch": stop, "updated_unix_time": time.time()})

    for job in jobs:
        record = train_one(job, manifest, cache, args.output, progress)
        completed += 1
        print(json.dumps({"trained": job["id"], "epochs": record["epochs_completed"],
                          "seconds": record["elapsed_seconds"], "completed": completed,
                          "total": len(jobs)}), flush=True)
    seal_batch(manifest, args.batch, args.output)
    print(json.dumps({"sealed": args.batch, "jobs": len(jobs)}), flush=True)


if __name__ == "__main__":
    main()
