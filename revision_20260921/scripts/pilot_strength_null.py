"""Training-only dense weighted-null feasibility; independent of frozen runs."""
import argparse
import ast
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.utils.validation import check_random_state

from scripts.train_revision import TrainingCache, seeds_for, _stratified_split
from src.revision import atomic_json, plv_features, stream_seed


def reference_generator():
    folder = Path(__file__).parent / "vendor/netneurotools"
    source = (folder / "randomize.py").read_bytes()
    provenance = json.loads((folder / "PROVENANCE.json").read_text())
    assert hashlib.sha256(source).hexdigest() == provenance["sha256"]
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                and n.name == "strength_preserving_rand_sa")
    scope = {"np": np, "check_random_state": check_random_state,
             "tqdm": lambda x, **kwargs: x}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(folder / "randomize.py"), "exec"), scope)
    return scope["strength_preserving_rand_sa"]


def diagnostics(a, b):
    u = np.triu_indices(len(a), 1)
    weights, surrogate = a[u], b[u]
    strength = a.sum(1)
    error = b.sum(1) - strength
    relative = error / np.maximum(strength, 1e-12)
    # 2*P*variance is the expected squared displacement of a uniform shuffle.
    scale = 2 * len(weights) * weights.var()
    return {"relative_strength_rms": float(np.sqrt(np.mean(relative**2))),
            "relative_strength_max": float(np.abs(relative).max()),
            "strength_sse": float(np.sum(error**2)),
            "exact_multiset": bool(np.array_equal(np.sort(weights), np.sort(surrogate))),
            "symmetric": bool(np.array_equal(b, b.T)),
            "zero_diagonal": bool(np.all(np.diag(b) == 0)),
            "changed_edge_fraction": float(np.mean(weights != surrogate)),
            "displacement_over_uniform_expectation": float(np.sum((weights-surrogate)**2) / scale)
            if scale else 0.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.25)
    generate = reference_generator()
    cache = TrainingCache(args.data_root, torch.device("cuda"))
    args.out.mkdir(parents=True, exist_ok=True)
    plan = {"scope": "training-only feasibility; no classifier outcomes or E loading",
            "datasets": ["schirrmeister", "lee2019"], "subjects": [1, 4],
            "trial_selection": "first index of inner-training split for named replicate 1001",
            "stages": [30, 100], "iterations_per_stage": 10000, "temperature": 1000,
            "cooling": .5, "seeds": [1, 2],
            "provisional_gates": {"relative_strength_rms_max": .01,
                "relative_strength_max_max": .05, "changed_edge_fraction_min": .95,
                "displacement_over_uniform_expectation_min": .25,
                "pairwise_diversity_over_uniform_expectation_min": .25},
            "note": "Engineering feasibility tolerances, not biological standards; small pilot cannot validate all trials."}
    atomic_json(args.out / "plan.json", plan)
    rows = []
    for dataset in plan["datasets"]:
        for subject in plan["subjects"]:
            cache.load(dataset, subject)
            split_seed = seeds_for({"dataset": dataset, "subject": subject, "replicate": 1001})["split"]
            train, validation = _stratified_split(cache.y.cpu().numpy(), .2, split_seed)
            index = train[0]
            assert index not in validation
            values = plv_features(cache.x[index:index+1])[0].cpu().numpy().astype(float)
            n = cache.x.shape[1]
            u = np.triu_indices(n, 1)
            a = np.zeros((n,n)); a[u] = values; a += a.T
            for stages in plan["stages"]:
                previous = None
                for repeat in plan["seeds"]:
                    seed = stream_seed("strength-pilot", dataset, subject, stages, repeat)
                    r = np.zeros_like(a); r[u] = np.random.default_rng(seed).permutation(values); r += r.T
                    start = time.monotonic()
                    b, energy = generate(a, R=r, seed=seed, nstage=stages, niter=10000)
                    elapsed = time.monotonic()-start
                    row = {"dataset": dataset, "subject": subject, "training_trial_index": int(index),
                           "training_data_sha256": cache.data_hash, "stages": stages, "seed": seed,
                           "seconds": elapsed, "reported_energy": float(energy), **diagnostics(a,b)}
                    row["energy_verified"] = bool(np.isclose(energy, row["strength_sse"], rtol=1e-6, atol=1e-8))
                    row["unconstrained"] = diagnostics(a,r)
                    row["pairwise_diversity"] = None if previous is None else float(np.sum((b[u]-previous[u])**2)/(2*len(values)*values.var()))
                    row["passes_preservation_and_displacement"] = bool(row["relative_strength_rms"]<=.01 and row["relative_strength_max"]<=.05 and row["changed_edge_fraction"]>=.95 and row["displacement_over_uniform_expectation"]>=.25 and row["exact_multiset"] and row["symmetric"] and row["zero_diagonal"] and row["energy_verified"])
                    np.savez_compressed(args.out / f"{dataset}_s{subject:02d}_stages{stages}_seed{repeat}.npz", original=a, surrogate=b, initial=r)
                    previous = b
                    rows.append(row)
                    atomic_json(args.out / "results.json", {"complete": False, "cases": rows})
                    print(json.dumps(row), flush=True)
    atomic_json(args.out / "results.json", {"complete": True, "cases": rows})


if __name__ == "__main__":
    main()
