"""All-HGD-subject training-only confirmation of the 30-stage pilot setting."""
import argparse
import concurrent.futures
import multiprocessing
import time
from pathlib import Path

import numpy as np
import torch

from scripts.pilot_strength_null import reference_generator, diagnostics
from scripts.train_revision import TrainingCache, seeds_for, _stratified_split
from src.revision import atomic_json, plv_features, stream_seed


def generate_one(task):
    a, metadata, repeat, out = task
    seed = stream_seed("strength-expanded", metadata["subject"], metadata["trial"], repeat)
    u = np.triu_indices(len(a), 1)
    r = np.zeros_like(a); r[u] = np.random.default_rng(seed).permutation(a[u]); r += r.T
    start = time.monotonic()
    b, energy = reference_generator()(a, R=r, seed=seed, nstage=30, niter=10000)
    row = {**metadata, "repeat": repeat, "seed": seed, "seconds": time.monotonic()-start,
           **diagnostics(a,b)}
    row["energy_verified"] = bool(np.isclose(energy, row["strength_sse"], rtol=1e-6, atol=1e-8))
    row["passes"] = bool(row["relative_strength_rms"] <= .01 and row["relative_strength_max"] <= .05 and row["changed_edge_fraction"] >= .95 and row["displacement_over_uniform_expectation"] >= .25 and row["exact_multiset"] and row["symmetric"] and row["zero_diagonal"] and row["energy_verified"])
    name = f"s{metadata['subject']:02d}_t{metadata['trial']}_r{repeat}.npz"
    np.savez_compressed(Path(out)/name, original=a, surrogate=b)
    row["file"] = name
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out/"plan.json", {"stage": "training-only expanded feasibility",
        "subjects": list(range(1,15)), "trials_per_subject": 4, "surrogates_per_trial": 2,
        "trial_selection": "four evenly spaced indices in sorted inner-training split, replicate 1001",
        "nstage": 30, "niter": 10000, "temp": 1000, "frac": .5,
        "gates": "same as initial pilot: relative RMS <=1%, max<=5%, changed edges>=95%, normalized displacement and pairwise diversity>=.25",
        "selection_note": "30 stages selected after initial small training-only pilot; no classification outcomes used"})
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.25)
    cache = TrainingCache(args.data_root, torch.device("cuda"))
    tasks = []
    for subject in range(1,15):
        cache.load("schirrmeister",subject)
        seed=seeds_for({"dataset":"schirrmeister","subject":subject,"replicate":1001})["split"]
        train,_ = _stratified_split(cache.y.cpu().numpy(),.2,seed)
        ordered=sorted(train)
        ids=[ordered[i] for i in np.linspace(0,len(ordered)-1,4,dtype=int)]
        values=plv_features(cache.x[ids]).cpu().numpy().astype(float)
        u=np.triu_indices(128,1)
        for index,vector in zip(ids,values):
            a=np.zeros((128,128));a[u]=vector;a+=a.T
            for repeat in (1,2):
                tasks.append((a,{"subject":subject,"trial":int(index),"training_data_sha256":cache.data_hash},repeat,str(args.out)))
    del cache
    torch.cuda.empty_cache()
    rows=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context("spawn")) as pool:
        for row in pool.map(generate_one,tasks):
            rows.append(row)
            atomic_json(args.out/"results.json",{"complete":False,"cases":rows})
            print(f"completed {len(rows)}/{len(tasks)}; passed={row['passes']}",flush=True)
    diversity=[]
    for first,second in zip(rows[::2],rows[1::2]):
        assert (first['subject'],first['trial'])==(second['subject'],second['trial'])
        with np.load(args.out/first['file']) as f, np.load(args.out/second['file']) as s:
            a=f['original'];u=np.triu_indices(len(a),1)
            value=float(np.sum((f['surrogate'][u]-s['surrogate'][u])**2)/(2*len(u[0])*a[u].var()))
        diversity.append(value)
    atomic_json(args.out/"results.json",{"complete":True,"cases":rows,"pairwise_diversity":diversity,
        "all_gates_pass": bool(all(r['passes'] for r in rows) and min(diversity)>=.25)})


if __name__ == "__main__":
    main()
