"""Audit a candidate refinement on archived T-only v2 banks, never overwrite inputs."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from scripts.experimental.strength_targeted_refinement import refine
from scripts.pilot_strength_null import diagnostics


def quality(original,bank,gates):
    n=int((1+np.sqrt(1+8*len(original)))/2);u=np.triu_indices(n,1)
    def matrix(w):
        a=np.zeros((n,n),dtype=np.float64);a[u]=w;return a+a.T
    rows=[diagnostics(matrix(original),matrix(w)) for w in bank]
    scale=2*len(original)*np.var(original.astype(float))
    diversity=min(float(np.sum((bank[i].astype(float)-bank[j])**2)/scale) for i in range(len(bank)) for j in range(i))
    passed=all(r['exact_multiset'] and r['symmetric'] and r['zero_diagonal'] and r['relative_strength_rms']<=gates['relative_rms_max'] and r['relative_strength_max']<=gates['relative_max_max'] and r['changed_edge_fraction']>=gates['changed_min'] and r['displacement_over_uniform_expectation']>=gates['displacement_min'] for r in rows) and diversity>=gates['diversity_min']
    return dict(passed=bool(passed),metrics=rows,minimum_pairwise_diversity=diversity)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=512);a=p.parse_args()
    if a.out.exists():raise RuntimeError('pilot output must be new')
    a.out.mkdir(parents=True)
    gates=json.loads(Path('configs/strength_bank_T_gpu_v2r2_20260921.json').read_text())['gates'];rows=[];start=time.monotonic()
    for path in sorted(a.root.glob('**/bank.npz')):
        record=json.loads(path.with_name('record.json').read_text())
        with np.load(path) as f:original=f['original'];before=f['bank']
        refined=[];reports=[]
        for initial,key in zip(before,record['surrogate_keys']):
            w,report=refine(original,initial,tuple(key),maximum_steps=a.steps);refined.append(w.astype(np.float32));reports.append(report)
        bank=np.stack(refined);qa=quality(original,bank,gates)
        target=a.out/f"s{record['subject']:02d}_t{record['trial_index']:04d}.npz"
        np.savez_compressed(target,original=original,bank=bank)
        row=dict(subject=record['subject'],trial=record['trial_index'],file=target.name,parent_passed=record['passed'],reports=reports,**qa);rows.append(row)
        (a.out/'results.json').write_text(json.dumps(dict(complete=False,banks=rows),indent=2)+'\n')
        print(f"{len(rows)} {target.name} passed={qa['passed']} max={max(m['relative_strength_max'] for m in qa['metrics']):.6f} diversity={qa['minimum_pairwise_diversity']:.6f}",flush=True)
    (a.out/'results.json').write_text(json.dumps(dict(complete=True,maximum_steps=a.steps,elapsed_seconds=time.monotonic()-start,all_pass=all(r['passed'] for r in rows),banks=rows),indent=2)+'\n')


if __name__=='__main__':main()
