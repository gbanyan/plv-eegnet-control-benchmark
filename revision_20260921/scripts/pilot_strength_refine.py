"""CPU-only longer-annealing checks on the already selected T-only matrices."""
import argparse
import concurrent.futures
import json
import multiprocessing
import time
from pathlib import Path
import numpy as np
from scripts.pilot_strength_null import reference_generator, diagnostics
from src.revision import atomic_json


def one(task):
    folder,row,stages,out=task
    with np.load(Path(folder)/row['file']) as f:a=f['original']
    seed=row['seed'];u=np.triu_indices(len(a),1);r=np.zeros_like(a)
    r[u]=np.random.default_rng(seed).permutation(a[u]);r+=r.T
    start=time.monotonic()
    b,e=reference_generator()(a,R=r,seed=seed,nstage=stages,niter=10000)
    result={k:row[k] for k in ['subject','trial','repeat','seed','training_data_sha256','file']}
    result.update(stages=stages,seconds=time.monotonic()-start,**diagnostics(a,b))
    result['energy_verified']=bool(np.isclose(e,result['strength_sse'],rtol=1e-6,atol=1e-8))
    result['passes']=bool(result['relative_strength_rms']<=.01 and result['relative_strength_max']<=.05 and result['changed_edge_fraction']>=.95 and result['displacement_over_uniform_expectation']>=.25 and result['exact_multiset'] and result['symmetric'] and result['zero_diagonal'] and result['energy_verified'])
    np.savez_compressed(Path(out)/row['file'],original=a,surrogate=b)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--stages',type=int,required=True);p.add_argument('--workers',type=int,required=True);p.add_argument('--only-failed',action='store_true');a=p.parse_args()
    original=json.loads((a.input/'results.json').read_text());assert original['complete']
    selected=[r for r in original['cases'] if not a.only_failed or not r['passes']]
    a.out.mkdir(parents=True,exist_ok=True)
    atomic_json(a.out/'plan.json',{'input':str(a.input),'nstage':a.stages,'niter':10000,'workers':a.workers,'only_failed':a.only_failed,'cases':len(selected),'note':'Training-only method refinement after 30-stage feasibility failures; unchanged quality gates. No classifier or E-set access.'})
    rows=[];start=time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for row in pool.map(one,[(str(a.input),r,a.stages,str(a.out)) for r in selected]):
            rows.append(row);atomic_json(a.out/'results.json',{'complete':False,'cases':rows})
            print(f"complete {len(rows)}/{len(selected)} passed={row['passes']}",flush=True)
    diversity=[];pairs={}
    for row in rows:pairs.setdefault((row['subject'],row['trial']),[]).append(row)
    for group in pairs.values():
        if len(group)!=2:continue
        with np.load(a.out/group[0]['file']) as x,np.load(a.out/group[1]['file']) as y:
            u=np.triu_indices(len(x['original']),1);v=x['original'][u]
            diversity.append(float(np.sum((x['surrogate'][u]-y['surrogate'][u])**2)/(2*len(v)*v.var())))
    atomic_json(a.out/'results.json',{'complete':True,'cases':rows,'pairwise_diversity':diversity,'wall_seconds':time.monotonic()-start,'all_gates_pass':bool(all(r['passes'] for r in rows) and diversity and min(diversity)>=.25)})


if __name__=='__main__':main()
