"""CPU-only, resumable T-partition finite-bank generation. Never opens E files."""
import argparse
import concurrent.futures
import fcntl
import json
import multiprocessing
import time
from pathlib import Path

import numpy as np
import torch

from scripts.pilot_strength_null import reference_generator, diagnostics
from scripts.revision_manifest import load_manifest
from scripts.train_revision import TrainingCache
from src.revision import atomic_json, plv_features, sha256_file, stream_seed


def build_trial(task):
    vector, subject, index, trial_id, data_hash, plan, plan_hash, output = task
    path = Path(output)/f"s{subject:02d}"/f"t{index:04d}"
    path.mkdir(parents=True, exist_ok=True)
    record = path/'record.json'
    if record.exists():
        old=json.loads(record.read_text())
        if old['plan_sha256']!=plan_hash or old['training_data_sha256']!=data_hash:
            raise RuntimeError('incompatible existing bank')
        if sha256_file(path/'bank.npz')!=old['bank_sha256']:
            raise RuntimeError('bank checksum mismatch')
        return old
    start=time.monotonic(); u=np.triu_indices(128,1)
    a=np.zeros((128,128));a[u]=vector.astype(float);a+=a.T
    function=reference_generator();bank=[];metrics=[]
    for member in range(plan['bank_size']):
        seed=stream_seed(plan['protocol'],subject,int(trial_id),member)
        r=np.zeros_like(a);r[u]=np.random.default_rng(seed).permutation(a[u]);r+=r.T
        b,energy=function(a,R=r,seed=seed,nstage=plan['stages'],niter=plan['iterations'],temp=plan['temperature'],frac=plan['cooling'])
        d=diagnostics(a,b);d['seed']=seed
        d['energy_verified']=bool(np.isclose(energy,d['strength_sse'],rtol=1e-6,atol=1e-8))
        g=plan['gates']
        d['passed']=bool(d['exact_multiset'] and d['symmetric'] and d['zero_diagonal'] and d['energy_verified'] and d['relative_strength_rms']<=g['relative_rms_max'] and d['relative_strength_max']<=g['relative_max_max'] and d['changed_edge_fraction']>=g['changed_min'] and d['displacement_over_uniform_expectation']>=g['displacement_min'])
        bank.append(b[u].astype(np.float32));metrics.append(d)
    bank=np.stack(bank)
    scale=2*len(vector)*vector.astype(float).var()
    diversity=min(float(np.sum((bank[i].astype(float)-bank[j].astype(float))**2)/scale) if scale else 0.0 for i in range(len(bank)) for j in range(i))
    passed=all(x['passed'] for x in metrics) and diversity>=plan['gates']['diversity_min']
    temporary=path/'bank.npz.tmp'
    with temporary.open('wb') as f:np.savez_compressed(f,original=vector,bank=bank,trial_id=trial_id)
    temporary.replace(path/'bank.npz')
    result={'plan_sha256':plan_hash,'subject':subject,'trial_index':index,'trial_id':int(trial_id),'training_data_sha256':data_hash,'passed':bool(passed),'minimum_pairwise_diversity':diversity,'metrics':metrics,'elapsed_seconds':time.monotonic()-start,'bank_sha256':sha256_file(path/'bank.npz')}
    atomic_json(record,result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=24);args=p.parse_args()
    plan=json.loads(args.plan.read_text());plan_hash=sha256_file(args.plan)
    for name,digest in plan['source_sha256'].items():
        if sha256_file(Path(name))!=digest:raise RuntimeError('generation source changed: '+name)
    load_manifest(Path('configs/dsp_revision_20260919.json'))
    args.out.mkdir(parents=True,exist_ok=True)
    lock=(args.out/'generator.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    saved=args.out/'plan.json'
    if saved.exists() and json.loads(saved.read_text())!=plan:raise RuntimeError('output plan mismatch')
    atomic_json(saved,plan)
    torch.set_num_threads(4)
    cache=TrainingCache(args.data_root,torch.device('cpu'))
    completed=failed=0;start=time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for subject in plan['subjects']:
            atomic_json(args.out/'progress.json',{'stage':'extracting_T_features_cpu','subject':subject,'completed_trials':completed,'failed_trials':failed})
            cache.load('schirrmeister',subject)
            vectors=torch.cat([plv_features(x) for x in cache.x.split(8)]).numpy()
            tasks=[(v,subject,i,int(cache.ids[i]),cache.data_hash,plan,plan_hash,str(args.out)) for i,v in enumerate(vectors)]
            for row in pool.map(build_trial,tasks,chunksize=1):
                completed+=1;failed+=not row['passed']
                atomic_json(args.out/'progress.json',{'stage':'generating_T_bank_cpu','subject':subject,'completed_trials':completed,'failed_trials':failed,'total_trials':plan['expected_T_trials'],'updated_unix_time':time.time(),'elapsed_seconds':time.monotonic()-start})
                print(json.dumps({'completed_trials':completed,'subject':subject,'trial':row['trial_index'],'passed':row['passed'],'failed_trials':failed}),flush=True)
    if completed!=plan['expected_T_trials']:raise RuntimeError('unexpected T trial count')
    atomic_json(args.out/'completion.json',{'complete':True,'all_quality_gates_pass':failed==0,'trials':completed,'failed_trials':failed,'plan_sha256':plan_hash,'elapsed_seconds':time.monotonic()-start})
    atomic_json(args.out/'progress.json',{'stage':'finished' if failed==0 else 'finished_with_quality_failures','completed_trials':completed,'failed_trials':failed,'updated_unix_time':time.time()})


if __name__=='__main__':main()
