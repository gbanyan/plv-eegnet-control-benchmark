"""Operational subject allocation; delegates unchanged bank generation."""
import argparse
import concurrent.futures
import fcntl
import json
import multiprocessing
import time
from pathlib import Path
import numpy as np
import torch
from scripts.generate_strength_bank import build_trial
from scripts.revision_manifest import load_manifest
from scripts.train_revision import TrainingCache
from src.revision import atomic_json, plv_features, sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--allocation',type=Path,required=True);p.add_argument('--host',required=True);p.add_argument('--data-root',type=Path);p.add_argument('--features-root',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=24);a=p.parse_args()
    plan=json.loads(a.plan.read_text());ph=sha256_file(a.plan)
    for name,h in plan['source_sha256'].items():
        if sha256_file(Path(name))!=h:raise RuntimeError('frozen generator changed')
    load_manifest(Path('configs/dsp_revision_20260919.json'))
    allocation=json.loads(a.allocation.read_text());assert allocation['plan_sha256']==ph
    subjects=allocation['subjects_by_host'][a.host]
    all_subjects=[s for group in allocation['subjects_by_host'].values() for s in group]
    assert sorted(all_subjects)==plan['subjects'] and len(all_subjects)==len(set(all_subjects))
    expected=sum(allocation['trial_counts'][str(s)] for s in subjects)
    a.out.mkdir(parents=True,exist_ok=True)
    lock=(a.out/'generator.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    saved=a.out/'plan.json'
    if saved.exists() and json.loads(saved.read_text())!=plan:raise RuntimeError('output plan mismatch')
    atomic_json(saved,plan);atomic_json(a.out/'allocation.json',allocation)
    torch.set_num_threads(4)
    cache=TrainingCache(a.data_root,torch.device('cpu')) if not a.features_root else None
    completed=failed=0;start=time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for subject in subjects:
            atomic_json(a.out/'progress.json',{'stage':'loading_T_features_cpu','subject':subject,'host':a.host,'completed_trials':completed,'failed_trials':failed,'assigned_trials':expected})
            if a.features_root:
                file=a.features_root/f's{subject:02d}.npz';meta=json.loads(file.with_suffix('.json').read_text())
                assert meta['subject']==subject and sha256_file(file)==meta['feature_file_sha256']
                with np.load(file) as f:vectors=f['values'];ids=f['ids']
                data_hash=meta['training_data_sha256']
            else:
                cache.load('schirrmeister',subject);vectors=torch.cat([plv_features(x) for x in cache.x.split(8)]).numpy();ids=cache.ids.numpy();data_hash=cache.data_hash
            assert len(vectors)==allocation['trial_counts'][str(subject)] and vectors.shape[1]==8128
            tasks=[(v,subject,i,int(ids[i]),data_hash,plan,ph,str(a.out)) for i,v in enumerate(vectors)]
            for r in pool.map(build_trial,tasks,chunksize=1):
                completed+=1;failed+=not r['passed']
                atomic_json(a.out/'progress.json',{'stage':'generating_T_bank_cpu','host':a.host,'subject':subject,'completed_trials':completed,'failed_trials':failed,'assigned_trials':expected,'total_trials':plan['expected_T_trials'],'updated_unix_time':time.time(),'elapsed_seconds':time.monotonic()-start})
                print(json.dumps({'completed_trials':completed,'subject':subject,'trial':r['trial_index'],'passed':r['passed'],'failed_trials':failed}),flush=True)
    assert completed==expected
    final={'stage':'finished' if failed==0 else 'finished_with_quality_failures','host':a.host,'subjects':subjects,'complete_assigned_subjects':True,'all_quality_gates_pass':failed==0,'completed_trials':completed,'failed_trials':failed,'assigned_trials':expected,'plan_sha256':ph,'updated_unix_time':time.time()}
    atomic_json(a.out/'completion.json',final);atomic_json(a.out/'progress.json',final)


if __name__=='__main__':main()
