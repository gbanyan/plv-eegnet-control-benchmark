"""Apply the same locked T-only v3 refinement rule to every archived v2 member."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import multiprocessing
import os
import platform
import time
from pathlib import Path
import numpy as np
from scripts.experimental.strength_targeted_refinement import refine
from scripts.pilot_strength_targeted import quality


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');t.replace(p)


def one(task):
    parent,out,relative,entry,plan,ph=task
    src=Path(parent)/relative;dst=Path(out)/relative
    if sha(src/'record.json')!=entry['record_sha256'] or sha(src/'bank.npz')!=entry['bank_sha256']:
        raise RuntimeError('frozen parent checksum mismatch '+relative)
    record=json.loads((src/'record.json').read_text())
    if record['plan_sha256']!=plan['parent_plan_sha256'] or record['partition']!='T':raise RuntimeError('foreign parent')
    if (dst/'record.json').exists():
        saved=json.loads((dst/'record.json').read_text())
        if saved['plan_sha256']!=ph or saved['parent_record_sha256']!=entry['record_sha256'] or sha(dst/'bank.npz')!=saved['bank_sha256']:raise RuntimeError('incompatible resume')
        return relative,saved
    began=time.monotonic()
    with np.load(src/'bank.npz') as f:original=f['original'];before=f['bank'];trial_id=int(f['trial_id'])
    if before.shape!=(plan['bank_size'],8128) or original.dtype!=np.float32 or trial_id!=record['trial_id']:raise RuntimeError('unexpected parent layout')
    bank=[];reports=[]
    for initial,key in zip(before,record['surrogate_keys']):
        w,r=refine(original,initial,tuple(key),**plan['refinement'])
        b=w.astype(np.float32)
        if not np.array_equal(b.astype(float),w):raise RuntimeError('weight rounding')
        bank.append(b);reports.append(r)
    bank=np.stack(bank);qa=quality(original,bank,plan['gates'])
    dst.mkdir(parents=True,exist_ok=True)
    with (dst/'bank.npz.tmp').open('wb') as f:np.savez_compressed(f,original=original,bank=bank,trial_id=trial_id)
    (dst/'bank.npz.tmp').replace(dst/'bank.npz')
    saved=dict(protocol=plan['protocol'],plan_sha256=ph,partition='T',subject=record['subject'],trial_index=record['trial_index'],trial_id=trial_id,
               parent_record_sha256=entry['record_sha256'],parent_bank_sha256=entry['bank_sha256'],parent_passed=record['passed'],
               training_data_sha256=record['training_data_sha256'],feature_file_sha256=record['feature_file_sha256'],
               surrogate_keys=record['surrogate_keys'],refinement_reports=reports,bank_sha256=sha(dst/'bank.npz'),
               host=platform.node(),numpy=np.__version__,seconds=time.monotonic()-began,E_set_accessed=False,**qa)
    atomic(dst/'record.json',saved);return relative,saved


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--parent-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--host',required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    plan=json.loads(a.plan.read_text());ph=sha(a.plan)
    for path,digest in plan['source_sha256'].items():
        if sha(path)!=digest:raise RuntimeError('frozen source mismatch '+path)
    if sha(plan['parent_index'])!=plan['parent_index_sha256']:raise RuntimeError('parent index mismatch')
    index=json.loads(Path(plan['parent_index']).read_text())[a.host]
    if len(index)!=plan['expected_by_host'][a.host]:raise RuntimeError('missing parent grid')
    a.out.mkdir(parents=True,exist_ok=True);lock=(a.out/'generator.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (a.out/'plan.json').exists() and json.loads((a.out/'plan.json').read_text())!=plan:raise RuntimeError('output plan mismatch')
    atomic(a.out/'plan.json',plan);start=time.monotonic();done=failed=changed=0;worst=0.;diversity=float('inf')
    tasks=[(str(a.parent_root),str(a.out),k,v,plan,ph) for k,v in sorted(index.items())]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for relative,r in pool.map(one,tasks,chunksize=1):
            done+=1;failed+=not r['passed'];changed+=any(x['accepted'] for x in r['refinement_reports'])
            worst=max(worst,max(x['relative_strength_max'] for x in r['metrics']));diversity=min(diversity,r['minimum_pairwise_diversity'])
            progress=dict(stage='refining_T_v3',completed_trials=done,assigned_trials=len(tasks),failed_trials=failed,changed_banks=changed,elapsed_seconds=time.monotonic()-start,updated_unix_time=time.time())
            if done%20==0 or done==len(tasks):atomic(a.out/'progress.json',progress);print(json.dumps(progress),flush=True)
    completion=dict(progress,stage='finished' if failed==0 else 'finished_with_quality_failures',all_quality_gates_pass=failed==0,plan_sha256=ph,maximum_relative_strength_error=worst,minimum_diversity=diversity)
    atomic(a.out/'completion.json',completion);atomic(a.out/'progress.json',completion)
    if failed:raise SystemExit(2)


if __name__=='__main__':main()
