"""T-only GPU v2 banks; fixed two-phase generation, audited QA, fail closed."""
import argparse
import fcntl
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
import triton
from scripts.experimental.strength_gpu_backend import generate, matrices
from scripts.pilot_strength_null import diagnostics
from src.revision import atomic_json, sha256_file


def verify_plan(path):
    plan=json.loads(path.read_text())
    for name,digest in plan['source_sha256'].items():
        if sha256_file(Path(name))!=digest:
            raise RuntimeError('frozen GPU source mismatch: '+name)
    return plan,sha256_file(path)


def quality(original, bank, gates):
    a=matrices(original[None])[0]
    rows=[]
    for b in matrices(bank):
        d=diagnostics(a,b)
        d['passed']=bool(d['exact_multiset'] and d['symmetric'] and d['zero_diagonal']
            and d['relative_strength_rms']<=gates['relative_rms_max']
            and d['relative_strength_max']<=gates['relative_max_max']
            and d['changed_edge_fraction']>=gates['changed_min']
            and d['displacement_over_uniform_expectation']>=gates['displacement_min'])
        rows.append(d)
    scale=2*len(original)*original.astype(float).var()
    diversity=min(float(np.sum((x.astype(float)-y.astype(float))**2)/scale)
                  if scale else 0. for i,x in enumerate(bank) for y in bank[:i])
    return dict(passed=all(x['passed'] for x in rows) and diversity>=gates['diversity_min'],
                metrics=rows,minimum_pairwise_diversity=diversity)


def existing(directory, plan_hash, data_hash, original, trial_id):
    file=directory/'record.json'
    if not file.exists():return None
    d=json.loads(file.read_text())
    if (d['plan_sha256']!=plan_hash or d['training_data_sha256']!=data_hash
            or d['trial_id']!=trial_id or sha256_file(directory/'bank.npz')!=d['bank_sha256']):
        raise RuntimeError('incompatible or corrupted GPU bank')
    with np.load(directory/'bank.npz') as b:
        if not np.array_equal(b['original'],original):raise RuntimeError('original vector mismatch')
    return d


def save_bank(directory, original, bank, record):
    directory.mkdir(parents=True,exist_ok=True)
    tmp=directory/'bank.npz.tmp'
    with tmp.open('wb') as f:
        np.savez_compressed(f,original=original,bank=bank,trial_id=record['trial_id'])
    tmp.replace(directory/'bank.npz')
    record['bank_sha256']=sha256_file(directory/'bank.npz')
    atomic_json(directory/'record.json',record)


def load_features(root, subject, plan):
    file=root/f's{subject:02d}.npz';meta=json.loads(file.with_suffix('.json').read_text())
    expected=plan['feature_exports'][str(subject)]
    if meta!=expected or sha256_file(file)!=meta['feature_file_sha256']:
        raise RuntimeError('feature export provenance mismatch')
    with np.load(file) as f:values=f['values'];ids=f['ids']
    if values.dtype!=np.float32 or values.shape!=(meta['trials'],8128):
        raise RuntimeError('unexpected feature layout')
    if not np.isfinite(values).all() or (values<0).any():raise RuntimeError('invalid PLV values')
    return values,ids,meta


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--features-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--subjects',type=int,nargs='+')
    p.add_argument('--batch-trials',type=int,default=4)
    p.add_argument('--validation',action='store_true')
    p.add_argument('--acceptance',type=Path)
    a=p.parse_args();plan,ph=verify_plan(a.plan)
    if a.batch_trials<1 or a.batch_trials>8:raise ValueError('batch bound is 1–8 trials')
    if not a.validation:
        if a.acceptance is None:raise RuntimeError('formal launch requires validation acceptance')
        accepted=json.loads(a.acceptance.read_text())
        if not accepted.get('all_quality_gates_pass') or accepted['plan_sha256']!=ph:
            raise RuntimeError('GPU validation incomplete or failed')
        if accepted.get('validation') is not True or accepted['subjects']!=plan['subjects']:
            raise RuntimeError('acceptance must cover all subjects')
    subjects=a.subjects or plan['subjects']
    if len(set(subjects))!=len(subjects) or not set(subjects)<=set(plan['subjects']):
        raise ValueError('invalid subject assignment')
    a.out.mkdir(parents=True,exist_ok=True)
    lock=(a.out/'generator.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    saved=a.out/'plan.json'
    if saved.exists() and json.loads(saved.read_text())!=plan:raise RuntimeError('output plan mismatch')
    atomic_json(saved,plan)
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.4)
    done=failed=0;start=time.monotonic();all_rows=[]
    expected=sum(len(plan['validation_indices'][str(s)]) if a.validation
                 else plan['feature_exports'][str(s)]['trials'] for s in subjects)
    size=plan['bank_size']
    for subject in subjects:
        values,ids,meta=load_features(a.features_root,subject,plan)
        indices=plan['validation_indices'][str(subject)] if a.validation else list(range(len(values)))
        pending=[]
        for index in indices:
            path=a.out/f's{subject:02d}'/f't{index:04d}'
            old=existing(path,ph,meta['training_data_sha256'],values[index],int(ids[index]))
            if old is None:pending.append(index)
            else:done+=1;failed+=not old['passed'];all_rows.append(old)
        for offset in range(0,len(pending),a.batch_trials):
            selection=pending[offset:offset+a.batch_trials]
            keys=[(plan['protocol'],'T',subject,int(ids[i]),member)
                  for i in selection for member in range(size)]
            originals=np.repeat(values[selection],size,axis=0)
            began=time.monotonic()
            generated,report=generate(originals,keys,plan['phases'])
            bank=generated.astype(np.float32).reshape(len(selection),size,-1)
            if not np.array_equal(generated,bank.reshape(-1,8128).astype(np.float64)):
                raise RuntimeError('weight rounding would alter exact multiset')
            for position,index in enumerate(selection):
                checks=quality(values[index],bank[position],plan['gates'])
                record=dict(plan_sha256=ph,protocol=plan['protocol'],subject=subject,
                    trial_index=index,trial_id=int(ids[index]),partition='T',
                    training_data_sha256=meta['training_data_sha256'],
                    feature_file_sha256=meta['feature_file_sha256'],
                    surrogate_keys=[list(x) for x in keys[position*size:(position+1)*size]],
                    host=platform.node(),gpu=torch.cuda.get_device_name(),
                    numpy=np.__version__,torch=torch.__version__,triton=triton.__version__,
                    batch_trials=len(selection),generation_seconds=report['seconds'],
                    phase_energy=[dict(name=r['name'],
                        reported=r['reported_energy'][position*size:(position+1)*size],
                        recomputed=r['actual_energy'][position*size:(position+1)*size])
                        for r in report['phases']],E_set_accessed=False,**checks)
                save_bank(a.out/f's{subject:02d}'/f't{index:04d}',values[index],bank[position],record)
                done+=1;failed+=not record['passed'];all_rows.append(record)
            progress=dict(stage='generating_T_GPU_v2',validation=a.validation,
                subject=subject,completed_trials=done,failed_trials=failed,
                assigned_trials=expected,elapsed_seconds=time.monotonic()-start,
                batch_end_to_end_seconds=time.monotonic()-began,updated_unix_time=time.time())
            atomic_json(a.out/'progress.json',progress);print(json.dumps(progress),flush=True)
    if done!=expected:raise RuntimeError('incomplete assigned grid')
    completion=dict(stage='finished' if failed==0 else 'finished_with_quality_failures',
        validation=a.validation,subjects=subjects,completed_trials=done,failed_trials=failed,
        assigned_trials=expected,all_quality_gates_pass=failed==0,plan_sha256=ph,
        elapsed_seconds=time.monotonic()-start,updated_unix_time=time.time(),
        summary=dict(maximum_relative_strength_error=max(x['relative_strength_max'] for r in all_rows for x in r['metrics']),
                     minimum_diversity=min(r['minimum_pairwise_diversity'] for r in all_rows)))
    atomic_json(a.out/'completion.json',completion);atomic_json(a.out/'progress.json',completion)
    if failed:raise SystemExit(2)


if __name__=='__main__':main()
