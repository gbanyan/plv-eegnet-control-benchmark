"""Generate E banks only after the complete 140-fit training seal validates."""
import argparse,fcntl,json,time,platform
from pathlib import Path
import numpy as np
import torch
from scripts.revision_manifest import load_manifest,manifest_hash
from scripts.evaluate_revision import validate_seal
from scripts.experimental.strength_gpu_backend import generate
from scripts.experimental.strength_targeted_refinement import refine
from scripts.pilot_strength_targeted import quality
from src.datasets.bci2a import BCI2aNPZ,trial_id_offset
from src.train_all_subjects import DATASET_INFO
from src.revision import plv_features,sha256_file,atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--training-output',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--subjects',type=int,nargs='+');a=p.parse_args()
    m=load_manifest(a.manifest);ph=manifest_hash(m);_,train_seal_hash=validate_seal(m,'strength_schirrmeister',a.training_output)
    cfg=m['E_bank'];subjects=a.subjects or list(range(1,15))
    if len(set(subjects))!=len(subjects) or not set(subjects)<=set(range(1,15)):raise ValueError('invalid E subjects')
    a.output.mkdir(parents=True,exist_ok=True);lock=(a.output/'generator.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.4)
    done=failed=0;entries={};started=time.monotonic()
    for subject in subjects:
        file=a.data_root/'schirrmeister'/DATASET_INFO['schirrmeister']['filename_fmt'].format(sid=subject,suffix='E');data_hash=sha256_file(file);data=BCI2aNPZ(file)
        if len(data)!=160:raise RuntimeError('unexpected E trial count')
        # Match the T export's CPU float32 PLV computation, without using labels.
        features=torch.cat([plv_features(x) for x in data.X.split(16)]).numpy()
        ids=np.arange(len(data),dtype=np.int64)+trial_id_offset(subject,evaluation=True)
        pending=[]
        for i in range(len(data)):
            relative=f's{subject:02d}/t{i:04d}';directory=a.output/relative;rp=directory/'record.json'
            if rp.exists():
                r=json.loads(rp.read_text())
                if r['classifier_manifest_sha256']!=ph or r['evaluation_data_sha256']!=data_hash or r['trial_id']!=int(ids[i]) or sha256_file(directory/'bank.npz')!=r['bank_sha256']:raise RuntimeError('incompatible E resume')
                with np.load(directory/'bank.npz') as f:
                    if not np.array_equal(f['original'],features[i]):raise RuntimeError('E features changed')
                done+=1;failed+=not r['passed'];entries[relative]={'record_sha256':sha256_file(rp),'bank_sha256':r['bank_sha256']}
            else:pending.append(i)
        for offset in range(0,len(pending),4):
            selection=pending[offset:offset+4];keys=[(cfg['rng_namespace'],'E',subject,int(ids[i]),member) for i in selection for member in range(16)]
            originals=np.repeat(features[selection],16,axis=0)
            generated,_=generate(originals,keys,cfg['phases']);refined=[];reports=[]
            for original,initial,key in zip(originals,generated,keys):
                w,r=refine(original,initial,key,**cfg['refinement']);refined.append(w);reports.append(r)
            doubles=np.stack(refined);bank=doubles.astype(np.float32).reshape(len(selection),16,8128)
            if not np.array_equal(doubles,bank.reshape(-1,8128).astype(float)):raise RuntimeError('E weight rounding')
            for k,i in enumerate(selection):
                relative=f's{subject:02d}/t{i:04d}';directory=a.output/relative;directory.mkdir(parents=True,exist_ok=True)
                qa=quality(features[i],bank[k],cfg['gates'])
                with (directory/'bank.npz.tmp').open('wb') as f:np.savez_compressed(f,original=features[i],bank=bank[k],trial_id=int(ids[i]))
                (directory/'bank.npz.tmp').replace(directory/'bank.npz')
                r=dict(classifier_manifest_sha256=ph,training_seal_sha256=train_seal_hash,partition='E',subject=subject,trial_index=i,trial_id=int(ids[i]),evaluation_data_sha256=data_hash,bank_sha256=sha256_file(directory/'bank.npz'),surrogate_keys=keys[k*16:(k+1)*16],refinement_reports=reports[k*16:(k+1)*16],host=platform.node(),**qa)
                atomic_json(directory/'record.json',r);entries[relative]={'record_sha256':sha256_file(directory/'record.json'),'bank_sha256':r['bank_sha256']};done+=1;failed+=not qa['passed']
            progress=dict(stage='generating_E_v3',completed_trials=done,assigned_trials=160*len(subjects),failed_trials=failed,elapsed_seconds=time.monotonic()-started,updated_unix_time=time.time())
            atomic_json(a.output/'progress.json',progress);print(json.dumps(progress),flush=True)
    complete=dict(stage='finished',partition='E',accepted=failed==0 and len(entries)==2240,classifier_manifest_sha256=ph,training_seal_sha256=train_seal_hash,subjects=subjects,completed_trials=done,failed_trials=failed,records=entries,elapsed_seconds=time.monotonic()-started)
    atomic_json(a.output/'completion.json',complete)
    atomic_json(a.output/'progress.json',{k:v for k,v in complete.items() if k!='records'})
    if failed:raise SystemExit(2)
    if complete['accepted']:atomic_json(a.output/'E_bank_seal.json',complete)


if __name__=='__main__':main()
