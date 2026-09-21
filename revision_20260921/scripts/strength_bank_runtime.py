"""Finite-bank adapters for the unchanged train/checkpoint/seal runtime."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from scripts import train_revision,evaluate_revision
from scripts.revision_manifest import load_manifest
from src.revision import stream_seed,sha256_file


class BankControl:
    def __init__(self,values,ids,train_indices,condition,seeds,**kwargs):
        if values.ndim!=3 or len(values)!=len(ids):raise ValueError('expected trial/member/coordinate bank')
        self.values=values;self.train_indices=train_indices;self.mode=condition['mode'];self.redraw_count=0
        if self.mode not in ('fixed','resampled'):raise ValueError('unsupported bank mode')
        self.generator=torch.Generator().manual_seed(stream_seed(seeds['permutation'],'finite-bank'))
        chosen=torch.tensor([int(torch.randint(values.shape[1],(1,),generator=torch.Generator().manual_seed(stream_seed(seeds['fixed'],'finite-bank',int(i))))[0]) for i in ids.tolist()],device=values.device)
        self.fixed=values[torch.arange(len(values),device=values.device),chosen];self.current=self.fixed.clone()
    def begin_epoch(self,epoch):
        if epoch==1 or self.mode=='fixed':return
        chosen=torch.randint(self.values.shape[1],(len(self.train_indices),),generator=self.generator).to(self.values.device)
        selected=self.train_indices.to(self.values.device)
        self.current[selected]=self.values[selected,chosen];self.redraw_count+=len(selected)


SETTINGS={}


def load_banks(partition,subject,ids,data_hash,device):
    seal=SETTINGS[partition+'seal'];root=SETTINGS[partition+'root'];rows=[]
    for i,trial_id in enumerate(ids.tolist()):
        relative=f's{subject:02d}/t{i:04d}';entry=seal['records'][relative];directory=root/relative
        record=directory/'record.json'
        if sha256_file(record)!=entry['record_sha256']:raise RuntimeError('sealed bank record changed')
        d=json.loads(record.read_text())
        if not d['passed'] or d['partition']!=partition or d['trial_id']!=trial_id:raise RuntimeError('bank identity/QA mismatch')
        actual_data_hash=d['training_data_sha256'] if partition=='T' else d['evaluation_data_sha256']
        if actual_data_hash!=data_hash or sha256_file(directory/'bank.npz')!=entry['bank_sha256']:raise RuntimeError('bank/data checksum mismatch')
        with np.load(directory/'bank.npz') as f:
            if int(f['trial_id'])!=trial_id or f['bank'].shape!=(16,8128):raise RuntimeError('bank layout mismatch')
            rows.append(f['bank'])
    return torch.from_numpy(np.stack(rows)).to(device)


class BankTrainingCache(train_revision.TrainingCache):
    def features(self,condition):
        if 'bank' not in self.values:self.values['bank']=load_banks('T',self.key[1],self.ids,self.data_hash,self.device)
        return self.values['bank'],torch.arange(8128)


class BankEvaluationCache(evaluate_revision.EvaluationCache):
    def features(self,condition):
        if 'bank' not in self.values:self.values['bank']=load_banks('E',self.key[1],self.ids,self.e_hash,self.device)
        return self.values['bank']


def main():
    stage=sys.argv.pop(1)
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--bank-root-T',type=Path,required=True);p.add_argument('--bank-seal-T',type=Path,required=True)
    p.add_argument('--bank-root-E',type=Path);p.add_argument('--bank-seal-E',type=Path)
    args,remaining=p.parse_known_args();manifest_path=Path(remaining[remaining.index('--manifest')+1]);m=load_manifest(manifest_path)
    if sha256_file(args.bank_seal_T)!=m['T_bank_seal_sha256']:raise RuntimeError('wrong training-bank seal')
    for partition in ('T','E') if stage=='evaluate' else ('T',):
        root=getattr(args,'bank_root_'+partition);path=getattr(args,'bank_seal_'+partition)
        if root is None or path is None:raise ValueError('missing bank root/seal')
        seal=json.loads(path.read_text())
        if not seal['accepted'] or seal['partition']!=partition:raise RuntimeError('bank not accepted')
        if partition=='E' and seal['classifier_manifest_sha256']!=evaluate_revision.manifest_hash(m):raise RuntimeError('E bank from another protocol')
        SETTINGS[partition+'root']=root;SETTINGS[partition+'seal']=seal
    if stage=='train':runtime=train_revision;runtime.TrainingCache=BankTrainingCache
    elif stage=='evaluate':runtime=evaluate_revision;runtime.TrainingCache=BankTrainingCache;runtime.EvaluationCache=BankEvaluationCache
    else:raise ValueError(stage)
    runtime.FeatureControl=BankControl
    sys.argv=[sys.argv[0],*remaining];runtime.main()


if __name__=='__main__':main()
