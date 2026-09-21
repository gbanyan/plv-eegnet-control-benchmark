"""Resumable, fail-closed T training -> sealed E banks -> fixed-E evaluation."""
import argparse,fcntl,json,subprocess,sys,time
from pathlib import Path
from scripts.revision_manifest import load_manifest,manifest_hash
from src.revision import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--T-root',type=Path,required=True);p.add_argument('--T-seal',type=Path,required=True);p.add_argument('--E-root',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);lock=(a.output/'pipeline.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    m=load_manifest(a.manifest);atomic_json(a.output/'manifest.json',m);start=time.time()
    base=[sys.executable,'-m','scripts.strength_bank_runtime'];common=['--manifest',str(a.manifest),'--batch','strength_schirrmeister','--data-root',str(a.data_root),'--output',str(a.output),'--memory-fraction','.45','--bank-root-T',str(a.T_root),'--bank-seal-T',str(a.T_seal)]
    stages=[('training',base+['train']+common),('E_banks',[sys.executable,'-m','scripts.generate_strength_e','--manifest',str(a.manifest),'--training-output',str(a.output),'--data-root',str(a.data_root),'--output',str(a.E_root)]),('evaluation',base+['evaluate']+common+['--bank-root-E',str(a.E_root),'--bank-seal-E',str(a.E_root/'E_bank_seal.json')]),('summary',[sys.executable,'-m','scripts.summarize_strength_classifier','--manifest',str(a.manifest),'--root',str(a.output)])]
    for stage,command in stages:
        load_manifest(a.manifest);atomic_json(a.output/'pipeline.json',dict(stage=stage,manifest_sha256=manifest_hash(m),started=start,updated=time.time()))
        subprocess.run(command,check=True)
    atomic_json(a.output/'pipeline.json',dict(stage='completed',manifest_sha256=manifest_hash(m),started=start,completed=time.time()))


if __name__=='__main__':main()
