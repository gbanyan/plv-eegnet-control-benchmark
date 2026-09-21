"""One bounded, resumable train/seal/evaluate pipeline; no host shutdown."""
import argparse,fcntl,json,subprocess,sys,time
from pathlib import Path
from scripts.revision_manifest import load_manifest,manifest_hash
from src.revision import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--seconds',type=int,default=28800);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);lock=(a.output/'worker.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    m=load_manifest(a.manifest);atomic_json(a.output/'manifest.json',m)
    start=time.time();atomic_json(a.output/'worker.json',{'started':start,'deadline':start+a.seconds,'manifest_sha256':manifest_hash(m)})
    common=['--manifest',str(a.manifest),'--batch','gcn_schirrmeister','--output',str(a.output),'--data-root',str(a.data_root),'--memory-fraction','.25']
    for stage in ['train','evaluate']:
        load_manifest(a.manifest)
        remaining=a.seconds-(time.time()-start)
        if remaining<=0:raise TimeoutError('overnight budget exhausted')
        try:
            subprocess.run([sys.executable,'-m','scripts.gcn_revision_stage',stage,*common],check=True,timeout=remaining)
        except subprocess.TimeoutExpired:
            atomic_json(a.output/'overnight_status.json',{'status':'time_budget_reached','stage':stage,'completed_fits_preserved':True});raise
    subprocess.run([sys.executable,'-m','scripts.summarize_gcn_revision','--manifest',str(a.manifest),'--root',str(a.output)],check=True)
    atomic_json(a.output/'overnight_status.json',{'status':'completed','elapsed_seconds':time.time()-start})


if __name__=='__main__':main()
