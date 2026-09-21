"""Export CPU PLV on the original host for identical cross-host bank inputs."""
import argparse
from pathlib import Path
import numpy as np
import torch
from scripts.train_revision import TrainingCache
from src.revision import plv_features, atomic_json, sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--subjects',nargs='+',type=int,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    cache=TrainingCache(a.data_root,torch.device('cpu'))
    for subject in a.subjects:
        cache.load('schirrmeister',subject)
        values=torch.cat([plv_features(x) for x in cache.x.split(8)]).numpy()
        path=a.out/f's{subject:02d}.npz';np.savez(path,values=values,ids=cache.ids.numpy())
        atomic_json(path.with_suffix('.json'),{'subject':subject,'trials':len(values),'training_data_sha256':cache.data_hash,'feature_file_sha256':sha256_file(path),'torch':torch.__version__,'host':'gblab146'})
        print('exported',subject,len(values),flush=True)


if __name__=='__main__':main()
