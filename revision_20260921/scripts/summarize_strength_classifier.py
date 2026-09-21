"""Report the locked finite-bank HGD schedule comparison on a complete grid."""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.revision_manifest import load_manifest,manifest_hash
from scripts.summarize_revision import contrast_statistics
from scripts.evaluate_revision import validate_seal
from src.revision import atomic_json,sha256_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    m=load_manifest(a.manifest);ph=manifest_hash(m);_,sh=validate_seal(m,'strength_schirrmeister',a.root);records={}
    for job in m['jobs']:
        folder=a.root/'jobs'/job['id'];r=json.loads((folder/'evaluation.json').read_text())
        if r['manifest_sha256']!=ph or r['seal_sha256']!=sh or sha256_file(folder/'predictions.npz')!=r['predictions_sha256']:raise RuntimeError('evaluation provenance mismatch')
        records[(job['subject'],job['replicate'],job['condition'])]=r
    contrasts={};cells={}
    for rule in ['primary','accuracy','fixed_budget']:
        values={condition:np.array([[records[s,replicate,condition]['metrics'][rule]['fixed_accuracy'] for replicate in range(1001,1006)] for s in range(1,15)]) for condition in ['fixed','resampled']}
        contrasts[rule]=contrast_statistics(values['resampled']-values['fixed']);cells[rule]={k:float(v.mean()*100) for k,v in values.items()}
    summary=dict(complete=True,fits=len(records),manifest_sha256=ph,contrasts=contrasts,cells=cells,note='Single declared primary HGD schedule contrast; secondary checkpoints are descriptive. Finite shared-bank generation uncertainty is not represented by classifier replicate SEs. No equivalence claim.')
    atomic_json(a.root/'summary.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
