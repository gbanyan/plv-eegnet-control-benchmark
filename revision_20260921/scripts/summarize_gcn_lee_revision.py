"""Report only the complete separately locked GCN study."""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.revision_manifest import load_manifest,manifest_hash
from scripts.summarize_revision import contrast_statistics
from src.revision import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    m=load_manifest(a.manifest);ph=manifest_hash(m);records={}
    for job in m['jobs']:
        path=a.root/'jobs'/job['id']/'evaluation.json'
        if not path.exists():raise RuntimeError('incomplete GCN evaluation grid')
        r=json.loads(path.read_text());assert r['manifest_sha256']==ph
        records[(job['condition'],job['subject'],job['replicate'])]=r
    def matrix(condition,rule):
        return np.array([[records[(condition,s,r)]['metrics'][rule]['fixed_accuracy'] for r in range(2001,2006)] for s in range(1,55)])
    results={};cells={}
    for rule in ['primary','accuracy','fixed_budget']:
        values={c:matrix(c,rule) for c in m['conditions']}
        cells[rule]={c:float(v.mean()*100) for c,v in values.items()}
        for name,first,second in [('schedule','resampled','fixed'),('real_minus_identity','real','baseline'),('real_minus_fixed','real','fixed'),('real_minus_resampled','real','resampled')]:
            result=contrast_statistics(values[first]-values[second])
            result['status']='single prespecified contrast in its family' if rule=='primary' and name in ['schedule','real_minus_identity'] else 'descriptive sensitivity'
            results[f'{rule}:{name}']=result
    atomic_json(a.root/'summary.json',{'manifest_sha256':ph,'complete':True,'fits':len(records),'cells':cells,'contrasts':results,'limitation':'Separate Lee2019 channel-graph/log-power study; not a pure architecture intervention relative to PLV-EEGNet.'})
    lines=['# Completed Lee2019 GCN extension','','1080 evaluated fits; 54 subjects, 5 paired replicates.','', '| Contrast | Difference (pp) | Model 95% CI | Unadjusted p |','| --- | ---: | --- | ---: |']
    for k,r in results.items():
        lo,hi=r['model_95ci_pp'];lines.append(f"| {k} | {r['mean_difference_pp']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {r['p_two_sided']:.5g} |")
    lines+=['','The primary schedule contrast is the sole primary hypothesis; real-minus-identity is a separately declared secondary contrast. Other rows are descriptive. No cross-architecture causal attribution or universal generalization is established.']
    (a.root/'STATUS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
