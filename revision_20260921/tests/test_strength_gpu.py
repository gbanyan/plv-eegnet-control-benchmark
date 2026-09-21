"""GPU protocol checks run on gblinux; no E inputs or classifier outcomes."""
import json
import numpy as np
import pytest
import torch
from scripts.experimental.strength_gpu_backend import generate, matrices, stage_tapes, stable_seed
from scripts.generate_strength_gpu import existing, save_bank, quality

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='GPU required')

PHASES=[dict(name='absolute',relative=False,stages=2,iterations=200,temperature=1000.,cooling=.5),
        dict(name='relative',relative=True,stages=2,iterations=200,temperature=.0001,cooling=.8)]


def reference(original,key,phases):
    current=np.random.default_rng(stable_seed(*key,'initial')).permutation(original)
    target=matrices(original[None])[0].sum(1);n=len(target);uv=np.triu_indices(n,1)
    for phase in phases:
        sb=matrices(current[None])[0].sum(1)
        def energy(s):
            diff=s-target
            if phase['relative']:diff=diff/np.maximum(target,1.e-12)
            return np.sum(diff*diff)
        e=minimum=energy(sb);best=current.copy()
        for stage in range(phase['stages']):
            first,second,prob=stage_tapes([key],phase['name'],stage,len(current),phase['iterations'])
            for i,j,p in zip(first[0],second[0],prob[0]):
                a,b=uv[0][i],uv[1][i];c,d=uv[0][j],uv[1][j]
                candidate=sb.copy()
                candidate[[a,b]]=candidate[[a,b]]-current[i]+current[j]
                candidate[[c,d]]=candidate[[c,d]]+current[i]-current[j]
                ep=energy(candidate)
                if ep<e or p<np.exp(-(ep-e)/(phase['temperature']*phase['cooling']**stage)):
                    sb=candidate;e=ep;current[i],current[j]=current[j],current[i]
                    if ep<minimum:minimum=ep;best=current.copy()
        current=best
    return current


def test_reference_and_batch_order_independence():
    values=np.random.default_rng(4).integers(1,100,size=(3,6)).astype(np.float64)/128
    keys=[('unit','T',1,100,i) for i in range(3)]
    together,_=generate(values,keys,PHASES)
    for i in range(3):
        alone,_=generate(values[i:i+1],[keys[i]],PHASES)
        assert np.array_equal(alone[0],together[i])
        assert np.array_equal(reference(values[i],keys[i],PHASES),together[i])
    reordered,_=generate(values[::-1],keys[::-1],PHASES)
    assert np.array_equal(reordered,together[::-1])


def test_shared_node_and_same_edge_tapes_are_present():
    x,y,_=stage_tapes([('unit',)],'absolute',0,6,200)
    assert np.any(x==y)
    uv=np.triu_indices(4,1)
    assert any(i!=j and len(set([uv[0][i],uv[1][i],uv[0][j],uv[1][j]]))==3
               for i,j in zip(x[0],y[0]))


def test_persist_resume_and_tamper_rejection(tmp_path):
    original=np.arange(1,7,dtype=np.float32)/8
    bank=np.stack([original,original[::-1]])
    record=dict(plan_sha256='plan',training_data_sha256='data',trial_id=99)
    save_bank(tmp_path,original,bank,record)
    assert existing(tmp_path,'plan','data',original,99)['trial_id']==99
    with pytest.raises(RuntimeError):existing(tmp_path,'wrong','data',original,99)
    (tmp_path/'bank.npz').write_bytes(b'corrupted')
    with pytest.raises(RuntimeError):existing(tmp_path,'plan','data',original,99)


def test_quality_rejects_identity_and_keeps_exact_weights():
    gates=json.load(open('configs/strength_bank_T_v1_20260920.json'))['gates']
    original=np.arange(1,7,dtype=np.float32)/8
    report=quality(original,np.stack([original]*16),gates)
    assert not report['passed'] and report['minimum_pairwise_diversity']==0
    assert report['metrics'][0]['exact_multiset']


def test_late_temperature_remains_positive_float64():
    phases=[dict(name='absolute',relative=False,stages=300,iterations=10,
                 temperature=1000.,cooling=.5)]
    original=np.array([[1,3,8,11,18,25]],dtype=np.float64)/32
    key=('cold-test',)
    output,report=generate(original,[key],phases)
    temperature=report['phases'][0]['minimum_temperature']
    assert 0 < temperature < np.finfo(np.float32).tiny
    assert np.array_equal(output[0],reference(original[0],key,phases))
