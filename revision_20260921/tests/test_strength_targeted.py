"""Independent finite-state checks for the candidate targeted refinement."""
import numpy as np
import pytest
from scripts.experimental.strength_targeted_refinement import refine,named_seed,strengths


def test_single_step_matches_exhaustive_actual_matrix_energy():
    rng=np.random.default_rng(33)
    for n in (4,7,12):
        u,v=np.triu_indices(n,1)
        original=rng.uniform(.1,1,len(u));initial=rng.permutation(original);key=('oracle',n)
        target=strengths(original,u,v,n);before=strengths(initial,u,v,n)
        a=int(np.argmax(abs((before-target)/target)))
        g=np.random.Generator(np.random.PCG64(named_seed((*key,'targeted-relative-v3'))))
        b=int(g.integers(n-1));b+=b>=a
        i=int(np.flatnonzero(((u==a)&(v==b))|((u==b)&(v==a)))[0])
        energies=[];candidates=[]
        for j in range(len(u)):
            w=initial.copy();w[i],w[j]=w[j],w[i]
            energies.append(np.sum(((strengths(w,u,v,n)-target)/target)**2));candidates.append(w)
        expected=candidates[int(np.argmin(energies))]
        actual,report=refine(original,initial,key,maximum_steps=1,maximum_relative=1e-15,rms_relative=1e-15)
        assert np.array_equal(actual,expected)
        assert np.isclose(report['final_energy'],min(energies),atol=1e-14)


def test_original_is_valid_stationary_and_repeats_are_deterministic():
    original=np.arange(1,29,dtype=float)/32
    unchanged,report=refine(original,original,('stationary',))
    assert np.array_equal(unchanged,original) and report['accepted']==0
    initial=original[::-1]
    a,r=refine(original,initial,('repeat',));b,s=refine(original,initial,('repeat',))
    assert np.array_equal(a,b) and r==s and np.array_equal(np.sort(a),np.sort(original))
    assert r['final_energy']<=r['initial_energy']


def test_invalid_input_is_rejected():
    with pytest.raises(ValueError):refine(np.arange(6.),np.arange(6.)+1,('bad',))
    with pytest.raises(ValueError):refine(np.arange(6.),np.arange(6.),('bad',),maximum_steps=0)


def test_production_persistence_failure_flag_and_tamper(tmp_path):
    import json
    from scripts.refine_strength_bank import one,sha
    parent=tmp_path/'parent';out=tmp_path/'out';relative='s01/t0000';src=parent/relative;src.mkdir(parents=True)
    original=np.arange(8128,dtype=np.float32)/8192
    with (src/'bank.npz').open('wb') as f:np.savez_compressed(f,original=original,bank=np.stack([original]*16),trial_id=1)
    record=dict(plan_sha256='parent-plan',partition='T',subject=1,trial_index=0,trial_id=1,passed=False,training_data_sha256='data',feature_file_sha256='features',surrogate_keys=[['test',i] for i in range(16)])
    (src/'record.json').write_text(json.dumps(record))
    entry={'record_sha256':sha(src/'record.json'),'bank_sha256':sha(src/'bank.npz')}
    gates=dict(relative_rms_max=.01,relative_max_max=.05,changed_min=.95,displacement_min=.25,diversity_min=.25)
    plan=dict(protocol='unit',parent_plan_sha256='parent-plan',bank_size=16,refinement={'maximum_steps':2},gates=gates)
    task=(str(parent),str(out),relative,entry,plan,'plan')
    _,saved=one(task);assert not saved['passed'] and saved['minimum_pairwise_diversity']==0
    assert one(task)[1]==saved
    (out/relative/'bank.npz').write_bytes(b'corrupted')
    with pytest.raises(RuntimeError,match='incompatible resume'):one(task)
