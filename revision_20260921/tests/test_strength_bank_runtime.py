import json
from pathlib import Path
import numpy as np
import pytest
import torch
from scripts.strength_bank_runtime import BankControl,load_banks,SETTINGS
from src.revision import sha256_file


def test_bank_draws_match_marginal_support_and_preserve_validation():
    bank=torch.arange(12*16*6,dtype=torch.float32).reshape(12,16,6)
    ids=torch.arange(12)+100;train=torch.arange(8);seeds={'fixed':17,'permutation':99}
    state=torch.random.get_rng_state().clone()
    f=BankControl(bank,ids,train,{'mode':'fixed'},seeds);r=BankControl(bank,ids,train,{'mode':'resampled'},seeds)
    assert torch.equal(f.fixed,r.fixed) and torch.equal(torch.random.get_rng_state(),state)
    for epoch in range(1,6):
        f.begin_epoch(epoch);r.begin_epoch(epoch)
        assert torch.equal(r.current[8:],r.fixed[8:])
        assert torch.equal(f.current,f.fixed)
        assert all(any(torch.equal(r.current[i],row) for row in bank[i]) for i in range(12))
    assert r.redraw_count==32 and f.redraw_count==0
    assert not torch.equal(r.current[:8],r.fixed[:8])
    assert torch.equal(torch.random.get_rng_state(),state)


def test_bank_input_mismatch_rejected():
    with pytest.raises(ValueError):BankControl(torch.zeros(3,6),torch.arange(3),torch.arange(2),{'mode':'fixed'},{})


def test_loader_enforces_sealed_checksum_partition_and_data(tmp_path):
    directory=tmp_path/'s01/t0000';directory.mkdir(parents=True)
    np.savez_compressed(directory/'bank.npz',bank=np.ones((16,8128),dtype=np.float32),original=np.ones(8128,dtype=np.float32),trial_id=100)
    rp=directory/'record.json';rp.write_text(json.dumps(dict(passed=True,partition='T',trial_id=100,training_data_sha256='data')))
    SETTINGS['Troot']=tmp_path;SETTINGS['Tseal']={'records':{'s01/t0000':{'record_sha256':sha256_file(rp),'bank_sha256':sha256_file(directory/'bank.npz')}}}
    assert load_banks('T',1,torch.tensor([100]),'data',torch.device('cpu')).shape==(1,16,8128)
    with pytest.raises(RuntimeError,match='checksum'):load_banks('T',1,torch.tensor([100]),'other-data',torch.device('cpu'))
    with pytest.raises(RuntimeError,match='identity'):load_banks('T',1,torch.tensor([101]),'data',torch.device('cpu'))
    (directory/'bank.npz').write_bytes(b'tampered')
    with pytest.raises(RuntimeError,match='checksum'):load_banks('T',1,torch.tensor([100]),'data',torch.device('cpu'))
