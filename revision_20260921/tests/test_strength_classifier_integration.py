"""GPU synthetic-only train/seal/evaluate compatibility and E-access guard."""
import json,sys
from pathlib import Path
import pytest
import torch
from scripts import train_revision,evaluate_revision
from scripts.strength_bank_runtime import BankControl

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='synthetic GPU integration')


def test_complete_synthetic_pipeline_and_unsealed_E_guard(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    conditions={k:dict(mode=k,normalization='batchnorm',band=[8.,30.]) for k in ['fixed','resampled']}
    jobs=[dict(id='synthetic_'+k,dataset='schirrmeister',subject=1,replicate=1001,condition=k,batch='test') for k in conditions]
    manifest=dict(training=dict(inner_validation_fraction=.2,lr=.001,weight_decay=.0001,minimum_epochs=1,patience=1,min_delta=.0001,maximum_epochs=2,fixed_budget={'schirrmeister':2},batch_size=8),conditions=conditions,jobs=jobs,evaluation={'mc_conditions':[]})
    class Synthetic:
        device=torch.device('cuda');data_hash='synthetic-data';e_hash='synthetic-E'
        def __init__(self):
            g=torch.Generator().manual_seed(3)
            self.x=torch.randn(20,128,1000,generator=g).cuda();self.y=(torch.arange(20)%4).cuda();self.ids=torch.arange(20)+100
            self.bank=torch.rand(20,16,8128,generator=g).cuda();self.training=self
        def load(self,*args):pass
        def features(self,condition):return self.bank,torch.arange(8128)
    cache=Synthetic();monkeypatch.setattr(train_revision,'FeatureControl',BankControl);monkeypatch.setattr(evaluate_revision,'FeatureControl',BankControl)
    with pytest.raises(FileNotFoundError):evaluate_revision.validate_seal(manifest,'test',tmp_path)
    records=[train_revision.train_one(j,manifest,cache,tmp_path) for j in jobs]
    assert records[0]['initial_backbone_sha256']==records[1]['initial_backbone_sha256']
    assert records[0]['initial_classifier_sha256']==records[1]['initial_classifier_sha256']
    assert records[0]['fixed_features_sha256']==records[1]['fixed_features_sha256']
    assert all(not r['E_set_accessed'] for r in records)
    train_revision.seal_batch(manifest,'test',tmp_path);_,sh=evaluate_revision.validate_seal(manifest,'test',tmp_path)
    class Evaluation:
        x=cache.x;y=cache.y;ids=cache.ids;training=cache;device=cache.device;e_hash=cache.e_hash
        def load(self,*args):pass
        def features(self,condition):return cache.bank
    for job in jobs:
        result=evaluate_revision.evaluate_one(job,manifest,Evaluation(),tmp_path,sh)
        assert set(result['metrics'])=={'primary','accuracy','fixed_budget'}
        assert all(0<=r['fixed_accuracy']<=1 for r in result['metrics'].values())
    # A changed checkpoint invalidates the seal before any E loading.
    path=tmp_path/'jobs'/jobs[0]['id']/'primary.pt';path.write_bytes(b'tampered')
    with pytest.raises(RuntimeError,match='checkpoint changed'):evaluate_revision.validate_seal(manifest,'test',tmp_path)


def test_E_generation_refuses_missing_training_seal(tmp_path,monkeypatch):
    from scripts import generate_strength_e
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'source_sha256':{},'jobs':[]}))
    def forbidden(*args,**kwargs):raise AssertionError('E loader must never be reached')
    monkeypatch.setattr(generate_strength_e,'BCI2aNPZ',forbidden)
    monkeypatch.setattr(sys,'argv',['generate_strength_e','--manifest',str(manifest),'--training-output',str(tmp_path/'unsealed'),'--data-root',str(tmp_path/'data'),'--output',str(tmp_path/'out')])
    with pytest.raises(FileNotFoundError):generate_strength_e.main()
    assert not (tmp_path/'out').exists()
