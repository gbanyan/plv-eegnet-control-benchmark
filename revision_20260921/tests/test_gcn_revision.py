import torch
from scripts.gcn_revision_model import ChannelGCN


def model(mode='real'):
    return ChannelGCN({'n_channels':4,'n_classes':2},{'mode':mode},71)


def test_adjacency_normalization_and_identity():
    m=model();v=torch.ones(2,6)
    assert torch.allclose(m.adjacency(v),torch.full((2,4,4),.25))
    assert torch.equal(model('baseline').adjacency(v),torch.eye(4).expand(2,-1,-1))
    assert torch.allclose(m.adjacency(v),m.adjacency(v).transpose(1,2))


def test_pairing_feature_cache_adjacency_sensitivity_and_gradients():
    torch.set_num_threads(2)
    m=model().eval();other=model('baseline').eval()
    for a,b in zip(m.state_dict().values(),other.state_dict().values()):assert torch.equal(a,b)
    x=torch.randn(3,4,1000);x[:,1]*=3;x[:,2]*=.2
    v=torch.rand(3,6);permuted=v[:,[3,0,5,1,2,4]]
    output=m(x,v)
    assert torch.allclose(output,m.from_embedding(m._eegnet_features(x),v))
    assert not torch.allclose(output,m(x,permuted))
    assert torch.equal(other(x,v),other(x,permuted))
    torch.nn.functional.cross_entropy(output,torch.tensor([0,1,0])).backward()
    for layer in m.eegnet.graph:
        assert torch.isfinite(layer.weight.grad).all() and layer.weight.grad.abs().sum()>0


def test_constant_input_finite():
    m=model();out=m(torch.zeros(2,4,1000),torch.zeros(2,6))
    assert torch.isfinite(out).all()


def test_gcn_training_seal_evaluation_and_tamper_guard(tmp_path,monkeypatch):
    import runpy
    from scripts import train_revision,evaluate_revision
    monkeypatch.setattr(train_revision,'RevisionModel',ChannelGCN)
    monkeypatch.setattr(evaluate_revision,'RevisionModel',ChannelGCN)
    suite=runpy.run_path('tests/test_revision_protocol.py')
    suite['test_training_without_E_and_sealed_evaluation'](tmp_path,monkeypatch)
