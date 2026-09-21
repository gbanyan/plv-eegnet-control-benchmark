"""Bounded channel-graph decoder; independent of the frozen EEGNet study."""
import torch
from torch import nn


class GraphBackbone(nn.Module):
    def __init__(self, channels, width=32):
        super().__init__()
        self.project = nn.Linear(8, width)
        self.channel_embedding = nn.Parameter(torch.randn(channels, width)*.02)
        self.graph = nn.ModuleList([nn.Linear(width, width) for _ in range(2)])
        self.norm = nn.ModuleList([nn.LayerNorm(width) for _ in range(2)])
        self.dropout = nn.Dropout(.25)

    def forward(self, features, adjacency):
        h = torch.nn.functional.gelu(self.project(features)) + self.channel_embedding
        for linear, norm in zip(self.graph, self.norm):
            update = torch.nn.functional.gelu(linear(adjacency @ h))
            h = norm(h + self.dropout(update))
        return h.flatten(1)


class ChannelGCN(nn.Module):
    def __init__(self, spec, condition, init_seed):
        super().__init__()
        torch.manual_seed(init_seed)
        self.channels = spec['n_channels']
        self.identity_graph = condition['mode'] == 'baseline'
        # Existing trainer hashes this entire module as its backbone; no EEGNet
        # is instantiated despite the compatibility attribute name.
        self.eegnet = GraphBackbone(self.channels)
        self.classifier = nn.Linear(self.channels*32, spec['n_classes'])
        self.plv_norm = nn.Identity()
        self.register_buffer('upper', torch.triu_indices(self.channels,self.channels,1), persistent=False)

    def adjacency(self, values):
        eye = torch.eye(self.channels,device=values.device,dtype=values.dtype)
        if self.identity_graph:
            return eye.expand(len(values),-1,-1)
        a = values.new_zeros((len(values),self.channels,self.channels))
        a[:,self.upper[0],self.upper[1]] = values
        a = a + a.transpose(1,2) + eye
        inverse = a.sum(-1).rsqrt()
        return inverse[:,:,None]*a*inverse[:,None,:]

    def _eegnet_features(self, x):
        # Eight nonoverlapping half-second windows of the existing 8–30Hz EEG.
        # Fixed log-variance node features, no labels or fitted statistics.
        return x.reshape(len(x),self.channels,8,125).var(-1,unbiased=False).clamp_min(1e-8).log()

    def from_embedding(self, embedding, values):
        return self.classifier(self.eegnet(embedding,self.adjacency(values)))

    def forward(self, x, values):
        return self.from_embedding(self._eegnet_features(x),values)
