"""Use proven train/seal/evaluate code with the explicitly declared GCN model."""
import sys
from scripts.gcn_revision_model import ChannelGCN


def main():
    stage=sys.argv.pop(1)
    if stage=='train':
        from scripts import train_revision as runtime
    elif stage=='evaluate':
        from scripts import evaluate_revision as runtime
    else:raise ValueError(stage)
    runtime.RevisionModel=ChannelGCN
    runtime.main()


if __name__=='__main__':main()
