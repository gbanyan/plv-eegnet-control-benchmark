"""Tests for the `shuffle_pairs_fixed` PLV ablation mode.

`shuffle_pairs_fixed` must behave exactly like `shuffle_pairs` at the
*marginal* level (same per-trial value multiset, channel-pair identity
destroyed) while being STABLE across forward passes for a given trial --
the property that isolates the same-marginal pair-position effect from the
per-epoch reshuffling stochastic regularizer.
"""
import torch

from src.models.plv_eegnet import PLVEEGNet


def _model(mode):
    torch.manual_seed(0)
    return PLVEEGNet(
        n_channels=8, n_classes=2, n_samples=512, sampling_rate=128,
        phase_band=(8.0, 30.0), plv_mode=mode,
    ).eval()


def _input():
    g = torch.Generator().manual_seed(123)
    return torch.randn(4, 8, 512, generator=g)


def test_fixed_shuffle_is_epoch_stable():
    # Same input twice -> identical permuted PLV (stable across "epochs").
    m = _model("shuffle_pairs_fixed")
    x = _input()
    with torch.no_grad():
        a = m._compute_plv(x)
        b = m._compute_plv(x)
    assert torch.equal(a, b)


def test_regular_shuffle_is_not_stable():
    # Control: the original mode redraws the permutation every forward pass.
    m = _model("shuffle_pairs")
    x = _input()
    with torch.no_grad():
        a = m._compute_plv(x)
        b = m._compute_plv(x)
    assert not torch.equal(a, b)


def test_fixed_shuffle_preserves_marginal_but_permutes():
    real = _model("real")
    fixed = _model("shuffle_pairs_fixed")
    x = _input()
    with torch.no_grad():
        r = real._compute_plv(x)
        f = fixed._compute_plv(x)
    # Same multiset of values per trial (marginal preserved)...
    assert torch.allclose(r.sort(dim=-1).values, f.sort(dim=-1).values, atol=1e-6)
    # ...but actually reordered (channel-pair identity destroyed).
    assert not torch.equal(r, f)


def test_fixed_shuffle_is_content_keyed_not_position_keyed():
    # The permutation must depend only on the trial's content, not on its
    # row in the batch. Swapping two trials must swap their permuted outputs
    # exactly -- this simultaneously proves (a) stability per trial-identity
    # and (b) that different trials receive different permutations.
    m = _model("shuffle_pairs_fixed")
    x = _input()
    x_swapped = x.clone()
    x_swapped[[0, 1]] = x[[1, 0]]
    with torch.no_grad():
        f = m._compute_plv(x)
        fs = m._compute_plv(x_swapped)
    assert torch.equal(fs[0], f[1])
    assert torch.equal(fs[1], f[0])
