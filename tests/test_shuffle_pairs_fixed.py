"""Tests for the `shuffle_pairs_fixed` PLV ablation mode.

`shuffle_pairs_fixed` must behave exactly like `shuffle_pairs` at the
*marginal* level (same per-trial value multiset, channel-pair identity
destroyed) while being STABLE across forward passes for a given trial --
the property that isolates the same-marginal pair-position effect from the
per-epoch reshuffling stochastic regularizer.
"""
import torch
import pytest

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


def _trial_ids():
    return torch.tensor([101, 205, 309, 412], dtype=torch.int64)


def test_fixed_shuffle_is_epoch_stable():
    # Same input twice -> identical permuted PLV (stable across "epochs").
    m = _model("shuffle_pairs_fixed")
    x = _input()
    trial_ids = _trial_ids()
    with torch.no_grad():
        a = m._compute_plv(x, trial_ids)
        b = m._compute_plv(x, trial_ids)
    assert torch.equal(a, b)


def test_fixed_shuffle_rejects_missing_trial_ids():
    model = _model("shuffle_pairs_fixed")
    with pytest.raises(ValueError, match="trial_ids are required"):
        model._compute_plv(_input())


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
    trial_ids = _trial_ids()
    with torch.no_grad():
        r = real._compute_plv(x)
        f = fixed._compute_plv(x, trial_ids)
    # Same multiset of values per trial (marginal preserved)...
    assert torch.allclose(r.sort(dim=-1).values, f.sort(dim=-1).values, atol=1e-6)
    # ...but actually reordered (channel-pair identity destroyed).
    assert not torch.equal(r, f)


def test_fixed_shuffle_is_trial_id_keyed_not_position_or_content_keyed():
    # Swapping inputs together with their IDs must swap outputs exactly. The
    # ID-derived permutations themselves must not change with signal content.
    m = _model("shuffle_pairs_fixed")
    x = _input()
    trial_ids = _trial_ids()
    x_swapped = x.clone()
    x_swapped[[0, 1]] = x[[1, 0]]
    ids_swapped = trial_ids.clone()
    ids_swapped[[0, 1]] = trial_ids[[1, 0]]
    with torch.no_grad():
        f = m._compute_plv(x, trial_ids)
        fs = m._compute_plv(x_swapped, ids_swapped)
    assert torch.equal(fs[0], f[1])
    assert torch.equal(fs[1], f[0])
    assert torch.equal(
        m._fixed_pair_permutations(trial_ids, m.n_pairs),
        m._fixed_pair_permutations(trial_ids.clone(), m.n_pairs),
    )


def test_train_resampled_eval_fixed_switches_with_module_mode():
    m = _model("shuffle_pairs_train_resampled_eval_fixed")
    x = _input()
    trial_ids = _trial_ids()

    m.train()
    with torch.no_grad():
        train_a = m._compute_plv(x)
        train_b = m._compute_plv(x)
    assert not torch.equal(train_a, train_b)

    m.eval()
    with torch.no_grad():
        eval_a = m._compute_plv(x, trial_ids)
        eval_b = m._compute_plv(x, trial_ids)
    assert torch.equal(eval_a, eval_b)


def test_train_resampled_eval_fixed_preserves_value_multiset_in_both_modes():
    real = _model("real")
    controlled = _model("shuffle_pairs_train_resampled_eval_fixed")
    x = _input()
    trial_ids = _trial_ids()

    with torch.no_grad():
        reference = real._compute_plv(x).sort(dim=-1).values
        controlled.train()
        train_values = controlled._compute_plv(x).sort(dim=-1).values
        controlled.eval()
        eval_values = controlled._compute_plv(x, trial_ids).sort(dim=-1).values

    assert torch.allclose(reference, train_values, atol=1e-6)
    assert torch.allclose(reference, eval_values, atol=1e-6)


def test_fixed_trial_id_permutation_is_backend_independent():
    m = _model("shuffle_pairs_fixed")
    trial_ids = _trial_ids()
    expected = m._fixed_pair_permutations(trial_ids, m.n_pairs)

    if torch.cuda.is_available():
        actual = m._fixed_pair_permutations(trial_ids.cuda(), m.n_pairs)
        assert torch.equal(actual, expected)
    if torch.backends.mps.is_available():
        actual = m._fixed_pair_permutations(trial_ids.to("mps"), m.n_pairs)
        assert torch.equal(actual, expected)


def test_fixed_permutations_are_cached_by_global_trial_id():
    model = _model("shuffle_pairs_fixed")
    trial_ids = _trial_ids()
    first = model._fixed_pair_permutations(trial_ids, model.n_pairs)
    assert set(model._fixed_permutation_cache) == set(trial_ids.tolist())
    cached_objects = dict(model._fixed_permutation_cache)

    second = model._fixed_pair_permutations(trial_ids.flip(0), model.n_pairs)
    assert torch.equal(second, first.flip(0))
    assert all(
        model._fixed_permutation_cache[key] is value
        for key, value in cached_objects.items()
    )


def test_resampling_uses_a_dedicated_rng_stream():
    model = _model("shuffle_pairs_train_resampled_eval_fixed")
    x = _input()
    model.train()

    torch.manual_seed(8172)
    global_state_before = torch.random.get_rng_state().clone()
    with torch.no_grad():
        first = model._compute_plv(x)
        second = model._compute_plv(x)
    global_state_after = torch.random.get_rng_state()

    assert torch.equal(global_state_before, global_state_after)
    assert not torch.equal(first, second)

    if torch.cuda.is_available():
        cuda_model = _model("shuffle_pairs_train_resampled_eval_fixed").cuda()
        cuda_model.train()
        cuda_x = x.cuda()
        torch.cuda.manual_seed_all(9183)
        cuda_state_before = torch.cuda.get_rng_state().clone()
        with torch.no_grad():
            cuda_first = cuda_model._compute_plv(cuda_x)
            cuda_second = cuda_model._compute_plv(cuda_x)
        cuda_state_after = torch.cuda.get_rng_state()
        assert torch.equal(cuda_state_before, cuda_state_after)
        assert not torch.equal(cuda_first, cuda_second)


def test_resampling_stream_is_reproducible_for_the_same_run_seed():
    first_model = _model("shuffle_pairs_train_resampled_eval_fixed")
    second_model = _model("shuffle_pairs_train_resampled_eval_fixed")
    x = _input()
    first_model.train()
    second_model.train()

    with torch.no_grad():
        first = first_model._compute_plv(x)
        second = second_model._compute_plv(x)

    assert torch.equal(first, second)

    first_model.reset_resampling_stream(119)
    with torch.no_grad():
        reset_first = first_model._compute_plv(x)
    first_model.reset_resampling_stream(119)
    with torch.no_grad():
        reset_second = first_model._compute_plv(x)
    assert torch.equal(reset_first, reset_second)

    first_model.reset_resampling_stream(8819)
    one_batch = first_model._resampled_pair_permutations(
        4, first_model.n_pairs, torch.device("cpu")
    )
    first_model.reset_resampling_stream(8819)
    two_batches = torch.cat([
        first_model._resampled_pair_permutations(
            2, first_model.n_pairs, torch.device("cpu")
        ),
        first_model._resampled_pair_permutations(
            2, first_model.n_pairs, torch.device("cpu")
        ),
    ])
    assert torch.equal(one_batch, two_batches)


def test_full_forward_advances_dropout_rng_identically_across_schedules():
    fixed = _model("shuffle_pairs_fixed")
    resampled = _model("shuffle_pairs_train_resampled_eval_fixed")
    resampled.load_state_dict(fixed.state_dict())
    x = _input()
    trial_ids = _trial_ids()
    fixed.train()
    resampled.train()

    torch.manual_seed(771)
    with torch.no_grad():
        fixed(x, trial_ids=trial_ids)
    state_after_fixed = torch.random.get_rng_state().clone()

    torch.manual_seed(771)
    with torch.no_grad():
        resampled(x, trial_ids=trial_ids)
    state_after_resampled = torch.random.get_rng_state()

    assert torch.equal(state_after_fixed, state_after_resampled)

    if torch.cuda.is_available():
        fixed_cuda = _model("shuffle_pairs_fixed").cuda().train()
        resampled_cuda = _model(
            "shuffle_pairs_train_resampled_eval_fixed"
        ).cuda().train()
        resampled_cuda.load_state_dict(fixed_cuda.state_dict())
        cuda_x = x.cuda()
        cuda_ids = trial_ids.cuda()

        torch.cuda.manual_seed_all(881)
        with torch.no_grad():
            fixed_cuda(cuda_x, trial_ids=cuda_ids)
        cuda_state_after_fixed = torch.cuda.get_rng_state().clone()

        torch.cuda.manual_seed_all(881)
        with torch.no_grad():
            resampled_cuda(cuda_x, trial_ids=cuda_ids)
        cuda_state_after_resampled = torch.cuda.get_rng_state()
        assert torch.equal(cuda_state_after_fixed, cuda_state_after_resampled)
