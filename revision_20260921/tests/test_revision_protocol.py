"""Scientific invariants, evaluation isolation, and recovery for revision runs."""
import itertools

import numpy as np
import pytest
import torch

from scripts.evaluate_revision import EvaluationCache, evaluate_one, validate_seal, checked_probabilities
from scripts.revision_manifest import make_manifest
from scripts.summarize_revision import contrast_statistics, make_summary
from scripts.train_revision import TrainingCache, feature_mask, seeds_for, state_hash, train_one, seal_batch
from src.models.plv_eegnet import PLVEEGNet
from src.revision import CheckpointSelector, FeatureControl, FrozenStandardization, RevisionModel, plv_features
from src.train_all_subjects import DATASET_INFO


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def condition(mode="fixed", normalization="batchnorm", **kwargs):
    return {"mode": mode, "normalization": normalization, "band": [8., 30.], **kwargs}


def job(name="fixed"):
    return {"id": f"bci2b_s01_r1001_{name}", "batch": "test_bci2b", "dataset": "bci2b",
            "subject": 1, "replicate": 1001, "condition": name, "worker": "gblinux"}


def control(mode, values=None, **kwargs):
    values = torch.arange(30, dtype=torch.float32).reshape(5, 6) if values is None else values
    return FeatureControl(values, torch.arange(5) + 1000, torch.tensor([0, 1, 2]),
                          condition(mode, **kwargs), seeds_for(job()))


@pytest.mark.parametrize("band", [(8., 30.), (8., 13.), (13., 30.)])
def test_cached_plv_matches_submission(band):
    torch.manual_seed(1)
    x = torch.randn(7, 3, 1000)
    expected = PLVEEGNet(n_channels=3, n_classes=2, phase_band=band)._compute_plv(x)
    assert torch.allclose(plv_features(x, band), expected, atol=2e-6, rtol=2e-6)
    assert torch.allclose(torch.cat([plv_features(row, band) for row in x.split(3)]), expected,
                          atol=2e-6, rtol=2e-6)


def test_cached_forward_matches_original():
    model = RevisionModel(DATASET_INFO["bci2b"], condition("real"), 91).eval()
    torch.manual_seed(91)
    original = PLVEEGNet(n_channels=3, n_classes=2).eval()
    x = torch.randn(4, 3, 1000)
    with torch.no_grad():
        assert torch.allclose(model(x, plv_features(x)), original(x), atol=2e-6, rtol=2e-6)


def test_nonfinite_predictions_fail_instead_of_becoming_class_zero():
    with pytest.raises(RuntimeError, match="nonfinite"):
        checked_probabilities(torch.tensor([[float("nan"), 1.0]]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_cached_forward_and_dropout_pairing():
    spec = DATASET_INFO["bci2b"]
    model = RevisionModel(spec, condition("real"), 91).cuda().train()
    torch.manual_seed(91)
    original = PLVEEGNet(n_channels=3, n_classes=2).cuda().train()
    x = torch.randn(8, 3, 1000, device="cuda")
    features = plv_features(x)
    torch.cuda.manual_seed(991)
    expected = original(x)
    expected.square().mean().backward()
    torch.cuda.manual_seed(991)
    actual = model(x, features)
    actual.square().mean().backward()
    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)
    assert torch.allclose(model.classifier.weight.grad, original.classifier.weight.grad,
                          atol=2e-5, rtol=2e-5)


def test_edge_pipeline_constant_phase_difference():
    time = torch.arange(1000) / 250
    wave = torch.sin(2 * torch.pi * 11 * time)
    features = plv_features(torch.stack((wave, wave, -wave))[None], edge=True)
    assert torch.allclose(features, torch.ones_like(features), atol=1e-5)


def test_schedule_multiset_and_rng_isolation():
    fixed, resampled = control("fixed"), control("resampled")
    assert torch.equal(fixed.fixed, resampled.fixed)
    before = torch.get_rng_state().clone()
    resampled.begin_epoch(1)
    assert torch.equal(resampled.current, fixed.current)
    resampled.begin_epoch(2)
    assert torch.equal(before, torch.get_rng_state())
    assert not torch.equal(resampled.current[:3], fixed.current[:3])
    assert torch.equal(resampled.current.sort(1).values, fixed.current.sort(1).values)
    assert torch.equal(resampled.current[3:], fixed.current[3:])


def test_refresh_endpoints_and_reproducibility():
    held, iid = control("refresh", refresh_q=0.), control("refresh", refresh_q=1.)
    reference, replica = control("resampled"), control("refresh", refresh_q=1.)
    for epoch in range(1, 5):
        for obj in (held, iid, reference, replica):
            obj.begin_epoch(epoch)
        assert torch.equal(held.current, held.fixed)
        assert torch.equal(iid.current, reference.current)
        assert torch.equal(iid.current, replica.current)


def test_donor_pool_and_self_exclusion():
    obj = control("donor_resampled")
    for epoch in range(1, 8):
        obj.begin_epoch(epoch)
        for row in range(5):
            matches = (obj.values == obj.current[row]).all(1).nonzero().flatten().tolist()
            assert len(matches) == 1 and matches[0] in (0, 1, 2)
            assert row >= 3 or matches[0] != row
        assert torch.equal(obj.current[3:], obj.fixed[3:])


def test_frozen_calibration_orbit_and_training_only():
    obj = control("resampled")
    mean, variance = obj.calibration()
    samples = torch.stack([row[list(order)] for row in obj.values[:3]
                           for order in itertools.permutations(range(6))])
    assert torch.allclose(mean, samples.mean(0))
    assert torch.allclose(variance, samples.var(0, unbiased=False))
    changed = obj.values.clone()
    changed[3:] = 10000
    other = control("resampled", changed)
    assert all(torch.equal(a, b) for a, b in zip(obj.calibration(), other.calibration()))
    frozen = FrozenStandardization(6)
    frozen.mean.copy_(mean)
    frozen.variance.copy_(variance)
    frozen.train()
    frozen(torch.randn(5, 6)).sum().backward()
    assert torch.equal(mean, frozen.mean)
    assert frozen.weight.grad is not None


def test_normalization_shared_initialization():
    models = [RevisionModel(DATASET_INFO["bci2b"], condition(normalization=n), 918)
              for n in ("batchnorm", "layernorm", "none", "frozen", "bn_noaffine")]
    assert len({state_hash(m.eegnet.state_dict()) for m in models}) == 1
    assert len({state_hash(m.classifier.state_dict()) for m in models}) == 1


def test_masks_nested():
    small = feature_mask(condition(dimension=128, mask=1), "schirrmeister", 8128)
    large = feature_mask(condition(dimension=512, mask=1), "schirrmeister", 8128)
    assert torch.equal(small, large[:128])
    assert len(torch.unique(large)) == len(large)


def test_selector_freezes_primary_and_keeps_fixed_budget():
    selector = CheckpointSelector(minimum=2, patience=2, min_delta=.01)
    assert selector.update(1, 1., .4) == (True, True)
    selector.update(2, 1.1, .4)
    selector.update(3, 1.2, .5)
    assert selector.stop_epoch == 3 and not selector.finished(3, 5, 9)
    assert selector.update(4, .1, .8) == (False, True)
    assert selector.best_epoch == 1 and selector.finished(5, 5, 9)


def test_full_matrix_counts():
    manifest = make_manifest()
    assert len(manifest["jobs"]) == len({j["id"] for j in manifest["jobs"]}) == 10760
    assert manifest["job_counts_by_worker"] == {"gblinux": 7470, "gblab146": 3290}


def test_training_without_E_and_sealed_evaluation(tmp_path, monkeypatch):
    root = tmp_path / "data" / "bci2b"
    root.mkdir(parents=True)
    rng = np.random.default_rng(918)
    np.savez(root / "B01T.npz", X=rng.normal(size=(20, 3, 1000)).astype(np.float32), y=np.arange(20) % 2)
    manifest = make_manifest()
    manifest["jobs"] = [job()]
    manifest["batches"] = [{"name": "test_bci2b", "worker": "gblinux", "dataset": "bci2b", "phase": "test"}]
    manifest["training"].update(minimum_epochs=1, patience=1, maximum_epochs=3,
                                 fixed_budget={"bci2b": 2}, batch_size=8)
    manifest["evaluation"]["mc_draws"] = 3
    output = tmp_path / "run"
    original_load = np.load
    def guarded_load(path, *args, **kwargs):
        assert not str(path).endswith("E.npz"), "training attempted E access"
        return original_load(path, *args, **kwargs)
    with monkeypatch.context() as ctx:
        ctx.setattr(np, "load", guarded_load)
        cache = TrainingCache(tmp_path / "data", torch.device("cpu"))
        record = train_one(job(), manifest, cache, output)
    assert record["E_set_accessed"] is False and record["epochs_completed"] >= 2
    path = output / "jobs" / job()["id"] / "train.json"
    before = path.read_bytes()
    train_one(job(), manifest, cache, output)
    assert path.read_bytes() == before
    with pytest.raises(FileNotFoundError):
        validate_seal(manifest, "test_bci2b", output)
    seal_batch(manifest, "test_bci2b", output)
    _, seal_hash = validate_seal(manifest, "test_bci2b", output)
    np.savez(root / "B01E.npz", X=rng.normal(size=(8, 3, 1000)).astype(np.float32), y=np.arange(8) % 2)
    result = evaluate_one(job(), manifest, EvaluationCache(tmp_path / "data", torch.device("cpu")), output, seal_hash)
    predictions = original_load(output / "jobs" / job()["id"] / "predictions.npz")
    p = predictions["primary_fixed_probabilities"]
    assert np.allclose(p.sum(1), 1)
    assert result["metrics"]["primary"]["fixed_accuracy"] == np.mean(p.argmax(1) == predictions["labels"])
    assert predictions["primary_mc_probabilities"].shape == (3, 8, 2)
    with (output / "jobs" / job()["id"] / "primary.pt").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(RuntimeError, match="changed after seal"):
        validate_seal(manifest, "test_bci2b", output)


def test_subject_streams_and_condition_pairing():
    original = seeds_for(job())
    assert original == seeds_for(job("resampled"))
    assert len(set(original.values())) == len(original)
    assert original != seeds_for({**job(), "subject": 2})


def test_uncertainty_coverage_simulation():
    rng = np.random.default_rng(7331)
    for subject_sd, run_sd in ((0., .04), (.02, .04), (.05, .01)):
        covered = 0
        for _ in range(1000):
            d = .01 + rng.normal(0, subject_sd, (9, 1)) + rng.normal(0, run_sd, (9, 10))
            lo, hi = contrast_statistics(d, bootstrap=1)["model_95ci_pp"]
            covered += lo <= 1. <= hi
        assert .93 <= covered / 1000 <= .995


def test_incomplete_summary_has_no_contrasts(tmp_path):
    result = make_summary(make_manifest(), [tmp_path])
    assert result["contrasts"] == {} and not result["all_locked_jobs_evaluated"]


def test_refresh_displacement_exact_small_orbit():
    vector = np.array([.1, .4, .9])
    orbit = np.array(list(itertools.permutations(vector)))
    independent = np.mean(np.sum((orbit[:, None] - orbit[None, :])**2, axis=-1))
    for q in (0., .05, .2, 1.):
        for lag in (1, 3, 10):
            assert np.isclose((1-(1-q)**lag)*independent, 2*len(vector)*vector.var()*(1-(1-q)**lag))
