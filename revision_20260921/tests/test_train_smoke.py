"""Training smoke test.

Generates tiny synthetic .npz splits, writes a temporary YAML config, runs
the training entry point for 2 epochs, and asserts a checkpoint is written.

Run as a script:
    python -m tests.test_train_smoke

Or via pytest:
    pytest tests/test_train_smoke.py
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


def _have(*mods: str) -> bool:
    for m in mods:
        try:
            __import__(m)
        except ImportError:
            return False
    return True


def _write_synthetic_npz(path: Path, n_trials: int, n_channels: int, n_samples: int,
                         n_classes: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_trials, n_channels, n_samples)).astype(np.float32)
    # Create a weak class-conditional signal so loss can actually decrease.
    y = rng.integers(0, n_classes, size=n_trials).astype(np.int64)
    for i, label in enumerate(y):
        X[i, label % n_channels, :] += 2.0 * np.sin(
            np.linspace(0, 4 * np.pi * (label + 1), n_samples)
        )
    np.savez_compressed(path, X=X, y=y)


@unittest.skipUnless(_have("torch", "yaml"), "torch and pyyaml are required")
class TestTrainSmoke(unittest.TestCase):
    def setUp(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        self.repo_root = repo_root
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_train_runs_and_writes_checkpoint(self) -> None:
        import yaml

        n_channels, n_samples, n_classes = 22, 1000, 4

        train_npz = self.tmp_path / "train.npz"
        val_npz = self.tmp_path / "val.npz"
        _write_synthetic_npz(train_npz, n_trials=16, n_channels=n_channels,
                             n_samples=n_samples, n_classes=n_classes, seed=0)
        _write_synthetic_npz(val_npz, n_trials=8, n_channels=n_channels,
                             n_samples=n_samples, n_classes=n_classes, seed=1)

        ckpt_dir = self.tmp_path / "checkpoints"
        cfg = {
            "seed": 0,
            "dataset": {
                "train_path": str(train_npz),
                "val_path": str(val_npz),
                "sampling_rate": 250,
                "bandpass": [8.0, 30.0],
                "standardize": True,
            },
            "model": {
                "n_channels": n_channels,
                "n_classes": n_classes,
                "n_samples": n_samples,
                "temporal_kernel": 64,
                "F1": 4, "D": 2, "F2": 8,
                "dropout": 0.25,
            },
            "optimizer": {"lr": 1.0e-3, "weight_decay": 1.0e-4},
            "batch_size": 4,
            "epochs": 2,
            "checkpoint_dir": str(ckpt_dir),
        }
        cfg_path = self.tmp_path / "smoke.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))

        # Drive src.train.main() via sys.argv so we test the real entry point.
        from src import train

        argv_backup = sys.argv[:]
        sys.argv = ["train.py", "--config", str(cfg_path)]
        try:
            train.main()
        finally:
            sys.argv = argv_backup

        best = ckpt_dir / "best.pt"
        last = ckpt_dir / "last.pt"
        self.assertTrue(last.exists(), "last.pt was not written")
        # `best.pt` only appears if val_acc improved over 0.0; with random data
        # and 2 epochs that's not guaranteed, so just check `last`.

        import torch
        state = torch.load(last, map_location="cpu", weights_only=False)
        self.assertIn("model", state)
        self.assertIn("config", state)
        self.assertIn("val_acc", state)
        self.assertIsInstance(state["val_acc"], float)
        self.assertGreaterEqual(state["val_acc"], 0.0)
        self.assertLessEqual(state["val_acc"], 1.0)


@unittest.skipUnless(_have("torch"), "torch is required")
class TestModelForward(unittest.TestCase):
    """Sanity check: EEGNet forward pass produces correctly-shaped logits."""

    def test_forward_shape(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        import torch
        from src.models.eegnet import EEGNet

        m = EEGNet(n_channels=22, n_classes=4, n_samples=1000, sampling_rate=250)
        out = m(torch.randn(8, 22, 1000))
        self.assertEqual(tuple(out.shape), (8, 4))

    def test_forward_rejects_wrong_channel_count(self) -> None:
        import torch
        from src.models.eegnet import EEGNet

        m = EEGNet(n_channels=22)
        with self.assertRaises(ValueError):
            m(torch.randn(2, 10, 1000))

    def test_tiny_transformer_forward_shape(self) -> None:
        import torch
        from src.models.tiny_transformer import TinyTransformer

        m = TinyTransformer(n_channels=22, n_classes=4, n_samples=1000)
        out = m(torch.randn(8, 22, 1000))
        self.assertEqual(tuple(out.shape), (8, 4))

    def test_tiny_transformer_rejects_wrong_channel_count(self) -> None:
        import torch
        from src.models.tiny_transformer import TinyTransformer

        m = TinyTransformer(n_channels=22)
        with self.assertRaises(ValueError):
            m(torch.randn(2, 10, 1000))

    def test_tiny_transformer_rejects_indivisible_patch(self) -> None:
        from src.models.tiny_transformer import TinyTransformer

        with self.assertRaises(ValueError):
            TinyTransformer(n_samples=1000, patch_size=33)

    def test_phase_transformer_forward_shape(self) -> None:
        import torch
        from src.models.phase_transformer import PhaseTransformer

        m = PhaseTransformer(n_channels=22, n_classes=4, n_samples=1000)
        out = m(torch.randn(4, 22, 1000))
        self.assertEqual(tuple(out.shape), (4, 4))

    def test_phase_transformer_rejects_wrong_channel_count(self) -> None:
        import torch
        from src.models.phase_transformer import PhaseTransformer

        m = PhaseTransformer(n_channels=22)
        with self.assertRaises(ValueError):
            m(torch.randn(2, 10, 1000))

    def test_phase_transformer_lambda_init_zero(self) -> None:
        from src.models.phase_transformer import PhaseTransformer

        m = PhaseTransformer(n_channels=22, n_layers=3, n_heads=4)
        for layer_name, vals in m.lambda_summary().items():
            self.assertEqual(len(vals), 4)
            for v in vals:
                self.assertEqual(v, 0.0, f"{layer_name}: lambda should init at 0")

    def test_phase_term_changes_output(self) -> None:
        """Critical control: with the same weights, lambda=0 vs lambda!=0
        must produce different logits — confirms the phase term wires up."""
        import torch
        from src.models.phase_transformer import PhaseTransformer

        torch.manual_seed(0)
        m = PhaseTransformer(n_channels=22, n_classes=4, n_samples=1000, dropout=0.0)
        m.train(False)  # inference mode (avoids dropout RNG drift)
        x = torch.randn(4, 22, 1000)
        out_zero = m(x).clone()

        # Bump every lambda to a strong value.
        with torch.no_grad():
            for layer in m.layers:
                layer.attn.lam.fill_(2.0)
        out_bias = m(x).clone()

        # Outputs must differ — and not in some negligible last-decimal way.
        self.assertGreater((out_zero - out_bias).abs().max().item(), 1e-3)

    def test_phase_transformer_freeze_lam(self) -> None:
        """freeze_lam=True must keep lambdas at their init value through
        a backward pass — needed for the lambda=0 ablation control."""
        import torch
        from src.models.phase_transformer import PhaseTransformer

        m = PhaseTransformer(
            n_channels=22, n_classes=4, n_samples=1000,
            lam_init=0.0, freeze_lam=True, dropout=0.0,
        )
        x = torch.randn(2, 22, 1000, requires_grad=False)
        target = torch.tensor([0, 1])
        loss = torch.nn.functional.cross_entropy(m(x), target)
        loss.backward()

        for layer in m.layers:
            self.assertFalse(layer.attn.lam.requires_grad)
            self.assertTrue(torch.all(layer.attn.lam == 0.0))

    def test_phase_transformer_ct_forward_shape(self) -> None:
        import torch
        from src.models.phase_transformer_ct import PhaseTransformerCT

        m = PhaseTransformerCT(n_channels=22, n_classes=4, n_samples=1000)
        out = m(torch.randn(4, 22, 1000))
        self.assertEqual(tuple(out.shape), (4, 4))
        self.assertEqual(m.n_tokens, 22 * 20)  # patch_size=50 default

    def test_phase_transformer_ct_rejects_wrong_channels(self) -> None:
        import torch
        from src.models.phase_transformer_ct import PhaseTransformerCT

        m = PhaseTransformerCT(n_channels=22)
        with self.assertRaises(ValueError):
            m(torch.randn(2, 10, 1000))

    def test_phase_transformer_ct_phase_term_changes_output(self) -> None:
        """Same control as the v1 model: bumping lambdas off zero must
        produce different logits."""
        import torch
        from src.models.phase_transformer_ct import PhaseTransformerCT

        torch.manual_seed(0)
        m = PhaseTransformerCT(n_channels=22, n_samples=1000, dropout=0.0)
        m.train(False)
        x = torch.randn(4, 22, 1000)
        out_zero = m(x).clone()

        with torch.no_grad():
            for layer in m.layers:
                layer.attn.lam.fill_(2.0)
        out_bias = m(x).clone()

        self.assertGreater((out_zero - out_bias).abs().max().item(), 1e-3)

    def test_phase_transformer_ct_freeze_lam(self) -> None:
        import torch
        from src.models.phase_transformer_ct import PhaseTransformerCT

        m = PhaseTransformerCT(
            n_channels=22, n_samples=1000,
            lam_init=0.0, freeze_lam=True, dropout=0.0,
        )
        x = torch.randn(2, 22, 1000)
        target = torch.tensor([0, 1])
        loss = torch.nn.functional.cross_entropy(m(x), target)
        loss.backward()

        for layer in m.layers:
            self.assertFalse(layer.attn.lam.requires_grad)
            self.assertTrue(torch.all(layer.attn.lam == 0.0))

    def test_phase_eegnet_forward_shape(self) -> None:
        import torch
        from src.models.phase_eegnet import PhaseEEGNet

        m = PhaseEEGNet(n_channels=22, n_classes=4, n_samples=1000)
        out = m(torch.randn(8, 22, 1000))
        self.assertEqual(tuple(out.shape), (8, 4))

    def test_phase_eegnet_rejects_wrong_channels(self) -> None:
        import torch
        from src.models.phase_eegnet import PhaseEEGNet

        m = PhaseEEGNet(n_channels=22)
        with self.assertRaises(ValueError):
            m(torch.randn(2, 10, 1000))

    def test_phase_eegnet_phase_vs_shuffle_differs(self) -> None:
        """With identical seeds and weights, phase mode and shuffle mode
        must produce different logits — confirms the shuffle ablation
        actually changes the input signal seen by EEGNet."""
        import torch
        from src.models.phase_eegnet import PhaseEEGNet

        torch.manual_seed(0)
        m_phase = PhaseEEGNet(n_channels=22, phase_mode="phase", dropout=0.0)
        torch.manual_seed(0)
        m_shuf = PhaseEEGNet(n_channels=22, phase_mode="shuffle", dropout=0.0)
        m_phase.train(False)
        m_shuf.train(False)

        x = torch.randn(4, 22, 1000)
        diff = (m_phase(x) - m_shuf(x)).abs().max().item()
        self.assertGreater(diff, 1e-3)

    def test_phase_eegnet_rejects_bad_mode(self) -> None:
        from src.models.phase_eegnet import PhaseEEGNet

        with self.assertRaises(ValueError):
            PhaseEEGNet(n_channels=22, phase_mode="bogus")  # type: ignore[arg-type]

    def test_plv_eegnet_forward_shape(self) -> None:
        import torch
        from src.models.plv_eegnet import PLVEEGNet

        m = PLVEEGNet(n_channels=22, n_classes=4, n_samples=1000)
        out = m(torch.randn(8, 22, 1000))
        self.assertEqual(tuple(out.shape), (8, 4))
        self.assertEqual(m.n_pairs, 22 * 21 // 2)

    def test_plv_eegnet_rejects_wrong_channels(self) -> None:
        import torch
        from src.models.plv_eegnet import PLVEEGNet

        m = PLVEEGNet(n_channels=22)
        with self.assertRaises(ValueError):
            m(torch.randn(2, 10, 1000))

    def test_plv_values_in_unit_interval(self) -> None:
        """PLV must be in [0, 1] by definition."""
        import torch
        from src.models.plv_eegnet import PLVEEGNet

        m = PLVEEGNet(n_channels=22, plv_mode="real")
        x = torch.randn(4, 22, 1000)
        plv = m._compute_plv(x)
        self.assertGreaterEqual(plv.min().item(), 0.0)
        # Allow tiny float tolerance.
        self.assertLessEqual(plv.max().item(), 1.0 + 1e-5)

    def test_plv_zero_mode_is_zero(self) -> None:
        import torch
        from src.models.plv_eegnet import PLVEEGNet

        m = PLVEEGNet(n_channels=22, plv_mode="zero")
        x = torch.randn(4, 22, 1000)
        plv = m._compute_plv(x)
        self.assertTrue(torch.all(plv == 0.0))

    def test_plv_eegnet_real_vs_zero_differs(self) -> None:
        """Same arch + weights, different PLV input -> different output.
        Confirms the PLV branch actually flows through to the classifier."""
        import torch
        from src.models.plv_eegnet import PLVEEGNet

        torch.manual_seed(0)
        m_real = PLVEEGNet(n_channels=22, plv_mode="real", dropout=0.0)
        torch.manual_seed(0)
        m_zero = PLVEEGNet(n_channels=22, plv_mode="zero", dropout=0.0)
        m_real.train(False)
        m_zero.train(False)

        x = torch.randn(4, 22, 1000)
        diff = (m_real(x) - m_zero(x)).abs().max().item()
        self.assertGreater(diff, 1e-3)

    def test_plv_eegnet_rejects_bad_mode(self) -> None:
        from src.models.plv_eegnet import PLVEEGNet

        with self.assertRaises(ValueError):
            PLVEEGNet(n_channels=22, plv_mode="bogus")  # type: ignore[arg-type]

    def test_plv_shuffle_pairs_preserves_distribution(self) -> None:
        """shuffle_pairs must give the same sorted PLV values per trial,
        just in a different order. This is what makes it a valid
        capacity+distribution-controlled ablation."""
        import torch
        from src.models.plv_eegnet import PLVEEGNet

        torch.manual_seed(0)
        m_real = PLVEEGNet(n_channels=22, plv_mode="real")
        torch.manual_seed(0)
        m_shuf = PLVEEGNet(n_channels=22, plv_mode="shuffle_pairs")

        x = torch.randn(4, 22, 1000)
        plv_real = m_real._compute_plv(x)
        plv_shuf = m_shuf._compute_plv(x)

        # Sorted values must match per trial.
        self.assertTrue(torch.allclose(
            plv_real.sort(dim=-1).values, plv_shuf.sort(dim=-1).values, atol=1e-6
        ))
        # But unsorted order must differ for at least some trials.
        self.assertTrue((plv_real != plv_shuf).any())

    def test_plv_shuffle_pairs_is_fresh_each_call(self) -> None:
        """Each forward pass should get a fresh permutation — the model
        cannot rely on a fixed (mis)labeling to learn pair identity."""
        import torch
        from src.models.plv_eegnet import PLVEEGNet

        m = PLVEEGNet(n_channels=22, plv_mode="shuffle_pairs")
        x = torch.randn(4, 22, 1000)
        plv1 = m._compute_plv(x)
        plv2 = m._compute_plv(x)
        # Same sorted distribution, different order on at least some elements.
        self.assertTrue(torch.allclose(
            plv1.sort(dim=-1).values, plv2.sort(dim=-1).values, atol=1e-6
        ))
        self.assertTrue((plv1 != plv2).any())


if __name__ == "__main__":
    unittest.main(verbosity=2)
