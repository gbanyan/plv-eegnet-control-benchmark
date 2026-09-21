from __future__ import annotations

import numpy as np

from src.datasets.bci2a import BCI2aNPZ, trial_id_offset


def test_optional_trial_ids_use_a_declared_namespace(tmp_path) -> None:
    path = tmp_path / "trials.npz"
    np.savez(
        path,
        X=np.zeros((3, 2, 16), dtype=np.float32),
        y=np.asarray([0, 1, 0], dtype=np.int64),
    )
    dataset = BCI2aNPZ(
        path,
        bandpass=None,
        standardize=False,
        return_trial_id=True,
        trial_id_offset=trial_id_offset(7, evaluation=True),
    )

    _, _, first_id = dataset[0]
    _, _, last_id = dataset[2]
    assert first_id == 7 * 2 ** 40 + 2 ** 32
    assert last_id == 7 * 2 ** 40 + 2 ** 32 + 2


def test_trial_id_namespaces_are_unique_across_subjects_and_splits() -> None:
    namespaces = {
        trial_id_offset(subject, evaluation=evaluation)
        for subject in (1, 2, 54)
        for evaluation in (False, True)
    }
    assert len(namespaces) == 6
