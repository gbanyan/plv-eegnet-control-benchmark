from scripts.evaluate_controlled_permutation_checkpoints import (
    _fixed_assignment_key_sha256 as evaluator_hash,
)
from scripts.verify_controlled_permutation_artifacts import (
    _fixed_assignment_key_sha256 as verifier_hash,
)


def test_evaluator_and_independent_verifier_hash_the_same_assignment_keys() -> None:
    kwargs = {
        "dataset": "bci2a",
        "subject": 3,
        "seed": 42,
        "n_trials": 288,
        "n_pairs": 231,
    }
    assert evaluator_hash(**kwargs) == verifier_hash(**kwargs)


def test_assignment_key_hash_changes_with_design_inputs() -> None:
    base = evaluator_hash(
        dataset="bci2a", subject=3, seed=42, n_trials=288, n_pairs=231
    )
    changed = evaluator_hash(
        dataset="bci2a", subject=3, seed=43, n_trials=288, n_pairs=231
    )
    assert base != changed
