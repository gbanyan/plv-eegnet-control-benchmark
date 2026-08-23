#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 DATASET SUBJECTS" >&2
    exit 2
fi

dataset=$1
subjects=$2
epochs=300
run_dir=results/controlled_permutation_schedule

if [[ ! -f "$run_dir/${dataset}.done" ]]; then
    echo "training sentinel is missing: $run_dir/${dataset}.done" >&2
    exit 1
fi

tags=(
    plv_eegnet_shuffle_pairs_train_resampled_eval_fixed_globaltrialid_exactrandperm_dedicatedrng
    plv_eegnet_shuffle_pairs_fixed_globaltrialid_exactrandperm_dedicatedrng
)
models=(
    plv_eegnet_shuffle_pairs_train_resampled_eval_fixed
    plv_eegnet_shuffle_pairs_fixed
)
labels=(resampled fixed)

for tag in "${tags[@]}"; do
    for seed in 42 43 44; do
        result="results/${dataset}_${tag}_seed${seed}_${epochs}ep_propersplit.json"
        checkpoint="results/checkpoints/${dataset}_${tag}_seed${seed}_${epochs}ep_propersplit"
        if [[ ! -f "$result" || ! -d "$checkpoint" ]]; then
            echo "incomplete training grid: $result or $checkpoint is missing" >&2
            exit 1
        fi
    done
done

for index in 0 1; do
    tag=${tags[$index]}
    model=${models[$index]}
    label=${labels[$index]}
    out="results/${dataset}_${label}_globaltrialid_exactrandperm_dedicatedrng_checkpoint_eval_mc20.json"
    log="$run_dir/${dataset}_${label}_globaltrialid_exactrandperm_dedicatedrng_checkpoint_eval_mc20.log"

    if [[ -e "$out" ]]; then
        echo "refusing to overwrite existing evaluation: $out" >&2
        exit 1
    fi

    .venv/bin/python -m scripts.evaluate_controlled_permutation_checkpoints \
        --dataset "$dataset" \
        --subjects "$subjects" \
        --seeds 42,43,44 \
        --draws 20 \
        --batch-size 64 \
        --checkpoint-template \
        "results/checkpoints/${dataset}_${tag}_seed{seed}_${epochs}ep_propersplit" \
        --training-model "$model" \
        --out "$out" \
        2>&1 | tee "$log"
done

touch "$run_dir/${dataset}.evaluation.done"
