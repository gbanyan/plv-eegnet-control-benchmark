#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 DATASET SUBJECTS EPOCHS" >&2
    exit 2
fi

dataset=$1
subjects=$2
epochs=$3
run_dir=results/targeted_dsp_controls

models=(
    plv_eegnet_shuffle_pairs_global_fixed
    plv_eegnet_shuffle_pairs_fixed_no_norm
    plv_eegnet_shuffle_pairs_train_resampled_eval_fixed_no_norm
)
tags=(
    global_fixed
    trial_fixed_no_norm
    train_resampled_eval_fixed_no_norm
)

mkdir -p "$run_dir"

for index in 0 1 2; do
    model=${models[$index]}
    tag=${tags[$index]}
    for seed in 42 43 44; do
        out="results/${dataset}_${tag}_seed${seed}_${epochs}ep_propersplit.json"
        log="$run_dir/${dataset}_${tag}_seed${seed}.log"

        if [[ -e "$out" ]]; then
            echo "refusing to overwrite existing result: $out" >&2
            exit 1
        fi

        echo "[$(date -Iseconds)] starting dataset=$dataset model=$model seed=$seed subjects=$subjects"
        .venv/bin/python -m src.train_all_subjects \
            --dataset "$dataset" \
            --model "$model" \
            --subjects "$subjects" \
            --epochs "$epochs" \
            --batch-size 64 \
            --seed "$seed" \
            --out "$out" \
            2>&1 | tee "$log"
        echo "[$(date -Iseconds)] completed dataset=$dataset model=$model seed=$seed"
    done
done

touch "$run_dir/${dataset}.done"
