#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 DATASET SUBJECTS" >&2
    exit 2
fi

dataset=$1
subjects=$2
epochs=200
run_dir=results/controlled_permutation_schedule

models=(
    plv_eegnet_shuffle_pairs_train_resampled_eval_fixed
    plv_eegnet_shuffle_pairs_fixed
)
tags=(
    plv_eegnet_shuffle_pairs_train_resampled_eval_fixed_globaltrialid_exactrandperm_dedicatedrng
    plv_eegnet_shuffle_pairs_fixed_globaltrialid_exactrandperm_dedicatedrng
)

mkdir -p "$run_dir"

for index in 0 1; do
    model=${models[$index]}
    tag=${tags[$index]}
    for seed in 42 43 44; do
        out="results/${dataset}_${tag}_seed${seed}_${epochs}ep_propersplit.json"
        ckpt_dir="results/checkpoints/${dataset}_${tag}_seed${seed}_${epochs}ep_propersplit"
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
            --ckpt-dir "$ckpt_dir" \
            2>&1 | tee "$log"
        echo "[$(date -Iseconds)] completed dataset=$dataset model=$model seed=$seed"
    done
done

touch "$run_dir/${dataset}.done"
