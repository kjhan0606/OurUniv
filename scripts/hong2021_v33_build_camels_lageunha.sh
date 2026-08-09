#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
log_root=/gpfs/kjhan/IllustrisTNG/TNG100-1/derived/hong2021_v33/logs
mkdir -p "$log_root"
cd "$repo"
export PYTHONPATH=$repo/src

if [[ $(hostname -s) != lageunha ]]; then
    echo "V33 CAMELS reconstruction is frozen on lageunha" >&2
    exit 1
fi

pids=()
labels=()
for domain in SIMBA Swift; do
    for split in train validation; do
        label=$(printf '%s_%s' "$domain" "$split" | tr '[:upper:]' '[:lower:]')
        python -u src/hong2021_v33_kinematic_data.py \
            --program config/hong2021_v33_intrinsic_velocity_moment_program.json \
            --repo "$repo" --domain "$domain" --split "$split" \
            >"$log_root/${label}.log" 2>&1 &
        pids+=("$!")
        labels+=("$label")
    done
done

failed=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        echo "[v33] ${labels[$index]} complete"
    else
        echo "[v33] ${labels[$index]} failed" >&2
        failed=1
    fi
done
exit "$failed"
