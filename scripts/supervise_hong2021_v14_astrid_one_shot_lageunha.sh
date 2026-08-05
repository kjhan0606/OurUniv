#!/usr/bin/env bash
set -euo pipefail

# Infrastructure failures may resume the exact same committed configuration;
# a terminal scientific pass/fail is never rerun.
repo=/home/kjhan/BACKUP/CF4
root=/gpfs/kjhan/CAMELS/Astrid/L25n256
seal=${1:-config/hong2021_v14_astrid_one_shot_seal.json}
state=$root/evaluation/hong2021_v14_astrid_one_shot/sequence_status.json
retry_seconds=${HONG2021_ASTRID_RETRY_SECONDS:-300}
if [[ ! $retry_seconds =~ ^[1-9][0-9]*$ ]]; then
    printf 'HONG2021_ASTRID_RETRY_SECONDS must be a positive integer\n' >&2
    exit 2
fi

cd "$repo"
while true; do
    if [[ -s $state ]]; then
        stage=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["stage"])' "$state")
        if [[ $stage == complete_* ]]; then
            printf '[terminal] %s\n' "$stage"
            exit 0
        fi
    fi
    set +e
    scripts/run_hong2021_v14_astrid_one_shot_lageunha.sh "$seal"
    result=$?
    set -e
    if [[ -s $state ]]; then
        stage=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["stage"])' "$state")
        if [[ $stage == complete_* ]]; then
            printf '[terminal] %s\n' "$stage"
            exit 0
        fi
    fi
    printf '[resume] exact configuration exited %s; retry in %ss\n' \
        "$result" "$retry_seconds"
    sleep "$retry_seconds"
done
