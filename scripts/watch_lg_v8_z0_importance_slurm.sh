#!/usr/bin/env bash
set -euo pipefail

job_id=${1:?usage: watch_lg_v8_z0_importance_slurm.sh JOB_ID}
root=/gpfs/kjhan/CF4/recon/linear_cr
status_dir=$root/v8_z0_importance_status
monitor_dir=$root/v8_z0_importance_monitor
terminal=$monitor_dir/terminal-$job_id.txt
mkdir -p "$monitor_dir"

while squeue -h -j "$job_id" | grep -q .; do
    state=$(squeue -h -j "$job_id" -o '%T|%M|%R')
    printf 'timestamp=%s job_id=%s state=%s\n' \
        "$(date -Is)" "$job_id" "$state" >"$monitor_dir/latest-$job_id.txt"
    sleep 30
done

state=$(sacct -n -X -j "$job_id" --format=State -P | sed -n '1{s/|//g;p}')
{
    echo "timestamp=$(date -Is)"
    echo "job_id=$job_id"
    echo "slurm_state=${state:-unknown}"
    if [[ -s "$status_dir/JOB_COMPLETE" ]]; then
        cat "$status_dir/JOB_COMPLETE"
    elif [[ -s "$status_dir/JOB_FAILED" ]]; then
        cat "$status_dir/JOB_FAILED"
    else
        echo "pipeline_marker=missing"
    fi
} >"$terminal"
