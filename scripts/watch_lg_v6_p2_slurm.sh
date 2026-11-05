#!/usr/bin/env bash
set -euo pipefail

job_id=${1:?usage: watch_lg_v6_p2_slurm.sh JOB_ID}
interval=${CF4_WATCH_INTERVAL_SECONDS:-60}
root=/gpfs/kjhan/CF4/recon/linear_cr
p2_dir=$root/v3_bgc_lg_peak_p2_v6_latent_midpoint
status_dir=$root/v6_latent_midpoint_p2_resume_status
monitor_dir=$root/v6_latent_midpoint_p2_monitor
log=$monitor_dir/watch-${job_id}.log
terminal=$monitor_dir/terminal-${job_id}.txt

mkdir -p "$monitor_dir"
exec >>"$log" 2>&1

echo "timestamp=$(date -Is) event=monitor_start job_id=$job_id"
while squeue -h -j "$job_id" | grep -q .; do
    state=$(squeue -h -j "$job_id" -o '%T' | head -1)
    reason=$(squeue -h -j "$job_id" -o '%R' | head -1)
    elapsed=$(squeue -h -j "$job_id" -o '%M' | head -1)
    echo "timestamp=$(date -Is) state=$state elapsed=$elapsed reason=$reason"
    sleep "$interval"
done

state=$(sacct -X -n -P -j "$job_id" -o State | head -1 | cut -d'|' -f1)
exit_code=$(sacct -X -n -P -j "$job_id" -o ExitCode | head -1 | cut -d'|' -f1)
node=$(sacct -X -n -P -j "$job_id" -o NodeList | head -1 | cut -d'|' -f1)
{
    echo "timestamp=$(date -Is)"
    echo "job_id=$job_id"
    echo "state=${state:-unknown}"
    echo "exit_code=${exit_code:-unknown}"
    echo "node=${node:-unknown}"
} >"$terminal"

if [[ "$state" == COMPLETED* ]]; then
    if [[ ! -s "$status_dir/JOB_COMPLETE" ]]; then
        echo "verdict=inconsistent_completed_job_without_JOB_COMPLETE" >>"$terminal"
        exit 4
    fi
    if [[ -s "$p2_dir/READY_FOR_PROMOTION_REVIEW" && -s "$p2_dir/AUTO_PASS" ]]; then
        echo "verdict=survivor_ready_for_manual_promotion_review" >>"$terminal"
    elif [[ -e "$p2_dir/AUTOMATIC_BATCH_FAILED" && ! -e "$p2_dir/AUTO_PASS" ]]; then
        echo "verdict=no_recentered_survivor_stop_v6" >>"$terminal"
    else
        echo "verdict=inconsistent_terminal_markers" >>"$terminal"
        exit 4
    fi
else
    if [[ -s "$status_dir/JOB_FAILED" ]]; then
        echo "verdict=compute_failed_with_failure_marker" >>"$terminal"
    else
        echo "verdict=compute_not_completed_without_failure_marker" >>"$terminal"
    fi
fi

cat "$terminal"
