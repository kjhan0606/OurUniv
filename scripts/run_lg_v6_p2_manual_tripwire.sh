#!/usr/bin/env bash
set -euo pipefail

job_id=${1:-276126}
node=${2:-syn101}
gpu_index=${3:-7}
repo=/home/kjhan/BACKUP/CF4
root=/gpfs/kjhan/CF4/recon/linear_cr
canonical_p2=$root/v3_bgc_lg_peak_p2_v6_latent_midpoint
attempt=$(date +%Y%m%dT%H%M%S)
manual_p2=$root/v3_bgc_lg_peak_p2_v6_manual_$attempt
manual_status=$root/v6_latent_midpoint_p2_manual_$attempt
monitor_dir=$root/v6_latent_midpoint_p2_manual_monitor
pidfile=$manual_status/remote_process_group.pid
stdout=$manual_status/manual.out
stderr=$manual_status/manual.err
terminal=$monitor_dir/terminal-$attempt.txt
held=false
promoted=false

mkdir -p "$manual_p2" "$manual_status" "$monitor_dir"

release_original() {
    if [[ "$held" == true && "$promoted" == false ]]; then
        scontrol release "$job_id" >/dev/null 2>&1 || true
    fi
}
trap release_original EXIT

state=$(squeue -h -j "$job_id" -o '%T')
if [[ "$state" != PENDING ]]; then
    echo "Original job $job_id is not pending: ${state:-absent}" >&2
    exit 2
fi
if scontrol show node "$node" | grep -Eq 'AllocTRES=.+(cpu|gres/gpu)'; then
    echo "Node $node already has allocated Slurm resources" >&2
    exit 2
fi
gpu_used=$(ssh -o BatchMode=yes "$node" \
    "nvidia-smi -i $gpu_index --query-gpu=memory.used --format=csv,noheader,nounits")
if (( gpu_used > 1024 )); then
    echo "GPU $node:$gpu_index is not idle: ${gpu_used} MiB used" >&2
    exit 2
fi

scontrol hold "$job_id"
held=true
echo "timestamp=$(date -Is) event=original_job_held job_id=$job_id" | tee "$terminal"

ssh -o BatchMode=yes "$node" \
    "cd $repo && exec setsid env CUDA_VISIBLE_DEVICES=$gpu_index CF4_V6_P2_DIR=$manual_p2 CF4_V6_P2_CONFIG=$canonical_p2/p2_targets_frozen.json CF4_V6_STATUS_DIR=$manual_status OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16 bash scripts/run_lg_v6_p2_manual_remote.sh $pidfile" \
    >"$stdout" 2>"$stderr" &
ssh_pid=$!

for _ in $(seq 1 30); do
    [[ -s "$pidfile" ]] && break
    kill -0 "$ssh_pid" 2>/dev/null || break
    sleep 1
done
if [[ ! -s "$pidfile" ]]; then
    wait "$ssh_pid" || true
    echo "timestamp=$(date -Is) verdict=remote_start_failed" | tee -a "$terminal"
    exit 3
fi
remote_pgid=$(tr -d '[:space:]' <"$pidfile")
echo "timestamp=$(date -Is) event=manual_start node=$node gpu=$gpu_index pgid=$remote_pgid" \
    | tee -a "$terminal"

tripped=false
while kill -0 "$ssh_pid" 2>/dev/null; do
    if scontrol show node "$node" | grep -Eq 'AllocTRES=.+(cpu|gres/gpu)'; then
        tripped=true
        echo "timestamp=$(date -Is) event=tripwire_slurm_allocation_detected" \
            | tee -a "$terminal"
        ssh -o BatchMode=yes "$node" "kill -TERM -- -$remote_pgid" >/dev/null 2>&1 || true
        sleep 2
        ssh -o BatchMode=yes "$node" "kill -KILL -- -$remote_pgid" >/dev/null 2>&1 || true
        break
    fi
    sleep 1
done

set +e
wait "$ssh_pid"
run_rc=$?
set -e
if [[ "$tripped" == true ]]; then
    echo "timestamp=$(date -Is) verdict=manual_killed_original_job_released" \
        | tee -a "$terminal"
    exit 75
fi
if (( run_rc != 0 )); then
    echo "timestamp=$(date -Is) verdict=manual_compute_failed exit_code=$run_rc" \
        | tee -a "$terminal"
    exit "$run_rc"
fi
if [[ ! -s "$manual_status/JOB_COMPLETE" ]]; then
    echo "timestamp=$(date -Is) verdict=manual_missing_completion_marker" \
        | tee -a "$terminal"
    exit 4
fi

exec 9>"$root/.v6_p2_manual_promotion.lock"
flock -n 9 || { echo "Could not acquire promotion lock" >&2; exit 4; }
if find "$canonical_p2" -maxdepth 1 -type f \
       ! -name p2_targets_frozen.json -print -quit | grep -q .; then
    echo "Canonical P2 directory changed during manual run" >&2
    exit 4
fi
for path in "$manual_p2"/*; do
    [[ -e "$path" ]] || continue
    mv "$path" "$canonical_p2/"
done
cp "$manual_status/JOB_START" "$manual_status/JOB_COMPLETE" \
    "$root/v6_latent_midpoint_p2_resume_status/"

tmux kill-session -t cf4-v6-watch >/dev/null 2>&1 || true
scancel "$job_id"
promoted=true
held=false
echo "timestamp=$(date -Is) verdict=manual_complete_promoted_original_job_cancelled" \
    | tee -a "$terminal"
