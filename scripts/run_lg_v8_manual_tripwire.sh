#!/usr/bin/env bash
set -euo pipefail

job_id=${1:-301523}
node=${2:-syn101}
gpu_index=${3:-7}
resume_mode=${4:-fresh}
repo=/home/kjhan/BACKUP/CF4
root=/gpfs/kjhan/CF4/recon/linear_cr
attempt=$(date +%Y%m%dT%H%M%S)
monitor_dir=$root/v8_z0_importance_manual_monitor
attempt_dir=$monitor_dir/attempt-$attempt
pidfile=$attempt_dir/remote_process_group.pid
stdout=$attempt_dir/manual.out
stderr=$attempt_dir/manual.err
terminal=$attempt_dir/terminal.txt
held=false
complete=false

outputs=(
    "$root/v3_bgc_lg_peak_proposals_v8_z0_importance"
    "$root/v3_bgc_lg_peak_parent_projections_v8_z0_importance"
    "$root/v3_bgc_lg_peak_p1_v8_z0_importance"
    "$root/v3_bgc_lg_peak_p2_v8_z0_importance"
    "$root/v8_z0_importance_status"
)

mkdir -p "$attempt_dir"

release_original() {
    if [[ "$held" == true && "$complete" == false ]]; then
        scontrol release "$job_id" >/dev/null 2>&1 || true
    fi
}
trap release_original EXIT

quarantine_partial() {
    local quarantine=$root/v8_z0_importance_interrupted_$attempt
    local path
    mkdir -p "$quarantine"
    for path in "${outputs[@]}"; do
        if [[ -e "$path" ]]; then
            mv "$path" "$quarantine/"
        fi
    done
    printf 'timestamp=%s\nsource_attempt=%s\nrecoverable=true\n' \
        "$(date -Is)" "$attempt_dir" >"$quarantine/QUARANTINED_PARTIAL_RUN"
    echo "$quarantine"
}

state=$(squeue -h -j "$job_id" -o '%T')
if [[ "$state" != PENDING ]]; then
    echo "Original job $job_id is not pending: ${state:-absent}" >&2
    exit 2
fi
if [[ "$resume_mode" == resume ]]; then
    if [[ ! -s "${outputs[0]}/lg_peak_proposals_manifest.json" \
          || ! -s "${outputs[1]}/parent_projection_manifest.json" ]]; then
        echo "Resume mode requires the two completed generation manifests." >&2
        exit 2
    fi
    for path in "${outputs[@]:2}"; do
        if [[ -d "$path" ]] && find "$path" -mindepth 1 -print -quit | grep -q .; then
            echo "Canonical post-generation V8 output contains data: $path" >&2
            exit 2
        fi
    done
    resume_environment=CF4_V8_RESUME_AFTER_GENERATION=1
elif [[ "$resume_mode" == fresh ]]; then
    for path in "${outputs[@]}"; do
        if [[ -d "$path" ]] && find "$path" -mindepth 1 -print -quit | grep -q .; then
            echo "Canonical V8 output already contains data: $path" >&2
            exit 2
        fi
    done
    resume_environment=CF4_V8_RESUME_AFTER_GENERATION=0
else
    echo "Mode must be fresh or resume." >&2
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
gpu_uuid=$(ssh -o BatchMode=yes "$node" \
    "nvidia-smi -i $gpu_index --query-gpu=uuid --format=csv,noheader,nounits" \
    | tr -d '[:space:]')
initial_gpu_pids=$(ssh -o BatchMode=yes "$node" \
    "nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null | awk -F',[[:space:]]*' -v uuid='$gpu_uuid' '\$1 == uuid {print \$2}'")
if [[ -n "$initial_gpu_pids" ]]; then
    echo "GPU $node:$gpu_index already has compute PID(s): $initial_gpu_pids" >&2
    exit 2
fi

scontrol hold "$job_id"
held=true
if scontrol show node "$node" | grep -Eq 'AllocTRES=.+(cpu|gres/gpu)'; then
    echo "A Slurm allocation appeared before manual launch" >&2
    exit 75
fi
echo "timestamp=$(date -Is) event=original_job_held job_id=$job_id" | tee "$terminal"

ssh -o BatchMode=yes "$node" \
    "cd $repo && exec setsid env CUDA_VISIBLE_DEVICES=$gpu_index $resume_environment SLURM_CPUS_PER_TASK=32 SLURM_JOB_ID=manual-$job_id SLURM_JOB_PARTITION=manual-tripwire OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 NUMEXPR_NUM_THREADS=32 bash scripts/run_lg_v8_manual_remote.sh $pidfile" \
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

unauthorized_gpu_pids() {
    local pid owner
    local current_user
    current_user=$(id -un)
    while IFS= read -r pid; do
        [[ -n "$pid" ]] || continue
        owner=$(ssh -o BatchMode=yes "$node" \
            "ps -o user= -p $pid 2>/dev/null | tr -d '[:space:]'" || true)
        # A vanished PID is a normal nvidia-smi/ps race. All processes owned
        # by the same account are user-authorized and may coexist.
        [[ -z "$owner" || "$owner" == "$current_user" ]] || printf '%s\n' "$pid"
    done < <(ssh -o BatchMode=yes "$node" \
        "nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null | awk -F',[[:space:]]*' -v uuid='$gpu_uuid' '\$1 == uuid {print \$2}'")
}

tripped=false
while kill -0 "$ssh_pid" 2>/dev/null; do
    trip_reason=
    if scontrol show node "$node" | grep -Eq 'AllocTRES=.+(cpu|gres/gpu)'; then
        trip_reason=slurm_allocation_detected
    else
        unauthorized_pids=$(unauthorized_gpu_pids)
        if [[ -n "$unauthorized_pids" ]]; then
            trip_reason="unauthorized_gpu_pids_${unauthorized_pids//$'\n'/_}"
        fi
    fi
    if [[ -n "$trip_reason" ]]; then
        tripped=true
        echo "timestamp=$(date -Is) event=tripwire_$trip_reason" | tee -a "$terminal"
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
    quarantine=$(quarantine_partial)
    echo "timestamp=$(date -Is) verdict=manual_killed partial=$quarantine original_job_released=true" \
        | tee -a "$terminal"
    exit 75
fi
if (( run_rc != 0 )); then
    quarantine=$(quarantine_partial)
    echo "timestamp=$(date -Is) verdict=manual_compute_failed exit_code=$run_rc partial=$quarantine original_job_released=true" \
        | tee -a "$terminal"
    exit "$run_rc"
fi
if [[ ! -s "$root/v8_z0_importance_status/JOB_COMPLETE" ]]; then
    quarantine=$(quarantine_partial)
    echo "timestamp=$(date -Is) verdict=manual_missing_completion_marker partial=$quarantine original_job_released=true" \
        | tee -a "$terminal"
    exit 4
fi

scancel "$job_id"
complete=true
held=false
echo "timestamp=$(date -Is) verdict=manual_complete_original_job_cancelled RAMSES_launched=false" \
    | tee -a "$terminal"
