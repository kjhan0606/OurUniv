#!/usr/bin/env bash
#SBATCH --job-name=cf4_v6_p2
#SBATCH --partition=h200,h100,a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/kjhan/CF4/recon/linear_cr/v6_p2_resume_slurm-%j.out
#SBATCH --error=/gpfs/kjhan/CF4/recon/linear_cr/v6_p2_resume_slurm-%j.err

set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
python=/home/kjhan/miniconda3/envs/circle/bin/python
root=/gpfs/kjhan/CF4/recon/linear_cr
proposal_manifest=$root/v3_bgc_lg_peak_proposals_v6_latent_midpoint/lg_peak_proposals_manifest.json
p1_result=$root/v3_bgc_lg_peak_p1_v6_latent_midpoint/p1_result.json
p2_dir=$root/v3_bgc_lg_peak_p2_v6_latent_midpoint
p2_config=$p2_dir/p2_targets_frozen.json
status_dir=$root/v6_latent_midpoint_p2_resume_status

declare -A expected_sha=(
    ["$proposal_manifest"]=f6834de385337a8e08e3a9c08a76d5be537f681da0bea138886b21d316a89114
    ["$p1_result"]=5ceef68e075df10212264e6f7d7ec0f4f6f70a99ca0782410ede1065e13a7a4e
    ["$p2_config"]=e45cbbe7aff03bf1c19ef015c5c1f3a13246b9cfa54b74e613061a88f86865b5
)

mkdir -p "$p2_dir" "$status_dir"
for path in "${!expected_sha[@]}"; do
    actual=$(sha256sum "$path" | awk '{print $1}')
    if [[ "$actual" != "${expected_sha[$path]}" ]]; then
        echo "Consumed v6 input SHA-256 mismatch: $path" >&2
        exit 2
    fi
done
if [[ -e "$p2_dir/p2_screen_result.json" || -e "$p2_dir/AUTO_PASS" \
      || -e "$p2_dir/AUTOMATIC_BATCH_FAILED" ]]; then
    echo "V6 P2 already has a final result; refusing duplicate execution." >&2
    exit 3
fi
if find "$p2_dir" -maxdepth 1 -type f \
       \( -name 'halos_*' -o -name 'result_*' \) -print -quit | grep -q .; then
    echo "Partial P2 member outputs exist; refusing an ambiguous resume." >&2
    exit 3
fi

failure_marker=$status_dir/JOB_FAILED
complete_marker=$status_dir/JOB_COMPLETE
rm -f "$failure_marker" "$complete_marker"
record_failure() {
    local rc=$?
    if (( rc != 0 )); then
        printf 'timestamp=%s\njob_id=%s\nnode=%s\nexit_code=%s\n' \
            "$(date -Is)" "${SLURM_JOB_ID:-unknown}" "$(hostname)" "$rc" \
            >"$failure_marker"
    fi
}
trap record_failure EXIT

{
    echo "timestamp=$(date -Is)"
    echo "job_id=${SLURM_JOB_ID:-unknown}"
    echo "node=$(hostname)"
    echo "partition=${SLURM_JOB_PARTITION:-unknown}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
    echo "repo_commit=$(git -C "$repo" rev-parse HEAD)"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
} | tee "$status_dir/JOB_START"

export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
cd "$repo"

echo "[v6-resume] N576 PM/FoF screen $(date -Is)"
"$python" src/cf4_p2_screen.py \
    --p1-result "$root/v3_bgc_p1_observer_extension_v3/p1_result.json" \
    --config "$p2_config" --outdir "$p2_dir"

echo "[v6-resume] recentered P1 gate for every eligible pair $(date -Is)"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
"$python" src/cf4_p2_recenter_p1_preview.py \
    --p2-result "$p2_dir/p2_screen_result.json" \
    --conditioned-p1-result "$p1_result" \
    --p1-config config/p1_targets_v2_observer.json \
    --halo-directory "$p2_dir" --out "$p2_dir/recentered_p1_preview.json"
"$python" src/cf4_select_recentered_candidate.py \
    --preview "$p2_dir/recentered_p1_preview.json" \
    --p2-result "$p2_dir/p2_screen_result.json" --outdir "$p2_dir"

if [[ -s "$p2_dir/AUTO_PASS" ]]; then
    printf '%s survivor_ready_for_review\n' "$(date -Is)" \
        >"$p2_dir/READY_FOR_PROMOTION_REVIEW"
    verdict=survivor_ready_for_review
else
    touch "$p2_dir/AUTOMATIC_BATCH_FAILED"
    verdict=no_recentered_survivor
fi
printf 'timestamp=%s\njob_id=%s\nnode=%s\nverdict=%s\n' \
    "$(date -Is)" "${SLURM_JOB_ID:-unknown}" "$(hostname)" "$verdict" \
    >"$complete_marker"
trap - EXIT
echo "[v6-resume] complete: $verdict $(date -Is)"
