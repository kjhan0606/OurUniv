#!/usr/bin/env bash
#SBATCH --job-name=cf4_lg_v6
#SBATCH --partition=h200,h100,a100,a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=/gpfs/kjhan/CF4/recon/linear_cr/v6_latent_midpoint_slurm-%j.out
#SBATCH --error=/gpfs/kjhan/CF4/recon/linear_cr/v6_latent_midpoint_slurm-%j.err

set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
python=/home/kjhan/miniconda3/envs/circle/bin/python
root=/gpfs/kjhan/CF4/recon/linear_cr
config=$repo/config/p2_lg_peak_likelihood_v6_latent_midpoint.json
expected_config_sha=afdc0eea25f82baaf2fabd2c024387ff62e650ed3962397142dc02327eb2e416
proposal_dir=$root/v3_bgc_lg_peak_proposals_v6_latent_midpoint
projection_dir=$root/v3_bgc_lg_peak_parent_projections_v6_latent_midpoint
p1_dir=$root/v3_bgc_lg_peak_p1_v6_latent_midpoint
p2_dir=$root/v3_bgc_lg_peak_p2_v6_latent_midpoint
manifest=$proposal_dir/lg_peak_proposals_manifest.json
projection_manifest=$projection_dir/parent_projection_manifest.json
p1_result=$p1_dir/p1_result.json
p2_config=$p2_dir/p2_targets_frozen.json
status_dir=$root/v6_latent_midpoint_slurm_status

mkdir -p "$p1_dir" "$p2_dir" "$status_dir"
if [[ "$(sha256sum "$config" | awk '{print $1}')" != "$expected_config_sha" ]]; then
    echo "Frozen v6 config SHA-256 mismatch." >&2
    exit 2
fi
if [[ -e "$p2_dir/p2_screen_result.json" || -e "$p2_dir/AUTO_PASS" \
      || -e "$p2_dir/AUTOMATIC_BATCH_FAILED" ]]; then
    echo "V6 has consumed outputs; refusing a duplicate run." >&2
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
    echo "config_sha256=$expected_config_sha"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
} | tee "$status_dir/JOB_START"

export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
cd "$repo"

echo "[v6] generating 64 latent-midpoint conditional realizations $(date -Is)"
"$python" src/cf4_lg_peak_cr.py --config "$config"

echo "[v6] parent-resolution P1 gate $(date -Is)"
"$python" src/cf4_parent_p1.py --manifest "$projection_manifest" \
    --config config/p1_targets_v2_observer.json --outdir "$p1_dir"

echo "[v6] freezing derived P2 config $(date -Is)"
"$python" src/cf4_prepare_p2_recentered_config.py \
    --proposal-manifest "$manifest" --conditioned-p1-result "$p1_result" \
    --outdir "$p2_dir" --out "$p2_config"

echo "[v6] N576 PM/FoF screen $(date -Is)"
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
"$python" src/cf4_p2_screen.py \
    --p1-result "$root/v3_bgc_p1_observer_extension_v3/p1_result.json" \
    --config "$p2_config" --outdir "$p2_dir"

echo "[v6] recentered P1 gate for every eligible pair $(date -Is)"
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
echo "[v6] complete: $verdict $(date -Is)"
