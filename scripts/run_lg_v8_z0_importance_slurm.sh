#!/usr/bin/env bash
#SBATCH --job-name=cf4_lg_v8
#SBATCH --partition=h200,h100,a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=/gpfs/kjhan/CF4/recon/linear_cr/v8_z0_importance_slurm-%j.out
#SBATCH --error=/gpfs/kjhan/CF4/recon/linear_cr/v8_z0_importance_slurm-%j.err

set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
python=/home/kjhan/miniconda3/envs/circle/bin/python
test_python=/home/kjhan/miniconda3/bin/python
root=/gpfs/kjhan/CF4/recon/linear_cr
program=$repo/config/p2_lg_z0_forward_importance_v8.json
proposal_dir=$root/v3_bgc_lg_peak_proposals_v8_z0_importance
projection_dir=$root/v3_bgc_lg_peak_parent_projections_v8_z0_importance
p1_dir=$root/v3_bgc_lg_peak_p1_v8_z0_importance
p2_dir=$root/v3_bgc_lg_peak_p2_v8_z0_importance
status_dir=$root/v8_z0_importance_status
manifest=$proposal_dir/lg_peak_proposals_manifest.json
projection_manifest=$projection_dir/parent_projection_manifest.json
p1_result=$p1_dir/p1_result.json
p2_config=$p2_dir/p2_targets_frozen.json
p2_result=$p2_dir/p2_screen_result.json
likelihood_result=$p2_dir/z0_importance_score.json
pair_input=$p2_dir/z0_likelihood_pair_recenter_input.json
preview=$p2_dir/z0_likelihood_recentered_p1.json
gate_result=$p2_dir/z0_importance_gate.json

declare -A expected_sha=(
    ["$program"]=6a89f5027f253282e18f21201146dde384837f0d689d725a25022def8ea7e6f2
    ["$repo/config/cf4_lg_v8_proposal_audit_result_record.json"]=31ea3f233c198773bad0a0f511d9781231b9fbff5de1b392efe9ac1596a86565
    ["$repo/src/cf4_lg_peak_cr.py"]=d6fb1d3fd0fba27aa8bb3a12aa0efb51e3e6f064db2b660be30d80798688e6a9
    ["$repo/src/cf4_lg_midpoint_proposal.py"]=24cb00fb681ae5c410760f378209b70f128290f38dc79975362995a53b28ab69
    ["$repo/src/cf4_lg_z0_likelihood.py"]=0fa110b3165de723896666b9249e81d81400d2b9a4a5968d108fecf6cb9d4f22
    ["$repo/src/cf4_lg_z0_importance.py"]=fde4384fee2ee39bea2fff7b967db6702eec9a3ad2c193d4eddd7537a57eb3f8
)

for path in "${!expected_sha[@]}"; do
    actual=$(sha256sum "$path" | awk '{print $1}')
    if [[ "$actual" != "${expected_sha[$path]}" ]]; then
        echo "Frozen V8 input SHA-256 mismatch: $path" >&2
        exit 2
    fi
done
for directory in "$proposal_dir" "$projection_dir" "$p1_dir" "$p2_dir"; do
    if [[ -d "$directory" ]] && find "$directory" -mindepth 1 -print -quit | grep -q .; then
        echo "V8 output already exists; refusing to overwrite: $directory" >&2
        exit 3
    fi
done

mkdir -p "$proposal_dir" "$projection_dir" "$p1_dir" "$p2_dir" "$status_dir"
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
    echo "program_sha256=${expected_sha[$program]}"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
} | tee "$status_dir/JOB_START"

export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
cd "$repo"

echo "[v8] validating frozen implementation $(date -Is)"
"$test_python" -m pytest -q \
    tests/test_cf4_lg_peak_cr.py \
    tests/test_cf4_lg_midpoint_proposal_audit.py \
    tests/test_cf4_lg_z0_likelihood.py \
    tests/test_cf4_lg_z0_importance.py \
    tests/test_cf4_lg_v8_program.py \
    tests/test_lg_v8_slurm_scripts.py

echo "[v8] generating 256 fresh defensive-mixture realizations $(date -Is)"
"$python" src/cf4_lg_peak_cr.py --config "$program"

echo "[v8] unchanged parent-resolution P1 gate $(date -Is)"
"$python" src/cf4_parent_p1.py --manifest "$projection_manifest" \
    --config config/p1_targets_v2_observer.json --outdir "$p1_dir"

echo "[v8] freezing unchanged hard-P2 bridge $(date -Is)"
"$python" src/cf4_prepare_p2_recentered_config.py \
    --proposal-manifest "$manifest" --conditioned-p1-result "$p1_result" \
    --outdir "$p2_dir" --out "$p2_config"

echo "[v8] N576 PM/FoF forward and unchanged hard-P2 screen $(date -Is)"
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
"$python" src/cf4_p2_screen.py \
    --p1-result "$root/v3_bgc_p1_observer_extension_v3/p1_result.json" \
    --config "$p2_config" --outdir "$p2_dir"

echo "[v8] normalized z=0 likelihood and exact p(q)/g(q) terms $(date -Is)"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
"$python" src/cf4_lg_z0_importance.py score \
    --program "$program" --proposal-manifest "$manifest" \
    --p2-result "$p2_result" --out "$likelihood_result" \
    --pair-out "$pair_input"

echo "[v8] unchanged five P1 gates at every likelihood-pair midpoint $(date -Is)"
"$python" src/cf4_p2_recenter_p1_preview.py \
    --p2-result "$pair_input" --conditioned-p1-result "$p1_result" \
    --p1-config config/p1_targets_v2_observer.json \
    --halo-directory "$p2_dir" --out "$preview"

echo "[v8] joint hard-P2/recentered-P1 importance gate $(date -Is)"
"$python" src/cf4_lg_z0_importance.py gate \
    --program "$program" --likelihood-result "$likelihood_result" \
    --recentered-p1-result "$preview" --p2-result "$p2_result" \
    --out "$gate_result"

status=$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$gate_result")
if [[ "$status" == complete_pass_waiting_RAMSSES_review ]]; then
    touch "$p2_dir/READY_FOR_RAMSSES_REVIEW"
    verdict=pass_waiting_RAMSSES_review
else
    touch "$p2_dir/V8_CLOSED_NO_RAMSES"
    verdict=failed_importance_or_joint_physical_gate
fi
printf 'timestamp=%s\njob_id=%s\nnode=%s\nverdict=%s\nRAMSES_launched=false\n' \
    "$(date -Is)" "${SLURM_JOB_ID:-unknown}" "$(hostname)" "$verdict" \
    >"$complete_marker"
trap - EXIT
echo "[v8] complete: $verdict; RAMSES remains stopped $(date -Is)"
