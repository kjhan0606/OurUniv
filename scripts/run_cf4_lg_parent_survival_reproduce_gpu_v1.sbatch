#!/usr/bin/env bash
#SBATCH --job-name=cf4_lg_survive_repro
#SBATCH --partition=a100_pcie,a40,a100,h100,h200
#SBATCH --exclude=syn06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -Eeuo pipefail

root=/home/kjhan/BACKUP/CF4
search=/gpfs/kjhan/CF4/recon/linear_cr/parent_b3292_beta025_lg_pair_diagnostic_v3/result.json
base_config="$root/config/cf4_lg_highk_conditioning_b3292_v3_pair_diagnostic.json"
config_root=/gpfs/kjhan/CF4/recon/linear_cr/survival_calibration_v1_configs
output_root=/gpfs/kjhan/CF4/recon/linear_cr/parent_b3292_beta025_lg_survival_calibration_v1
mkdir -p "$config_root" "$output_root"

cd "$root"
export JAX_ENABLE_X64=True XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
for index in 124 129 242 248; do
  seed=$((40101 + index))
  tag=$(printf 'i%03d_s%d' "$index" "$seed")
  selected_config="$config_root/${tag}.json"
  output="$output_root/$tag"
  if [[ -s "$selected_config" && -s "$output/result.json" \
        && -s "$output/passing_field_000.npz" ]]; then
    printf 'LG_SURVIVAL_GPU_REPRODUCTION_SKIP_COMPLETE tag=%s\n' "$tag"
    continue
  fi
  if [[ -e "$selected_config" || -e "$output" ]]; then
    echo "refusing inconsistent partial candidate artifacts for $tag" >&2
    exit 3
  fi
  /home/kjhan/miniconda3/envs/circle/bin/python3.11 \
    scripts/cf4_select_and_reproduce_lg_environment_candidate.py \
    --search-result "$search" \
    --base-config "$base_config" \
    --search-index "$index" \
    --selected-config "$selected_config" \
    --output "$output"
done
printf 'LG_SURVIVAL_GPU_REPRODUCTION_PASS job=%s candidates=4\n' "$SLURM_JOB_ID"
