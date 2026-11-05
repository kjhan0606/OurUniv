#!/usr/bin/env bash
#SBATCH --job-name=cf4_R2_angray
#SBATCH --partition=h200
#SBATCH --gres=gpu:H200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
cd /home/kjhan/BACKUP/CF4
: "${SLURM_JOB_ID:?}" "${CF4_EXPECTED_COMMIT:?}" \
  "${CF4_EXPECTED_SOURCE_BUNDLE_SHA256:?}" "${CF4_R2_OUT_DIR:?}"
printf 'job_id=%s partition=%s requested_gres=%s slurm_gres=%s\n' \
  "$SLURM_JOB_ID" "${SLURM_JOB_PARTITION:-unknown}" \
  "${CF4_REQUESTED_GRES:-gpu:H200:1}" "${SLURM_JOB_GRES:-unknown}"
actual_commit=$(git rev-parse HEAD)
if [[ "$actual_commit" != "$CF4_EXPECTED_COMMIT" ]]; then
  printf 'source commit mismatch: expected %s, found %s\n' "$CF4_EXPECTED_COMMIT" "$actual_commit" >&2
  exit 2
fi
actual_bundle_sha256=$(sha256sum \
  src/cf4_2mpp_joint_likelihood_jax.py \
  src/cf4_r2_count_exposure.py \
  src/cf4_r2_shell_cdf_count.py \
  scripts/cf4_r2_v6_angular_ray_reference.py \
  scripts/run_cf4_r2_v6_angular_ray_reference.sbatch \
  tests/test_cf4_r2_shell_cdf_count.py \
  tests/test_cf4_r2_count_exposure.py \
  tests/test_cf4_r2_v6_pixel_angular_local.py | sha256sum | awk '{print $1}')
if [[ "$actual_bundle_sha256" != "$CF4_EXPECTED_SOURCE_BUNDLE_SHA256" ]]; then
  printf 'source bundle mismatch: expected %s, found %s\n' \
    "$CF4_EXPECTED_SOURCE_BUNDLE_SHA256" "$actual_bundle_sha256" >&2
  exit 3
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
python=/home/kjhan/miniconda3/envs/circle/bin/python3.11
"$python" -m unittest discover -s tests -p 'test_cf4_r2_v6_pixel_angular_local.py' -v
"$python" -m unittest discover -s tests -p 'test_cf4_r2_count_exposure.py' -v
"$python" -m unittest discover -s tests -p 'test_cf4_r2_shell_cdf_count.py' -v
"$python" \
  scripts/cf4_r2_v6_pixel_angular_setup.py
exec "$python" scripts/cf4_r2_v6_angular_ray_reference.py
