#!/usr/bin/env bash
#SBATCH --job-name=cf4_chunk_replay
#SBATCH --partition=h200
#SBATCH --gres=gpu:H200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=01:15:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
: "${CF4_EXPECTED_COMMIT:?}" "${CF4_R2_OUT_DIR:?}" "${CF4_GPU_MODE:?}"
CF4_REPLAY_ENTRY=${CF4_REPLAY_ENTRY:-scripts/cf4_r2_chunked_replay.py}
case "$CF4_REPLAY_ENTRY" in
  scripts/cf4_r2_chunked_replay.py|scripts/cf4_r2_broad_trial_replay.py|scripts/cf4_r2_conditional_hmc_pilot.py) ;;
  *) exit 2 ;;
esac
module purge
cd /home/kjhan/BACKUP/CF4
test "$(git rev-parse HEAD)" = "$CF4_EXPECTED_COMMIT"
git diff --quiet "$CF4_EXPECTED_COMMIT" -- src/ scripts/cf4_r2_conditional_map.py scripts/cf4_r2_prior_split_hmc.py "$CF4_REPLAY_ENTRY"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=src:scripts
exec /home/kjhan/miniconda3/envs/circle/bin/python3.11 "$CF4_REPLAY_ENTRY"
