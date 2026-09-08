#!/usr/bin/env bash
#SBATCH --job-name=cf4_C_flow_context
#SBATCH --partition=a40,a100,h100,h200
#SBATCH --exclude=syn06
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?}" "${EXPECTED_COMMIT:?}"
cd /home/kjhan/BACKUP/CF4
git diff --quiet "$EXPECTED_COMMIT" -- src scripts config tests
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src:scripts:tests
/home/kjhan/miniconda3/bin/python3.13 -m unittest test_cf4_split_flow -v
/home/kjhan/miniconda3/bin/python3.13 scripts/cf4_bundle_c_flow_checkpoint_audit.py
# Only exit0 (paired evidence plus inverse/Jacobian checks) permits this repair.
export CF4_FLOW_OUTPUT_NAME=conditional_flow_v2_full_context
export CF4_FLOW_CONFIG=config/cf4_bundle_c_flow_full_context_v2.json
exec /home/kjhan/miniconda3/bin/python3.13 scripts/cf4_bundle_c_flow_pilot.py
