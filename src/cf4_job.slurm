#!/bin/bash
#SBATCH -J cf4
#SBATCH -p a10,a40,a100,h100,h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -t 1-00:00:00
#SBATCH -o /home/kjhan/BACKUP/CF4/logs/%x_%j.out
#SBATCH -e /home/kjhan/BACKUP/CF4/logs/%x_%j.err
# Usage: sbatch cf4_job.slurm <script.py> [args...]
set -e
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export JAX_ENABLE_X64=1
PY=/home/kjhan/miniconda3/envs/circle/bin/python
cd /home/kjhan/BACKUP/CF4
echo "[slurm] node=$(hostname) gpu=$CUDA_VISIBLE_DEVICES job=$SLURM_JOB_ID"
echo "[slurm] running: $PY -u $*"
$PY -u "$@"
