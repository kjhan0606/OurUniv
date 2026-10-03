#!/usr/bin/env bash
#SBATCH --job-name=cf4_transfer_rebind
#SBATCH --partition=a100_pcie,a40,a100,h100,h200
#SBATCH --exclude=syn06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/%x_%j.err
set -Eeuo pipefail
cd /home/kjhan/BACKUP/CF4
exec /home/kjhan/miniconda3/envs/circle/bin/python3.11 scripts/cf4_rebind_transfer_n256_v1.py
