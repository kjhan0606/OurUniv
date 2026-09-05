#!/bin/bash
#SBATCH --job-name=cf4_unconstrained_p1_ref
#SBATCH --partition=a10,a40,h100,h200,a100,a100_pcie
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --chdir=/home/kjhan/BACKUP/CF4
#SBATCH --output=/home/kjhan/BACKUP/CF4/logs/unconstrained_p1_reference_v1-%j.log
#SBATCH --error=/home/kjhan/BACKUP/CF4/logs/unconstrained_p1_reference_v1-%j.log
#SBATCH --open-mode=append

# Historical resource record for completed job 326158.  The immutable executed
# version is commit a590a4d with grant commit 32c3e96.  V1 forbids resubmission;
# future execution must use a new, simple versioned program and grant.
set -euo pipefail
echo 'ARCHIVED_COMPLETED_V1_RESUBMISSION_FORBIDDEN' >&2
exit 64
