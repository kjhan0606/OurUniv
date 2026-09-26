#!/bin/bash
#SBATCH --job-name=cf4_lg_roles
#SBATCH --partition=a10,a40,a100_pcie
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/cf4_lg_roles_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/cf4_lg_roles_%j.err
set -euo pipefail
source /etc/profile.d/lmod.sh
module purge
module load gnu13/13.2.0
cd /home/kjhan/BACKUP/CF4
test -s /gpfs/kjhan/CF4/ramses/s40349_zoom_l19_z0_v1/job_1109268/FoF_Data/FoF.00003/GALCATALOG.LIST.00003
export PYTHONPATH="$PWD/src:$PWD/scripts"
exec /home/kjhan/miniconda3/bin/python scripts/cf4_lg_newgal_role_probe.py
