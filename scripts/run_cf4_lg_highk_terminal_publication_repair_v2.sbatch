#!/bin/bash
#SBATCH --job-name=cf4_terminal_publish_v2
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --chdir=/home/kjhan/BACKUP/CF4
#SBATCH --output=/gpfs/kjhan/CF4/recon/linear_cr/lg_highk_terminal_publication_repair_v2_slurm-%j.log
#SBATCH --error=/gpfs/kjhan/CF4/recon/linear_cr/lg_highk_terminal_publication_repair_v2_slurm-%j.log
#SBATCH --open-mode=truncate

set -Eeuo pipefail

readonly repo=/home/kjhan/BACKUP/CF4
readonly python=/home/kjhan/miniconda3/envs/circle/bin/python3.11
readonly config="$repo/config/cf4_lg_highk_terminal_publication_repair_program_v2.json"
readonly runner="$repo/src/cf4_lg_highk_terminal_publication_repair_v2.py"
readonly checker="$repo/scripts/check_cf4_lg_highk_terminal_publication_repair_v2.py"
readonly slurm_test="$repo/tests/test_cf4_lg_highk_terminal_publication_repair_slurm_v2.py"
readonly expected_config_sha=f93119f8efea564c970a25790b49060a6110919884bd10fbdf3405d755e86944
readonly expected_runner_sha=87b19b927345abd2e3713df50fd3b45717f687b9b4963798a009e9a90829a9c5
readonly expected_checker_sha=8877697a06bb5a9e3fbfc5b5b793217d23b30523c281d33bfd37e344a1e710ba
readonly expected_slurm_test_sha=a34a4f748d4ef5c0a26f4151208cf159b5c63e72e6a4e540aa7233bb013bf4de
readonly required_design_audit=DESIGN_GO
readonly required_code_audit=CODE_GO

[[ "$required_design_audit" == DESIGN_GO ]]
[[ "$required_code_audit" == CODE_GO ]]
[[ -x "$python" ]]
[[ "$(readlink -f "$python")" == "$python" ]]
"$python" -I -P - <<'PY'
import sys
import os
assert sys.executable == "/home/kjhan/miniconda3/envs/circle/bin/python3.11"
assert sys.version.split()[0] == "3.11.15"
assert os.environ["SLURM_JOB_PARTITION"] == "a10"
assert os.environ["SLURM_NNODES"] == "1"
assert os.environ["SLURM_NTASKS"] == "1"
assert os.environ["SLURM_CPUS_PER_TASK"] == "1"
assert os.environ.get("SLURM_GPUS", "") == ""
assert os.environ.get("SLURM_JOB_GPUS", "") == ""
assert os.environ.get("CUDA_VISIBLE_DEVICES", "") in {"", "NoDevFiles"}
PY
[[ "$(sha256sum "$config" | awk '{print $1}')" == "$expected_config_sha" ]]
[[ "$(sha256sum "$runner" | awk '{print $1}')" == "$expected_runner_sha" ]]
[[ "$(sha256sum "$checker" | awk '{print $1}')" == "$expected_checker_sha" ]]
[[ "$(sha256sum "$slurm_test" | awk '{print $1}')" == "$expected_slurm_test_sha" ]]

export PYTHONPATH="$repo/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1

"$python" -P "$runner" --config "$config" --lineage-preflight
"$python" -P "$runner" --config "$config" --test-only
exec "$python" -P "$runner" --config "$config"
