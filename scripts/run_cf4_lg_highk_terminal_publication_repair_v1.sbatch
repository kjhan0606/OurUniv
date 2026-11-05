#!/bin/bash
#SBATCH --job-name=cf4_terminal_publish
#SBATCH --partition=debug
#SBATCH --nodelist=grammar-debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --chdir=/home/kjhan/BACKUP/CF4
#SBATCH --output=/gpfs/kjhan/CF4/recon/linear_cr/lg_highk_terminal_publication_repair_v1_slurm-%j.log
#SBATCH --error=/gpfs/kjhan/CF4/recon/linear_cr/lg_highk_terminal_publication_repair_v1_slurm-%j.log
#SBATCH --open-mode=truncate

set -Eeuo pipefail

readonly repo=/home/kjhan/BACKUP/CF4
readonly python=/home/kjhan/miniconda3/envs/circle/bin/python3.11
readonly config="$repo/config/cf4_lg_highk_terminal_publication_repair_program_v1.json"
readonly runner="$repo/src/cf4_lg_highk_terminal_publication_repair.py"
readonly checker="$repo/scripts/check_cf4_lg_highk_terminal_publication_repair_v1.py"
readonly slurm_test="$repo/tests/test_cf4_lg_highk_terminal_publication_repair_slurm.py"
readonly expected_config_sha=e41616445a3e664df744f41f36c96b4356e3a0a8951966b039c81ed5efc1c5f8
readonly expected_runner_sha=a52204cef95369dd29e73a03f218a1411d5b79194baa4c57d14db239c00f2596
readonly expected_checker_sha=4f14ccc6f56bb35547ba96d92a435ee336d5be583b458e7ae6548479afb09f9e
readonly expected_slurm_test_sha=d297b41c9b0c4eb87416ed4b63681754c74e084502d73e49e3cbcfd46b0536a3
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
assert os.environ["SLURM_JOB_PARTITION"] == "debug"
assert os.environ["SLURMD_NODENAME"] == "grammar-debug"
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
