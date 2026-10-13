#!/bin/bash
#SBATCH --job-name=cf4_highk_terminal
#SBATCH --partition=a40,a100,h100,h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --chdir=/home/kjhan/BACKUP/CF4
#SBATCH --output=/gpfs/kjhan/CF4/recon/linear_cr/lg_highk_terminal_aggregation_v1_slurm-%j.log
#SBATCH --error=/gpfs/kjhan/CF4/recon/linear_cr/lg_highk_terminal_aggregation_v1_slurm-%j.log
#SBATCH --open-mode=truncate

set -Eeuo pipefail

readonly repo=/home/kjhan/BACKUP/CF4
readonly python=/home/kjhan/miniconda3/envs/circle/bin/python3.11
readonly config="$repo/config/cf4_lg_highk_terminal_aggregation_v1.json"
readonly runner="$repo/src/cf4_lg_highk_terminal_aggregation.py"
readonly checker="$repo/scripts/check_cf4_lg_highk_terminal_aggregation_v1.py"
readonly slurm_test="$repo/tests/test_cf4_lg_highk_terminal_aggregation_slurm.py"
readonly expected_config_sha=9c4d57705b7a423fbfc73898f0fee731e11a5aa87f0ab3e5603110b8c02eb213
readonly expected_runner_sha=c5647fb706b3791085080e431b6283e44c612d67a3d68cb45b1de7520836f3e5
readonly expected_checker_sha=e72a595c322ec10921a8435b650f07d82198ae0baab81e3824089d5a0d496aa4
readonly expected_slurm_test_sha=baccfd0b2cc76784b2eac3db0bda8f64eba04c574bd5908d45c159e764297007
readonly required_code_audit=CODE_GO

[[ "$required_code_audit" == CODE_GO ]]
[[ -x "$python" ]]
[[ "$(readlink -f "$python")" == "$python" ]]
"$python" -I -P - <<'PY'
import importlib.metadata as metadata
import jax
import pmwd
import sys
assert sys.executable == "/home/kjhan/miniconda3/envs/circle/bin/python3.11"
assert sys.version.split()[0] == "3.11.15"
assert metadata.version("jax") == "0.10.1"
assert metadata.version("jaxlib") == "0.10.1"
assert metadata.version("pmwd") == "0.1.dev131+g0552bc013"
assert jax.__file__ and pmwd.__file__
PY
[[ "$(sha256sum "$config" | awk '{print $1}')" == "$expected_config_sha" ]]
[[ "$(sha256sum "$runner" | awk '{print $1}')" == "$expected_runner_sha" ]]
[[ "$(sha256sum "$checker" | awk '{print $1}')" == "$expected_checker_sha" ]]
[[ "$(sha256sum "$slurm_test" | awk '{print $1}')" == "$expected_slurm_test_sha" ]]
[[ "$(git -C "$repo" rev-parse HEAD)" == "$(git -C "$repo" rev-parse '@{upstream}')" ]]

git -C "$repo" status --porcelain=v1 -z --untracked-files=all -- . \
    ':(exclude)scripts/tripwire/**' | "$python" -I -P -c 'import sys
entries=[row for row in sys.stdin.buffer.read().split(b"\0") if row]
if entries: raise SystemExit("worktree is not clean outside scripts/tripwire/: "+repr(entries))'

# The pinned Python lineage validator enforces baseline 0f74bcec (parent
# 9f4fa6c9), approved correction1 863792f4 (parent 0f74bcec), and runtime
# correction2 (parent 863792f4), with exact A/M rows and no renames.  It performs
# the equivalent strict probes
# `git rev-list --parents -n 1 HEAD`,
# `git diff-tree --root --no-commit-id --name-status -r HEAD`, and
# `git ls-tree HEAD` with required mode 100644.  Its baseline six paths are:
# config/cf4_lg_highk_terminal_aggregation_v1.json
# scripts/check_cf4_lg_highk_terminal_aggregation_v1.py
# scripts/run_cf4_lg_highk_terminal_aggregation_v1.sbatch
# src/cf4_lg_highk_terminal_aggregation.py
# tests/test_cf4_lg_highk_terminal_aggregation.py
# tests/test_cf4_lg_highk_terminal_aggregation_slurm.py
PYTHONPATH="$repo/src" "$python" -P -m cf4_lg_highk_terminal_aggregation \
    --config "$config" --lineage-preflight

mapfile -t visible_memory < <(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
[[ "${#visible_memory[@]}" -eq 1 ]]
(( visible_memory[0] >= 40960 ))

export PYTHONPATH="$repo/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

"$python" -P "$runner" --config "$config" --test-only
"$python" -P "$runner" --config "$config"
exec "$python" -P "$checker" --config "$config"
