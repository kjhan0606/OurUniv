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
readonly expected_config_sha=24500fe5e538e4f69593f511f3bce264b8944b75a9a8409f199c5f71796c0c51
readonly expected_runner_sha=c86e278e02107c59aa813910c5adae4b70652c3cc2509ab98cd3a430b615d968
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

# The pinned Python lineage validator performs the equivalent strict probes
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
