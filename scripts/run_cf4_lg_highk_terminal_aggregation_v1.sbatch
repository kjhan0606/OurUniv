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
readonly expected_config_sha=7da982f7cff64362500ec66ebb8ab21717779dbb9f17d6b3b5596021218fabaf
readonly expected_runner_sha=45cb9dc9a61ab96daa64824bec3c51e7e31a0a6dafeea5105667565a05cde2a0
readonly expected_checker_sha=e72a595c322ec10921a8435b650f07d82198ae0baab81e3824089d5a0d496aa4
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
[[ "$(git -C "$repo" rev-parse HEAD)" == "$(git -C "$repo" rev-parse '@{upstream}')" ]]

git -C "$repo" status --porcelain=v1 -z --untracked-files=all -- . \
    ':(exclude)scripts/tripwire/**' | "$python" -I -P -c 'import sys
entries=[row for row in sys.stdin.buffer.read().split(b"\0") if row]
if entries: raise SystemExit("worktree is not clean outside scripts/tripwire/: "+repr(entries))'

read -r -a commit_line <<<"$(git -C "$repo" rev-list --parents -n 1 HEAD)"
[[ "${#commit_line[@]}" -eq 2 ]]
readonly expected_commit_entries=$'A\tconfig/cf4_lg_highk_terminal_aggregation_v1.json\nA\tscripts/check_cf4_lg_highk_terminal_aggregation_v1.py\nA\tscripts/run_cf4_lg_highk_terminal_aggregation_v1.sbatch\nA\tsrc/cf4_lg_highk_terminal_aggregation.py\nA\ttests/test_cf4_lg_highk_terminal_aggregation.py\nA\ttests/test_cf4_lg_highk_terminal_aggregation_slurm.py'
[[ "$(git -C "$repo" diff-tree --root --no-commit-id --name-status -r HEAD | LC_ALL=C sort)" == "$expected_commit_entries" ]]
while IFS= read -r path; do
    [[ "$(git -C "$repo" ls-tree HEAD -- "$path" | awk '{print $1}')" == 100644 ]]
done <<<"$(printf '%s\n' "$expected_commit_entries" | cut -f2)"

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
