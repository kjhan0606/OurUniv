#!/usr/bin/env bash
set -Eeuo pipefail
readonly repo=/home/kjhan/BACKUP/CF4
readonly python=/home/kjhan/miniconda3/envs/circle/bin/python
readonly module=cf4_aggregate_evidence_smc_execution_authorized_v6_open_pilot_execution
readonly branch=agent/freeze-zoom-pipeline
readonly receipt=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_receipts
readonly pilot=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_disposable_pilot
readonly data=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open
readonly state=/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_run

host=$(hostname); readonly host
short=${host%%.*}; readonly short
lower=$(LC_ALL=C tr '[:upper:]' '[:lower:]' <<<"$short"); readonly lower
[[ "$lower" == lageunha ]] || { printf 'host gate failed: %s\n' "$host" >&2; exit 69; }
[[ $(nproc) -ge 8 ]] || { printf 'pilot requires at least 8 CPUs\n' >&2; exit 65; }
available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo); readonly available_kib
[[ "$available_kib" -ge 67108864 ]] || { printf 'pilot requires at least 64 GiB available RAM\n' >&2; exit 65; }
for path in "$receipt" "$pilot" "$data" "$state"; do
  [[ ! -e "$path" ]] || { printf 'pilot one-shot path already exists: %s\n' "$path" >&2; exit 65; }
done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 MALLOC_ARENA_MAX=2
export PYTHONSAFEPATH=1 PYTHONPATH="$repo/src"
cd "$repo"
head=$(git rev-parse HEAD); readonly head
tracking=$(git rev-parse '@{upstream}'); readonly tracking
remote=$(git ls-remote origin "refs/heads/$branch" | awk '{print $1}'); readonly remote
[[ "$head" == "$tracking" && "$head" == "$remote" ]] || { printf 'git lineage is not synchronized\n' >&2; exit 65; }
git diff --quiet HEAD -- config src scripts tests || { printf 'tracked science paths are dirty\n' >&2; exit 65; }
git status --porcelain=v1 -z --untracked-files=all | PYTHONPATH= "$python" -I -P -c 'import sys
entries=[raw.decode("utf-8") for raw in sys.stdin.buffer.read().split(b"\0") if raw]
bad=[entry for entry in entries if not entry.startswith("?? scripts/tripwire/")]
if bad: raise SystemExit("unauthorized worktree entries: "+repr(bad))'

"$python" -P -m "$module" --preflight
exec "$python" -P -m "$module" --run
