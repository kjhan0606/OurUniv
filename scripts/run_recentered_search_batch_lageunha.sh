#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname | tr '[:upper:]' '[:lower:]')" != lageunha ]]; then
    echo "This search batch must run on Lageunha." >&2; exit 2
fi
if (( $# != 2 )); then
    echo "usage: $0 LABEL LIKELIHOOD_CONFIG" >&2; exit 2
fi

label=$1
likelihood_config=$2
repo=/home/kjhan/BACKUP/CF4
python=/home/kjhan/miniconda3/envs/circle/bin/python
root=/gpfs/kjhan/CF4/recon/linear_cr
read -r proposal_dir projection_dir p1_dir p2_dir < <(
    "$python" - "$likelihood_config" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]))["storage"]
print(s["proposal_directory"],s["parent_projection_directory"],
      s["p1_directory"],s["screen_directory"])
PY
)
manifest=$proposal_dir/lg_peak_proposals_manifest.json
projection_manifest=$projection_dir/parent_projection_manifest.json
p1_result=$p1_dir/p1_result.json
p2_config=$p2_dir/p2_targets_frozen.json
log=$root/${label}_automatic_chain.log

mkdir -p "$p1_dir" "$p2_dir"
exec > >(tee -a "$log") 2>&1
echo "[batch] $label start $(date -Is)"
cd "$repo"
export JAX_ENABLE_X64=True XLA_PYTHON_CLIENT_PREALLOCATE=false

# Lageunha has two 32-core sockets.  The concurrent FDM run is pinned to
# socket 0 (CPUs 0-31 with SMT siblings 64-95), so keep the CF4 host work on
# the disjoint physical cores and memory of socket/NUMA node 1.
cpu_node=(numactl --physcpubind=32-63 --membind=1)
echo "[batch] host affinity: NUMA1 physical CPUs 32-63"

"${cpu_node[@]}" "$python" src/cf4_lg_peak_cr.py --config "$likelihood_config"
"${cpu_node[@]}" "$python" src/cf4_parent_p1.py --manifest "$projection_manifest" \
    --config config/p1_targets_v2_observer.json --outdir "$p1_dir"
"$python" src/cf4_prepare_p2_recentered_config.py \
    --proposal-manifest "$manifest" --conditioned-p1-result "$p1_result" \
    --outdir "$p2_dir" --out "$p2_config"

export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.98
"${cpu_node[@]}" "$python" src/cf4_p2_screen.py \
    --p1-result "$root/v3_bgc_p1_observer_extension_v3/p1_result.json" \
    --config "$p2_config" --outdir "$p2_dir"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
"${cpu_node[@]}" "$python" src/cf4_p2_recenter_p1_preview.py \
    --p2-result "$p2_dir/p2_screen_result.json" \
    --conditioned-p1-result "$p1_result" \
    --p1-config config/p1_targets_v2_observer.json \
    --halo-directory "$p2_dir" --out "$p2_dir/recentered_p1_preview.json"
"$python" src/cf4_select_recentered_candidate.py \
    --preview "$p2_dir/recentered_p1_preview.json" \
    --p2-result "$p2_dir/p2_screen_result.json" --outdir "$p2_dir"

if [[ -s "$p2_dir/AUTO_PASS" ]]; then
    if [[ "${CF4_STOP_BEFORE_PROMOTION:-0}" == 1 ]]; then
        echo "[batch] survivor found; stopping before RAMSES for review"
        printf '%s survivor_ready_for_review\n' "$(date -Is)" \
            >"$p2_dir/READY_FOR_PROMOTION_REVIEW"
        exit 0
    fi
    echo "[batch] survivor found; starting automatic promotion"
    CF4_P2_DIR="$p2_dir" CF4_CONDITIONED_P1_RESULT="$p1_result" \
        CF4_BATCH_LABEL="$label" \
        exec bash "$repo/scripts/promote_recentered_candidate_lageunha.sh"
fi
echo "[batch] $label completed with zero recentered survivors $(date -Is)"
touch "$p2_dir/AUTOMATIC_BATCH_FAILED"
