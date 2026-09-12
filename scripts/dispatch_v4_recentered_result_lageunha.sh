#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname | tr '[:upper:]' '[:lower:]')" != lageunha ]]; then
    echo "This dispatcher must run on Lageunha." >&2; exit 2
fi
repo=/home/kjhan/BACKUP/CF4
python=/home/kjhan/miniconda3/envs/circle/bin/python
root=/gpfs/kjhan/CF4/recon/linear_cr
p2_dir=$root/v3_bgc_lg_peak_p2_v4_recentered_search
marker=$p2_dir/RESEARCH_GATE_COMPLETE
log=$p2_dir/automatic_dispatch.log
exec > >(tee -a "$log") 2>&1
echo "[dispatch] waiting $(date -Is)"

while [[ ! -e "$marker" ]]; do
    if ! tmux has-session -t cf4_lg_v4_chain 2>/dev/null; then
        echo "[dispatch] v4 chain vanished before gate marker" >&2
        exit 1
    fi
    sleep 30
done

"$python" "$repo/src/cf4_select_recentered_candidate.py" \
    --preview "$p2_dir/recentered_p1_preview.json" \
    --p2-result "$p2_dir/p2_screen_result.json" --outdir "$p2_dir"

if [[ -s "$p2_dir/AUTO_PASS" ]]; then
    echo "[dispatch] survivor found; launching promotion $(date -Is)"
    tmux new-session -d -s cf4_lg_auto_promote \
        "CF4_P2_DIR='$p2_dir' CF4_CONDITIONED_P1_RESULT='$root/v3_bgc_lg_peak_p1_v4_recentered_search/p1_result.json' CF4_BATCH_LABEL=v4 bash '$repo/scripts/promote_recentered_candidate_lageunha.sh'"
    exit 0
fi

echo "[dispatch] no survivor; freezing unchanged-likelihood v5 batch $(date -Is)"
v5_config=$root/v3_bgc_lg_peak_v5_recentered_search_config.json
"$python" "$repo/src/cf4_create_next_peak_batch.py" \
    --base "$repo/config/p2_lg_peak_likelihood_v4_recentered_search.json" \
    --failed-preview "$p2_dir/recentered_p1_preview.json" \
    --seed-start 5141 --count 64 --label v5_recentered_search \
    --out "$v5_config"
tmux new-session -d -s cf4_lg_v5_chain \
    "bash '$repo/scripts/run_recentered_search_batch_lageunha.sh' v5_recentered_search '$v5_config'"
echo "[dispatch] v5 launched $(date -Is)"
