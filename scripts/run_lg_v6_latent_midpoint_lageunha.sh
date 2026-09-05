#!/usr/bin/env bash
set -euo pipefail

if [[ "$(hostname | tr '[:upper:]' '[:lower:]')" != lageunha ]]; then
    echo "This one-shot bank must run on Lageunha." >&2
    exit 2
fi

repo=/home/kjhan/BACKUP/CF4
config=$repo/config/p2_lg_peak_likelihood_v6_latent_midpoint.json
screen=/gpfs/kjhan/CF4/recon/linear_cr/v3_bgc_lg_peak_p2_v6_latent_midpoint
wait_log=/gpfs/kjhan/CF4/recon/linear_cr/v6_latent_midpoint_wait.log

if [[ ! -r "$config" ]]; then
    echo "Missing frozen v6 config: $config" >&2
    exit 2
fi
if [[ -e "$screen/p2_screen_result.json" || -e "$screen/AUTO_PASS" \
      || -e "$screen/AUTOMATIC_BATCH_FAILED" ]]; then
    echo "V6 already has consumed forward outputs; refusing to overwrite." >&2
    exit 3
fi

mkdir -p "$(dirname "$wait_log")"
echo "$(date -Is) waiting for all RAMSES MPI work to leave Lageunha" \
    | tee -a "$wait_log"
while pgrep -f '[/]ramses[^ ]* .*\.nml' >/dev/null; do
    echo "$(date -Is) RAMSES active; v6 remains sealed" >>"$wait_log"
    sleep 60
done

echo "$(date -Is) RAMSES clear; launching frozen v6 on the Ada GPU" \
    | tee -a "$wait_log"
export CF4_STOP_BEFORE_PROMOTION=1
exec bash "$repo/scripts/run_recentered_search_batch_lageunha.sh" \
    v6_latent_midpoint "$config"
