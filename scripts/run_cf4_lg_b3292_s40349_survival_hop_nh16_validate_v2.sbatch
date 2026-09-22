#!/usr/bin/env bash
#SBATCH --job-name=cf4_s40349_h16val
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/gpfs/kjhan/CF4/logs/grammar_%x_%j.out
#SBATCH --error=/gpfs/kjhan/CF4/logs/grammar_%x_%j.err
set -Eeuo pipefail

root=/home/kjhan/BACKUP/CF4
hop=/gpfs/kjhan/CF4/ramses/sv1_grammar/i248_s40349/job_1104302/HOP_00001_nh16_v2
python=/home/kjhan/miniconda3/envs/circle/bin/python3.11
common=(
  --galaxyfinder /gpfs/kjhan/CF4/diagnostics/cf4_lg_b3292_i248_s40349_survival_v1.json
  --p1 "$root/config/p1_targets_v2_observer.json"
  --p2 "$root/config/p2_lg_targets_v11_parent_b3292_beta025_l9.json"
  --environment-policy standard-lg-isolation
)
"$python" "$root/scripts/cf4_hop_parent_environment.py" \
  --hop "$hop/raw00001.pos" "${common[@]}" \
  --output /gpfs/kjhan/CF4/diagnostics/cf4_lg_b3292_i248_s40349_hop_nh16_nomerge_v2.json
"$python" "$root/scripts/cf4_hop_parent_environment.py" \
  --hop "$hop/grp00001.pos" "${common[@]}" \
  --output /gpfs/kjhan/CF4/diagnostics/cf4_lg_b3292_i248_s40349_hop_nh16_regroup_v2.json
"$python" - <<'PY'
import json
from pathlib import Path

root = Path("/gpfs/kjhan/CF4/diagnostics")
raw = json.loads((root / "cf4_lg_b3292_i248_s40349_hop_nh16_nomerge_v2.json").read_text())
regrouped = json.loads((root / "cf4_lg_b3292_i248_s40349_hop_nh16_regroup_v2.json").read_text())
print(json.dumps({
    "nomerge_pair_mode": raw["environment"]["pair_crossmatch"]["mode"],
    "regroup_pair_mode": regrouped["environment"]["pair_crossmatch"]["mode"],
    "regroup_decision": regrouped["decision"],
}, sort_keys=True))
if regrouped["decision"] != "HOP_PARENT_STRUCTURE_PASS":
    raise SystemExit("N_hop=16 regrouped HOP did not independently resolve the LG pair")
PY
printf 'HOP_S40349_NH16_SCIENCE_GATE_PASS\n'
