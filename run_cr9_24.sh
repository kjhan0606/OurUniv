#!/bin/bash
# Autonomous cr9-24 environment enrichment. Uses the SAME HR field construction as cr1-8 (NOT
# cf4_env_screen.py's power_complete, which over-evacuates the centre and is inconsistent).
# Two-stage per fable: cheap embed=1 env probe, then embed-fishing only on env-passers.
# Waits for cr3 fishing to release the GPU (one 576^3 forward at a time on the 32 GB Ada).
cd /home/kjhan/BACKUP/CF4
export CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
PY=/home/kjhan/miniconda3/envs/circle/bin/python
LOG=logs/cr9_24.log
TSV=recon/screen3e_rows.tsv
SMAP=recon/cf4_map_cf4_real192_hr.npz

echo "=== WAIT for cr3 fishing to release GPU $(date +%H:%M) ===" > $LOG
while ! grep -q "CR3FISH_DONE" logs/cr3fish.log 2>/dev/null; do sleep 60; done
echo "=== GPU free, START cr9-24 $(date +%H:%M) ===" >> $LOG
rm -f $TSV

# --- generate cr9-16 HR fields (reuse s_map; skip if present) ---
echo "--- PHASE gen cr9-16 $(date +%H:%M) ---" >> $LOG
for cs in $(seq 9 16); do
  if [ -f recon/cf4_map_cr${cs}.npz ]; then echo "cr$cs exists, skip" >> $LOG; continue; fi
  echo "  gen cr$cs $(date +%H:%M)" >> $LOG
  $PY -u src/cf4_explicit_map.py --N 192 --spacing 2.0 --real-npz data/cf4_clean.npz \
    --h 0.746 --A-s-1e9 1.63 --sig-floor 50 --vpec-max 3000 --chi2-target 0 --iters 70 \
    --cr-seed $cs --pc-seed -1 --load-smap-npz $SMAP --tag cr$cs --out recon >> $LOG 2>&1
done
echo "GEN_DONE $(date +%H:%M)" >> $LOG

# --- STAGE A: embed=1 environment probe on each cr9-16 ---
echo "--- STAGE A env probe (embed=1) $(date +%H:%M) ---" >> $LOG
for cs in $(seq 9 16); do
  RC=recon/cf4_map_cr${cs}.npz
  [ -f "$RC" ] || { echo "MISSING $RC" >> $LOG; continue; }
  $PY -u src/cf4_lg_screen3.py --recon $RC --cseed $cs --seed 1 --tsv $TSV >> $LOG 2>&1
done
echo "STAGEA_DONE $(date +%H:%M)" >> $LOG

# --- STAGE B: env-gate (infall 150-270, void 0.20-0.55, vOD>2.5), fish embed 2-24 on passers ---
# cols: 1cseed 2eseed 3npair 4virgoM 5infall 6void ... 15virgo_od
PASS=$(awk -F'\t' 'NR>1 && $5>=150 && $5<=270 && $6>=0.20 && $6<=0.55 && $15>2.5 {print $1}' $TSV | sort -un)
echo "--- STAGE B env-passers: $PASS $(date +%H:%M) ---" >> $LOG
for cs in $PASS; do
  RC=recon/cf4_map_cr${cs}.npz
  for es in $(seq 2 24); do
    $PY -u src/cf4_lg_screen3.py --recon $RC --cseed $cs --seed $es --tsv $TSV >> $LOG 2>&1
  done
done
echo "STAGEB_DONE $(date +%H:%M)" >> $LOG

$PY -u src/cf4_lg_screen3.py --rank-only --tsv $TSV >> $LOG 2>&1
echo "CR9_24_DONE $(date +%H:%M)" >> $LOG
