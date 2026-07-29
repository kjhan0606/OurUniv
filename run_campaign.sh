#!/bin/bash
cd /home/kjhan/BACKUP/CF4
export CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
PY=/home/kjhan/miniconda3/envs/circle/bin/python
LOG=logs/campaign.log
echo "=== PHASE A: generate cr_seeds 2-8 (reuse s_map) $(date +%H:%M) ===" > $LOG
for cs in 2 3 4 5 6 7 8; do
  if [ -f recon/cf4_map_cr${cs}.npz ]; then echo "cr$cs exists, skip" >> $LOG; continue; fi
  echo "--- gen cr$cs $(date +%H:%M) ---" >> $LOG
  $PY -u src/cf4_explicit_map.py --N 192 --spacing 2.0 --real-npz data/cf4_clean.npz \
    --h 0.746 --A-s-1e9 1.63 --sig-floor 50 --vpec-max 3000 --chi2-target 0 --iters 70 \
    --cr-seed $cs --pc-seed -1 --load-smap-npz recon/cf4_map_cf4_real192_hr.npz \
    --tag cr$cs --out recon >> $LOG 2>&1
done
echo "GEN_DONE $(date +%H:%M)" >> $LOG
echo "=== PHASE B: screen cr{1-8} x embed{1-10} $(date +%H:%M) ===" >> $LOG
rm -f recon/screen3b_rows.tsv
for cs in 1 2 3 4 5 6 7 8; do
  if [ $cs -eq 1 ]; then RC=recon/cf4_map_cf4_real192_hr.npz; else RC=recon/cf4_map_cr${cs}.npz; fi
  [ -f "$RC" ] || { echo "MISSING $RC, skip cr$cs" >> $LOG; continue; }
  for es in $(seq 1 10); do
    $PY -u src/cf4_lg_screen3.py --recon $RC --cseed $cs --seed $es >> $LOG 2>&1
  done
done
echo "SCREEN_DONE $(date +%H:%M)" >> $LOG
$PY -u src/cf4_lg_screen3.py --rank-only >> $LOG 2>&1
echo "CAMPAIGN_DONE $(date +%H:%M)" >> $LOG
