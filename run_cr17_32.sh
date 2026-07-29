#!/bin/bash
# cr17-32 HUNT for cr6-likes: clean, isolated LG at the observer (fable: ~1/16 rate).
# NEW binding gate = near-observer cleanliness obs_big>4 (no >5e12 within 4 Mpc of observer),
# which auto-rejects the cr3/cr12 group contamination the old env probe was blind to.
# iters 45 for SCREENING only (env/cleanliness converge by iter ~20, chi2/N~1.02); the eventual
# winner is regenerated at iters 70 for the GRAFIC/GOTPM production field.
cd /home/kjhan/BACKUP/CF4
export CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
PY=/home/kjhan/miniconda3/envs/circle/bin/python
LOG=logs/cr17_32.log
TSV=recon/screen3f_rows.tsv
SMAP=recon/cf4_map_cf4_real192_hr.npz

echo "=== WAIT for GPU free $(date +%H:%M) ===" > $LOG
while pgrep -f "cf4_lg_screen3|cf4_explicit_map" >/dev/null 2>&1; do sleep 60; done
echo "=== GPU free, START cr17-32 hunt $(date +%H:%M) ===" >> $LOG
rm -f $TSV

# --- generate cr17-32 HR fields (iters 45, screening-only) ---
echo "--- gen cr17-32 (iters 45) $(date +%H:%M) ---" >> $LOG
for cs in $(seq 17 32); do
  if [ -f recon/cf4_map_cr${cs}.npz ]; then echo "cr$cs exists, skip" >> $LOG; continue; fi
  echo "  gen cr$cs $(date +%H:%M)" >> $LOG
  $PY -u src/cf4_explicit_map.py --N 192 --spacing 2.0 --real-npz data/cf4_clean.npz \
    --h 0.746 --A-s-1e9 1.63 --sig-floor 50 --vpec-max 3000 --chi2-target 0 --iters 45 \
    --cr-seed $cs --pc-seed -1 --load-smap-npz $SMAP --tag cr$cs --out recon >> $LOG 2>&1
done
echo "GEN_DONE $(date +%H:%M)" >> $LOG

# --- STAGE A: embed=1 env+cleanliness probe on each cr17-32 ---
echo "--- STAGE A probe (embed=1) $(date +%H:%M) ---" >> $LOG
for cs in $(seq 17 32); do
  RC=recon/cf4_map_cr${cs}.npz
  [ -f "$RC" ] || { echo "MISSING $RC" >> $LOG; continue; }
  $PY -u src/cf4_lg_screen3.py --recon $RC --cseed $cs --seed 1 --tsv $TSV >> $LOG 2>&1
done
echo "STAGEA_DONE $(date +%H:%M)" >> $LOG

# --- STAGE B: gate obs_big>4 (clean observer) AND vOD>2.5 AND infall[120,280] AND void[0.20,0.55] ---
# cols: 5infall 6void 15virgo_od 16obs_big
PASS=$(awk -F'\t' 'NR>1 && $16>4 && $15>2.5 && $5>=120 && $5<=280 && $6>=0.20 && $6<=0.55 {print $1}' $TSV | sort -un)
echo "--- STAGE B clean+env passers: $PASS $(date +%H:%M) ---" >> $LOG
for cs in $PASS; do
  RC=recon/cf4_map_cr${cs}.npz
  for es in $(seq 2 24); do
    $PY -u src/cf4_lg_screen3.py --recon $RC --cseed $cs --seed $es --tsv $TSV >> $LOG 2>&1
  done
done
echo "STAGEB_DONE $(date +%H:%M)" >> $LOG

$PY -u src/cf4_lg_screen3.py --rank-only --tsv $TSV >> $LOG 2>&1
echo "CR17_32_DONE $(date +%H:%M)" >> $LOG
