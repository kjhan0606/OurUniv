#!/bin/bash
# cr3 embed-fishing (best environment: Virgo 3.5e14, infall +203, void 0.42) -- try to land a
# near-observer 0.6-Mpc pair. Plus one pass on the 3 promoted survivors to capture virgo_od.
cd /home/kjhan/BACKUP/CF4
export CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
PY=/home/kjhan/miniconda3/envs/circle/bin/python
LOG=logs/cr3fish.log
TSV=recon/screen3d_rows.tsv
CR3=recon/cf4_map_cr3.npz
echo "=== CR3 FISH embed 25-60 + survivor vOD $(date +%H:%M) ===" > $LOG
rm -f $TSV
# survivors first (fast vOD capture on the promoted candidates)
$PY -u src/cf4_lg_screen3.py --recon recon/cf4_map_cr6.npz --cseed 6 --seed 19 --tsv $TSV >> $LOG 2>&1
$PY -u src/cf4_lg_screen3.py --recon recon/cf4_map_cr8.npz --cseed 8 --seed 5  --tsv $TSV >> $LOG 2>&1
$PY -u src/cf4_lg_screen3.py --recon $CR3            --cseed 3 --seed 1  --tsv $TSV >> $LOG 2>&1
$PY -u src/cf4_lg_screen3.py --recon recon/cf4_map_cf4_real192_hr.npz --cseed 1 --seed 13 --tsv $TSV >> $LOG 2>&1
echo "SURVIVORS_DONE $(date +%H:%M)" >> $LOG
# cr3 embed fishing
for es in $(seq 25 60); do
  $PY -u src/cf4_lg_screen3.py --recon $CR3 --cseed 3 --seed $es --tsv $TSV >> $LOG 2>&1
done
echo "FISH_DONE $(date +%H:%M)" >> $LOG
$PY -u src/cf4_lg_screen3.py --rank-only --tsv $TSV >> $LOG 2>&1
echo "CR3FISH_DONE $(date +%H:%M)" >> $LOG
