#!/bin/bash
cd /home/kjhan/BACKUP/CF4
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=0
PY=/home/kjhan/miniconda3/envs/circle/bin/python
$PY -u src/cf4_explicit_map.py --N 144 --spacing 2.667 --real-npz data/cf4_clean.npz \
  --h 0.746 --A-s-1e9 1.63 --sig-floor 50 --vpec-max 3000 --chi2-target 0 --iters 70 \
  --cr-seed 1 --pc-seed -1 --delta-constraints --dc-sigma 0.5 --dc-R 5.0 \
  --tag cf4_real144_dc --out recon > logs/dc_real.log 2>&1
echo "MAP_DC_DONE" >> logs/dc_real.log
$PY -u src/cf4_obs_compare2.py --recon recon/cf4_map_cf4_real144_dc.npz \
  --out recon/cf4_obs_compare2_dc.png >> logs/dc_real.log 2>&1
echo "DC_ALL_DONE" >> logs/dc_real.log
