#!/bin/bash
cd /home/kjhan/BACKUP/CF4
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=0
PY=/home/kjhan/miniconda3/envs/circle/bin/python
$PY -u src/cf4_explicit_map.py --N 192 --spacing 2.0 --real-npz data/cf4_clean.npz \
  --h 0.746 --A-s-1e9 1.63 --sig-floor 50 --vpec-max 3000 --chi2-target 0 --iters 2 \
  --cr-seed 1 --pc-seed -1 --delta-constraints --dc-sigma 0.5 --tag dc_probe --out /tmp/dc_smoke \
  > logs/dc_probe.log 2>&1
echo PROBE_DONE >> logs/dc_probe.log
