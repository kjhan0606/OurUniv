#!/bin/bash
cd /home/kjhan/BACKUP/CF4
export CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
PY=/home/kjhan/miniconda3/envs/circle/bin/python
for sd in $(seq 1 24); do
  echo "=== seed $sd ($(date +%H:%M:%S)) ===" >> logs/screen3.log
  $PY -u src/cf4_lg_screen3.py --seed $sd >> logs/screen3.log 2>&1
done
$PY -u src/cf4_lg_screen3.py --rank-only >> logs/screen3.log 2>&1
echo SCREEN3_DONE >> logs/screen3.log
