#!/bin/bash
cd /home/kjhan/BACKUP/CF4
export CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
PY=/home/kjhan/miniconda3/envs/circle/bin/python
$PY -u src/cf4_summary_viz.py --seed 22 > logs/viz.log 2>&1
echo VIZ_DONE >> logs/viz.log
