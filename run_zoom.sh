#!/bin/bash
cd /home/kjhan/BACKUP/CF4
export CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
PY=/home/kjhan/miniconda3/envs/circle/bin/python
$PY -u src/cf4_zoom_trace.py --seed 22 --rsel 5.0 > logs/zoom_trace.log 2>&1
echo ZOOM_DONE >> logs/zoom_trace.log
