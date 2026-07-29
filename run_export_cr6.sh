#!/bin/bash
cd /home/kjhan/BACKUP/CF4
export JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
PY=/home/kjhan/miniconda3/envs/circle/bin/python
LOG=logs/export_cr6.log
echo "=== GRAFIC export cr6 e19 (exact 576^3 field) $(date +%H:%M) ===" > $LOG
# 1) generate the EXACT field embed_ic(s_cr6, 576, 19) -- do NOT re-embed downstream (fable)
$PY -u - >> $LOG 2>&1 <<'PY'
import numpy as np, sys, os
sys.path.insert(0, "src")
from cf4_make_ic import embed_ic
z = np.load("recon/cf4_map_cr6.npz")
s = z["s_out"].astype(np.float64)
sf = embed_ic(s, 576, 19)
print(f"[gen] s576 shape={sf.shape} mean={sf.mean():.4f} std={sf.std():.4f}", flush=True)
np.save("recon/s_cr6_e19_576.npy", sf.astype(np.float32))
print(f"[gen] saved recon/s_cr6_e19_576.npy ({os.path.getsize('recon/s_cr6_e19_576.npy')/1e6:.0f} MB)", flush=True)
PY
# 2) export to GRAFIC1 (N=576, spacing 384/576=0.66667)
$PY -u src/cf4_export_grafic.py --s-npy recon/s_cr6_e19_576.npy --N 576 --spacing 0.666667 \
    --astart 0.02 --out recon/ic_cr6_e19 >> $LOG 2>&1
echo "EXPORT_DONE $(date +%H:%M)" >> $LOG
