#!/usr/bin/env python3
"""Rebind a physical transfer table to the current N=256 parent metadata."""
from pathlib import Path
import numpy as np

src = Path("/gpfs/kjhan/CF4/recon/linear_cr/v3_bgc_lg_peak_grafic_p3429_s5108_n576/transfer_p3429_s5108_n576.npz")
dst = Path("/gpfs/kjhan/CF4/zoom/transfer_parent_n256_v1.npz")
with np.load(src, allow_pickle=False) as d:
    payload = {k: d[k] for k in d.files}
payload["N"] = np.asarray(256, dtype=np.int64)
payload["box_hmpc"] = np.asarray(384.0, dtype=np.float64)
payload["kf"] = np.asarray(2.0 * np.pi / 384.0, dtype=np.float64)
payload["kNyq"] = np.asarray(np.pi * 256.0 / 384.0, dtype=np.float64)
dst.parent.mkdir(parents=True, exist_ok=True)
np.savez(dst, **payload)
print(dst)
