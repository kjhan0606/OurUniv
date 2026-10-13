"""Assess the two completed R2 development chains without rerunning gravity."""

import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cf4_datum_bearing_z0_phasec_pilot import (  # noqa: E402
    effective_sample_size,
    rank_normalized_rhat,
)

SOURCE = Path("/gpfs/kjhan/CF4/z0_density/r2_joint_nuisance_pilot_v2")
OUTPUT = SOURCE / "assessment.json"
NAMES = ["logtarget", "ic_mean_square"] + [f"tracer_white_{i}" for i in range(7)]


def scalar_assessment():
    with np.load(SOURCE / "transition_summaries.npz", allow_pickle=False) as data:
        summaries = []
        for chain in range(2):
            phase = data[f"chain{chain}_phase"]
            if phase.shape != (128,) or not np.array_equal(phase, np.r_[np.zeros(64), np.ones(64)]):
                raise ValueError(f"chain{chain} does not have the frozen 64+64 trace")
            summary = data[f"chain{chain}_summary"][phase == 1]
            if summary.shape[0] != 64 or not np.isfinite(summary).all():
                raise ValueError(f"chain{chain} retained trace invalid")
            summaries.append(np.column_stack((summary[:, :2], summary[:, -7:])))
        values = np.stack(summaries)
    rows = []
    for i, name in enumerate(NAMES):
        sample = values[:, :, i]
        first = sample[:, :32].mean(axis=1)
        last = sample[:, 32:].mean(axis=1)
        rows.append({
            "name": name,
            "chain_means": sample.mean(axis=1).tolist(),
            "first32_means": first.tolist(),
            "last32_means": last.tolist(),
            "half_shifts_over_chain_sd": ((last - first) / sample.std(axis=1, ddof=1)).tolist(),
            "rank_normalized_split_Rhat": rank_normalized_rhat(sample),
            "split_raw_ESS_approx": effective_sample_size(sample),
        })
    return rows


def field_metrics(left, right):
    a = left.astype(np.float64, copy=False).ravel()
    b = right.astype(np.float64, copy=False).ravel()
    aa, bb = a - a.mean(), b - b.mean()
    va, vb = float(np.dot(aa, aa)), float(np.dot(bb, bb))
    return {
        "pearson": float(np.dot(aa, bb) / np.sqrt(va * vb)) if va > 0 and vb > 0 else None,
        "rms_difference": float(np.sqrt(np.mean((a - b) ** 2))),
        "rms_chain0": float(np.sqrt(np.mean(aa ** 2))),
        "rms_chain1": float(np.sqrt(np.mean(bb ** 2))),
    }


def block_mean(array, width):
    n = array.shape[0]
    return array.reshape(n // width, width, n // width, width, n // width, width).mean(axis=(1, 3, 5))


def map_assessment():
    maps = []
    for chain in range(2):
        with np.load(SOURCE / f"chain{chain}_development_field_moments.npz", allow_pickle=False) as data:
            if int(data["sample_count"]) != 64 or data["mean"].shape != (4, 128, 128, 128):
                raise ValueError(f"chain{chain} map shape/count mismatch")
            maps.append((data["mean"].copy(), data["sample_sd"].copy()))
    (m0, s0), (m1, s1) = maps
    if not all(np.isfinite(a).all() for a in (m0, s0, m1, s1)):
        raise ValueError("nonfinite development map")
    axis = (np.arange(128) + .5 - 64.) * 3.
    x, y, z = axis[:, None, None], axis[None, :, None], axis[None, None, :]
    radius2 = x*x + y*y + z*z
    regions = {"LG_r15_cMpc_h": radius2 <= 15.**2,
               "survey_shell_r15_to180_cMpc_h": (radius2 > 15.**2) & (radius2 <= 180.**2)}
    report = {"density_mean_chain0": float(m0[0].mean()),
              "density_mean_chain1": float(m1[0].mean())}
    for name, mask in regions.items():
        report[name] = {
            "voxel_count": int(mask.sum()),
            "rho": field_metrics(m0[0, mask], m1[0, mask]),
            "rho_rms_within_chain_sample_sd": [float(np.sqrt(np.mean(s[0, mask]**2))) for s in (s0, s1)],
            "velocity_vector_rms_difference_km_s": float(np.sqrt(np.mean(np.sum((m0[1:, mask] - m1[1:, mask])**2, axis=0)))),
            "velocity_vector_rms_within_chain_sample_sd_km_s": [float(np.sqrt(np.mean(np.sum(s[1:, mask]**2, axis=0)))) for s in (s0, s1)],
        }
    for width in (4, 8):
        report[f"box_block_{width*3:g}_cMpc_h_rho"] = field_metrics(block_mean(m0[0], width), block_mean(m1[0], width))
    return report


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm allocation required")
    result = json.loads((SOURCE / "result.json").read_text())
    if result["status"] != "COMPLETE_SHORT_PARTIAL_MODEL_CHAINS_NOT_CONVERGED":
        raise ValueError("source chains not complete")
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    report = {"source_result": str(SOURCE / "result.json"),
              "source_job_id": result["job_id"],
              "status": "SHORT_CHAIN_DIAGNOSTIC_ONLY_NOT_POSTERIOR_CERTIFICATION",
              "scalar": scalar_assessment(), "maps": map_assessment()}
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
