"""Read-only stationarity assessment for matched-TF partial-target chains."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cf4_datum_bearing_z0_phasec_pilot import effective_sample_size, rank_normalized_rhat

NAMES = ["logtarget", "IC_white_mean_square", "TF_shared_zero_white",
         "FP_training_group_white_mean"] + [f"count_tracer_white_{i}" for i in range(7)]


def correlation(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    aa, bb = a-a.mean(), b-b.mean()
    var_a, var_b = float(np.dot(aa, aa)), float(np.dot(bb, bb))
    return dict(pearson=float(np.dot(aa, bb)/np.sqrt(var_a*var_b))
                if var_a > 0 and var_b > 0 else None,
                RMS_difference=float(np.sqrt(np.mean((a-b)**2))),
                RMS_spatial=[float(np.sqrt(np.mean(aa**2))),
                             float(np.sqrt(np.mean(bb**2)))])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", choices=("matched_v2", "long_path_v3"),
                        default="matched_v2")
    args = parser.parse_args()
    source_name = ("r2_all_method_sampler_pilot_v3_long_path" if
                   args.pilot == "long_path_v3" else
                   "r2_all_method_sampler_pilot_v2_matched")
    SOURCE = Path("/gpfs/kjhan/CF4/z0_density") / source_name
    OUT = SOURCE / "assessment.json"
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm allocation required")
    if OUT.exists():
        raise FileExistsError(OUT)
    result = json.loads((SOURCE / "result.json").read_text())
    if result["status"] != "COMPLETE_PARTIAL_TARGET_CHAINS_NOT_YET_ASSESSED":
        raise ValueError("source partial-target chains incomplete")
    if result.get("pilot", "matched_v2") != args.pilot:
        raise ValueError("requested pilot does not match source target")
    warmup, retained = result["warmup_per_chain"], result["retained_per_chain"]
    if retained % 2 or retained < 32:
        raise ValueError("retained trace cannot be split")
    with np.load(SOURCE / "transition_summaries.npz", allow_pickle=False) as saved:
        chains = []
        for chain in range(2):
            phase = saved[f"chain{chain}_phase"]
            if phase.shape != (warmup+retained,) or not np.array_equal(
                    phase, np.r_[np.zeros(warmup), np.ones(retained)]):
                raise ValueError("warmup/retained phase length mismatch")
            summary = saved[f"chain{chain}_summary"][phase == 1]
            if summary.shape != (retained, 22) or not np.isfinite(summary).all():
                raise ValueError("retained scalar trace invalid")
            chains.append(np.column_stack((summary[:, :2], summary[:, 10],
                                           summary[:, 14], summary[:, -7:])))
        values = np.stack(chains)
    rows = []
    for index, name in enumerate(NAMES):
        x = values[:, :, index]
        first, last = x[:, :retained//2].mean(axis=1), x[:, retained//2:].mean(axis=1)
        rows.append(dict(name=name, chain_means=x.mean(axis=1).tolist(),
            first_half_means=first.tolist(), last_half_means=last.tolist(),
            half_shift_over_chain_sd=((last-first)/x.std(axis=1, ddof=1)).tolist(),
            rank_normalized_split_Rhat=rank_normalized_rhat(x),
            split_raw_ESS_approx=effective_sample_size(x)))
    blocks = []
    for chain in range(2):
        with np.load(SOURCE / f"chain{chain}_terminal_density_block24.npz", allow_pickle=False) as saved:
            block = saved["rho_block24"].copy()
        if block.shape != (16, 16, 16) or not np.isfinite(block).all():
            raise ValueError("terminal blocked density invalid")
        blocks.append(block)
    assessment = dict(classification="R2_MATCHED_TF_PARTIAL_TARGET_CHAIN_ASSESSMENT",
        source_job_id=result["job_id"], source=str(SOURCE), pilot=args.pilot,
        scalar=rows, terminal_block24_density=correlation(*blocks),
        terminal_maps_are_not_posterior_means=True,
        source_selection_and_covariance_calibrated=False,
        R2_posterior=False)
    OUT.write_text(json.dumps(assessment, indent=2, allow_nan=False)+"\n")
    print(json.dumps(assessment, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
