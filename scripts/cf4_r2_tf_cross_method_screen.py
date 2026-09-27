"""Source-only CF4 TF/other-method consistency screen, not calibration."""

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/cf4_groups.csv"
CATALOGUE = Path("/gpfs/kjhan/CF4/z0_density/r2_tf_source_link_v1/tf_only_groups.npz")
OUT = Path("/gpfs/kjhan/CF4/z0_density/r2_tf_cross_method_screen_v1")
SOURCE_SHA = "bfdc0cfc0f172b48468e3a8fd05e87978c1ec68c341fb2d929fc1200f0123334"
METHODS = (("FP", "o_DMfp", "DMfp", "e_DMfp"),
           ("SNIa", "o_DMsnIa", "DMsnIa", "e_DMsnIa"),
           ("SBF_optical", "o_DMsbfo", "DMsbfo", "e_DMsbfo"),
           ("SBF_infrared", "o_DMsbfi", "DMsbfi", "e_DMsbfi"))


def summary(values):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return {"n": 0}
    median = float(np.median(values))
    return {"n": len(values), "median": median,
            "median_abs_deviation": float(np.median(np.abs(values-median))),
            "p10_p90": [float(x) for x in np.percentile(values, [10, 90])]}


def record(rows):
    delta, norm, cz, b, count = (np.asarray([row[k] for row in rows], dtype=float)
                                for k in ("delta", "normalized", "cz", "SGB", "n_tf"))
    result = {"delta_mag": summary(delta), "delta_over_formal_independent_error": summary(norm),
              "abs_delta_over_formal_error_gt_3_fraction": float(np.mean(np.abs(norm) > 3)),
              "TF_member_count": summary(count)}
    # Predeclared broad splits expose gross source/population dependencies.
    result["cz_below_7500"] = summary(delta[cz < 7500])
    result["cz_at_least_7500"] = summary(delta[cz >= 7500])
    result["supergalactic_north"] = summary(delta[b >= 0])
    result["supergalactic_south"] = summary(delta[b < 0])
    return result


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm allocation required")
    if OUT.exists():
        raise FileExistsError(OUT)
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError("CF4 group source changed")
    overlap = {method: [] for method, *_ in METHODS}
    total, tf_positive, eligible = 0, 0, 0
    with SOURCE.open(newline="") as stream:
        for row in csv.DictReader(stream):
            total += 1
            if not row["o_DMtf"] or int(row["o_DMtf"]) <= 0:
                continue
            tf_positive += 1
            tf = float(row["DMtf"] or "nan")
            tf_err = float(row["e_DMtf"] or "nan")
            cz = float(row["Vcmb"] or "nan")
            b = float(row["SGB"] or "nan")
            if (not np.isfinite([tf, tf_err, cz, b]).all() or tf_err <= 0
                    or not 1500 <= cz <= 18000):
                continue
            eligible += 1
            for method, used_col, dm_col, error_col in METHODS:
                if not row[used_col] or int(row[used_col]) <= 0:
                    continue
                dm = float(row[dm_col] or "nan")
                error = float(row[error_col] or "nan")
                if not np.isfinite([dm, error]).all() or error <= 0:
                    continue
                difference = tf-dm
                overlap[method].append(dict(delta=difference,
                    normalized=difference/np.hypot(tf_err, error), cz=cz, SGB=b,
                    n_tf=int(row["o_DMtf"])))
    with np.load(CATALOGUE, allow_pickle=False) as saved:
        train = ~saved["holdout"]
        catalogue = {"training_groups": int(train.sum()),
                     "heldout_groups": int((~train).sum()),
                     "TF_only_member_count": summary(saved["member_count"]),
                     "TF_only_reported_modulus_error": summary(saved["modulus_error"]),
                     "training_cz": summary(saved["observed_cz"][train]),
                     "heldout_cz": summary(saved["observed_cz"][~train])}
    result = {"classification": "INTERNAL_CROSS_METHOD_CONSISTENCY_ONLY",
              "job_id": os.environ["SLURM_JOB_ID"], "source_groups": total,
              "TF_positive_groups": tf_positive, "TF_eligible_overlap_window": eligible,
              "methods": {method: record(rows) if rows else {"n": 0}
                          for method, rows in overlap.items()},
              "TF_only_catalogue": catalogue,
              "limitations": ["CF4 methods have joint source calibration; these are not independent truth residuals.",
                              "The formal normalized difference assumes zero cross-method covariance only as a screen.",
                              "No TF group inclusion, shared calibration covariance, count bias, FoG, or sampler calibration follows."],
              "source_sha256": SOURCE_SHA,
              "TF_only_source_sha256": hashlib.sha256(CATALOGUE.read_bytes()).hexdigest(),
              "R2_posterior": False}
    OUT.mkdir(parents=True)
    (OUT / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
