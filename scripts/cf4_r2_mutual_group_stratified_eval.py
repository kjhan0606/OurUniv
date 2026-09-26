"""Frozen-model method/depth predictive check; no refit or new sources."""

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/gpfs/kjhan/CF4/z0_density/r2_mutual_group_conditional_v2")
OUT = Path("/gpfs/kjhan/CF4/z0_density/r2_mutual_group_stratified_v2")
GROUP_HASH = "bfdc0cfc0f172b48468e3a8fd05e87978c1ec68c341fb2d929fc1200f0123334"


def evaluate(z, mask):
    a = z[mask]
    if not a.size:
        return None
    result = {"n": int(a.size), "standardized_residual_median": float(np.median(a)),
              "standardized_residual_p90_abs": float(np.percentile(np.abs(a), 90))}
    for level in (.68, .9, .95):
        observed = float(np.mean(np.abs(a) <= t.ppf((1 + level) / 2, 4.0)))
        result[f"central_{int(level*100)}_coverage"] = observed
        result[f"deviation_from_nominal_{int(level*100)}"] = observed - level
        result[f"binomial_2se_{int(level*100)}"] = float(
            2 * np.sqrt(level * (1-level) / a.size))
    return result


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm allocation required")
    if OUT.exists():
        raise FileExistsError(OUT)
    source = json.loads((SOURCE / "result.json").read_text())
    if source["classification"] != "R2_MUTUAL_GROUP_SOURCE_CONDITIONAL_DIAGNOSTIC_NOT_JOINT_LIKELIHOOD":
        raise ValueError("upstream result changed")
    with np.load(SOURCE / "mutual_pairs.npz", allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    path = ROOT / "data/cf4_groups.csv"
    if hashlib.sha256(path.read_bytes()).hexdigest() != GROUP_HASH:
        raise ValueError("CF4 group source changed")
    with path.open(newline="", encoding="utf-8") as handle:
        groups = {int(row["1PGC"]): row for row in csv.DictReader(handle)}
    redshift = np.asarray([float(groups[int(p)]["Vcmb"]) for p in data["cf4_1pgc"]])
    holdout = data["cf4_holdout"]
    pattern = data["cf4_method_pattern"].astype(str)
    loc = source["velocity_candidate"]["location_km_s"]
    floor = source["velocity_candidate"]["fitted_floor_km_s"]
    scale = np.sqrt(floor**2 +
        (data["twompp_group_sigma_km_s"] / np.sqrt(data["twompp_group_rich"]))**2)
    z = (data["delta_vcmb_km_s"] - loc) / scale
    method = {"FP_only": pattern == "fp", "TF_only": pattern == "tf",
              "mixed_or_other": (pattern != "fp") & (pattern != "tf")}
    depth = {"Vcmb_below_10000": redshift < 10000,
             "Vcmb_at_least_10000": redshift >= 10000}
    report = {
        "classification": "R2_RESTRICTED_REDSHIFT_CONDITIONAL_STRATIFIED_CHECK_NOT_JOINT_LIKELIHOOD",
        "upstream": str(SOURCE / "result.json"),
        "fitted_parameters_unchanged": True,
        "method_strata": {key: evaluate(z, holdout & mask) for key, mask in method.items()},
        "depth_strata": {key: evaluate(z, holdout & mask) for key, mask in depth.items()},
        "train_method_strata": {key: evaluate(z, ~holdout & mask) for key, mask in method.items()},
        "scope": "mutual one-to-one source-bridge pairs only; original native CF4 holdout",
        "limitation": "Stratum checks are descriptive and correlated/multiple; 2SE is not a scientific acceptance threshold. No independent-sky validation, CF4 distance selection, method-specific distance-mark law, or field-conditioned joint likelihood follows."
    }
    OUT.mkdir(parents=True)
    (OUT / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
