#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
root=$tng/evaluation/tng100_simba_swift_v26_gaussian_copula_control_audit
cd "$repo"
export PYTHONPATH=$repo/src

[[ ${HOSTNAME,,} == lageunha ]] || {
  echo "V26 Gaussian-control audit must run on Lageunha" >&2
  exit 1
}
if [[ -d $root ]] && find "$root" -mindepth 1 -print -quit | grep -q .; then
  echo "V26 Gaussian-control audit refuses non-empty output: $root" >&2
  exit 1
fi
mkdir -p "$root"

run_domain() {
  local domain=$1 key=$2 seed=$3
  local domain_root=$root/$key
  mkdir -p "$domain_root"
  python -u scripts/hong2021_v26_gaussian_control.py \
    --repo "$repo" --domain "$domain" --seed "$seed" --ensemble 16 \
    --out "$domain_root/ensemble16.h5" --device cuda \
    >"$domain_root/sample.log" 2>&1
  python -u src/hong2021_residual_evaluate.py \
    --candidate "gaussian_copula=$domain_root/ensemble16.h5" \
    --out "$domain_root/ensemble_evaluation" --voxel-mpc-h 0.3125 \
    >"$domain_root/evaluate.log" 2>&1
}

run_domain TNG100 tng 35777
run_domain SIMBA simba_dev 36777
run_domain Swift swift_dev 37777

python - "$repo" "$root" <<'PY'
import json
import os
import sys
from pathlib import Path

from hong2021_v15_development_gate import canonical_digest
from hong2021_v18_init import sha256_file
from hong2021_v20_development_gate import marginal_diagnostics
from hong2021_v6_gate import field_gate

repo, root = map(Path, sys.argv[1:])
baseline_path = Path(
    "/gpfs/kjhan/IllustrisTNG/TNG100-1/evaluation/"
    "tng100_simba_swift_v24_e12_base48/development_decision.json"
)
baseline = json.loads(baseline_path.read_text())["candidates"][-1]
domains = {}
for key in ("tng", "simba_dev", "swift_dev"):
    domain_root = root / key
    ensemble = domain_root / "ensemble16.h5"
    metrics_path = domain_root / "ensemble_evaluation" / "metrics.json"
    metrics_payload = json.loads(metrics_path.read_text())
    if tuple(metrics_payload["candidates"]) != ("gaussian_copula",):
        raise ValueError("Gaussian-control evaluator candidate mismatch")
    metrics = metrics_payload["candidates"]["gaussian_copula"]
    marginal = marginal_diagnostics(ensemble)
    fields = field_gate(metrics)
    q3 = (
        abs(marginal["delta_q99_999_dex"]) <= 0.1
        and marginal["generated_max_above_truth_max_dex"] <= 0.3
    )
    q4 = marginal["generated_over_truth_mean_delta_squared"] <= 1.5
    old = baseline["domains"][key]
    domains[key] = {
        "ensemble": str(ensemble),
        "ensemble_sha256": sha256_file(ensemble),
        "metrics": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "field_gate": fields,
        "mechanism_Q3_Q4": marginal,
        "Q3_pass": q3,
        "Q4_pass": q4,
        "all_pass": fields["pass"] and q3 and q4,
        "comparison_to_v24_step30000": {
            "field_pass": [old["field_gate"]["pass"], fields["pass"]],
            "delta_q99_999_dex": [
                old["mechanism_Q3_Q4"]["delta_q99_999_dex"],
                marginal["delta_q99_999_dex"],
            ],
            "generated_max_above_truth_max_dex": [
                old["mechanism_Q3_Q4"]["generated_max_above_truth_max_dex"],
                marginal["generated_max_above_truth_max_dex"],
            ],
            "generated_over_truth_mean_delta_squared": [
                old["mechanism_Q3_Q4"]["generated_over_truth_mean_delta_squared"],
                marginal["generated_over_truth_mean_delta_squared"],
            ],
        },
    }
all_pass = all(row["all_pass"] for row in domains.values())
all_marginal = all(
    row["Q3_pass"] and row["Q4_pass"] for row in domains.values()
)
if all_pass:
    classification = "conditional_gaussian_copula_sufficient_edm_unnecessary"
    next_step = "freeze_the_explicit_conditional_gaussian_copula_as_v26"
elif all_marginal:
    classification = "gaussian_copula_calibrates_marginals_but_misses_spatial_dependence"
    next_step = "design_a_conditional_spatial_flow_on_the_frozen_v21_latent"
else:
    classification = "conditional_gaussian_copula_rejected"
    next_step = "design_a_conditional_spatial_flow_with_exact_likelihood_and_direct_sampling"
report = {
    "schema": "hong2021-v26-development-gaussian-copula-control-audit-v1",
    "purpose": (
        "Test whether the frozen V21 conditional marginal and train-only Gaussian "
        "copula are sufficient without an EDM trajectory."
    ),
    "model": (
        "V14 observable-conditioned location and Fourier-band scale, V21 "
        "voxel-conditional inverse marginal, source-balanced four-band Gaussian copula"
    ),
    "baseline": str(baseline_path),
    "baseline_sha256": sha256_file(baseline_path),
    "domains": domains,
    "all_domain_pass": all_pass,
    "all_domain_marginal_pass": all_marginal,
    "classification": classification,
    "next": next_step,
    "Astrid_accessed": False,
    "historical_EAGLE_accessed": False,
}
report["audit_digest_sha256"] = canonical_digest(report)
path = root / "audit_summary.json"
partial = path.with_suffix(path.suffix + ".partial")
partial.write_text(json.dumps(report, indent=2) + "\n")
os.replace(partial, path)
print(json.dumps(report, indent=2))
PY
