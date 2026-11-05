"""One bounded prior comparison within the goal-focused, actual-data bundle."""
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

from cf4_z6_native_physics import load_mock as load_z6, read_native
from cf4_z9_tracer_response import load_mock as load_z9
from cf4_pm_calibrated_z0 import fit_covariance
from cf4_quantile_z0 import QuantileFieldModel, fit_quantile_map, gaussianize

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "config/cf4_bundle_a_prior_to_data_v1.json"


def source(case, plan):
    name = case["source"]
    source_plan = json.loads((ROOT / plan["data"][f"{name}_plan"]).read_text())
    result = list((load_z9 if name == "Z9" else load_z6)(case["source_task"], source_plan))
    path = Path(source_plan["output_root"]) / f"task_{case['source_task']}" / "mock.npz"
    with np.load(path, allow_pickle=False) as saved:
        for key, value in input_arrays(result).items():
            np.testing.assert_array_equal(saved[key], value)
        result[5:8] = [saved[k].copy() for k in ("counts_train", "counts_holdout", "radial_mock")]
    return result


def input_arrays(result):
    return dict(truth_density=result[2]-1, truth_velocity=result[3],
                counts_train=result[5], counts_holdout=result[6], radial_mock=result[7])


def transformed(result, case, root):
    result = list(result)
    if case["fit"] == "quantile":
        with np.load(root / "prior.npz", allow_pickle=False) as data:
            mapping = {k: data[k].copy() for k in ("x", "y", "slopes")}
            covariance = {k: data[k].copy() for k in data.files if k not in mapping}
        result[0] = QuantileFieldModel(result[0], covariance, mapping)
    elif case["fit"] != "baseline":
        raise ValueError("unknown prior")
    result[4] = dict(result[4], bundle_A_case=case, independent_validation=False)
    result[-1] = np.zeros(result[0].size)
    return tuple(result)


def load_mock(task, plan):
    root = Path(plan["output_root"])
    if json.loads((root / "preflight.json").read_text())["status"] != "PASS":
        raise ValueError("Bundle A inputs/tests did not pass")
    case = plan["data"]["cases"][task]
    if task != case["task"]:
        raise ValueError("case index mismatch")
    result = transformed(source(case, plan), case, root)
    with np.load(root / f"input_task_{task}.npz", allow_pickle=False) as data:
        for key, value in input_arrays(result).items():
            np.testing.assert_array_equal(data[key], value)
    return result


def preflight(plan):
    root = Path(plan["output_root"])
    root.mkdir(parents=True, exist_ok=False)
    cfg = plan["model"]
    native = json.loads((ROOT / plan["data"]["Z6_plan"]).read_text())
    train = plan["data"]["training_indices"]
    if train != native["data"]["training_indices"] or set(train) & set(plan["data"]["evaluation_indices"]):
        raise ValueError("training split changed or overlaps evaluation")
    fields = [read_native(native, i) for i in train]
    mapping = fit_quantile_map([r for r, _ in fields], cfg["knots"], cfg["normal_extent"])
    coordinates = [gaussianize(np.log(r)-np.log(r).mean(), mapping) for r, _ in fields]
    covariance, report = fit_covariance(fields, 384., cfg["spectral_bins"], density_coordinates=coordinates)
    np.savez_compressed(root / "prior.npz", **mapping, **covariance)
    report.update(training_indices=train, evaluation_indices=plan["data"]["evaluation_indices"],
                  one_point_map_is_not_dynamics=True, calibration_certified=False)
    (root / "prior_fit.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    results = []
    for case in plan["data"]["cases"]:
        result = transformed(source(case, plan), case, root)
        model, design = result[:2]
        zero = jnp.zeros(model.size)
        value, gradient = jax.jit(jax.value_and_grad(model.nlp))(zero, jnp.asarray(result[5]), jnp.asarray(result[7]))
        if not np.isfinite(float(value)) or not np.isfinite(np.asarray(gradient)).all():
            raise ValueError("invalid initial likelihood/gradient")
        changed = result[7].copy(); changed[design["holdout"]] += 1e6
        np.testing.assert_allclose(jax.jit(model.nlp)(zero, jnp.asarray(result[5]), jnp.asarray(changed)), value, rtol=1e-12)
        np.savez_compressed(root / f"input_task_{case['task']}.npz", **input_arrays(result))
        results.append(result)
    for key, value in input_arrays(results[2]).items():
        np.testing.assert_array_equal(input_arrays(results[3])[key], value)
    (root / "preflight.json").write_text(json.dumps(dict(status="PASS", saved_source_data_identical=True,
        training_only_transform=True, initial_gradient_and_holdout_checks=True,
        no_truth_initializer=True, actual_data_fit_started=False), indent=2)+"\n")
    print("Bundle A prior inputs, saved-data identity and gradient checks PASS", flush=True)


def compare(plan):
    root = Path(plan["output_root"])
    rows = []
    for pair in plan["data"]["pairs"]:
        base_root = root if pair["baseline_source"] == "self" else Path(json.loads(
            (ROOT / plan["data"][f"{pair['baseline_source']}_plan"]).read_text())["output_root"])
        paths = [base_root / f"task_{pair['baseline_task']}", root / f"task_{pair['candidate_task']}"]
        row = dict(pair, status="MISSING_OR_FAILED_RESULT")
        if not all((p / "result.json").exists() for p in paths):
            rows.append(row); continue
        reports = [json.loads((p / "result.json").read_text()) for p in paths]
        with np.load(paths[0] / "mock.npz", allow_pickle=False) as a, np.load(paths[1] / "mock.npz", allow_pickle=False) as b:
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])
        with np.load(paths[0] / "heldout_scores.npz", allow_pickle=False) as a, np.load(paths[1] / "heldout_scores.npz", allow_pickle=False) as b:
            np.testing.assert_array_equal(a["count_support"], b["count_support"])
            row["heldout_count_delta"] = float((b["count_lppd"]-a["count_lppd"]).sum())
            row["heldout_velocity_delta"] = float((b["velocity_lppd"]-a["velocity_lppd"]).sum())
        old, new = [r["density"]["observed_support"] for r in reports]
        row.update(status="PASS_MECHANICS" if all(r["status"] == "MECHANICS_PASS_DEVELOPMENT_ONLY" for r in reports) else "FAIL_MECHANICS",
            RMSE_ratio=new["RMSE"]/old["RMSE"], correlation_gain=new["correlation"]-old["correlation"],
            coverage_error_increase={str(level): abs(new[f"coverage{level}"]-level/100)-abs(old[f"coverage{level}"]-level/100) for level in (68,95)},
            density_baseline=reports[0]["density"], density_candidate=reports[1]["density"],
            velocity_RMSE_ratios=[reports[1]["velocity"][a]["observed_support"]["RMSE"]/reports[0]["velocity"][a]["observed_support"]["RMSE"] for a in range(3)],
            curvature_baseline=reports[0].get("tracer_curvature_quantiles025_50_975"),
            curvature_candidate=reports[1].get("tracer_curvature_quantiles025_50_975"))
        rows.append(row)
    cfg = plan["assessment"]
    verdicts = []
    for truth in plan["data"]["evaluation_indices"]:
        subset = [r for r in rows if r["truth_index"] == truth]
        passed = bool(subset) and all(r["status"] == "PASS_MECHANICS" for r in subset)
        if passed:
            passed = (np.mean([r["RMSE_ratio"] for r in subset]) <= cfg["RMSE_ratio_max"]
                and np.mean([r["correlation_gain"] for r in subset]) >= cfg["correlation_gain_min"]
                and np.mean([r["heldout_count_delta"] for r in subset]) >= 0
                and all(np.mean([r["coverage_error_increase"][str(level)] for r in subset]) <= cfg["coverage_error_increase_max"] for level in (68,95))
                and all(max([r["RMSE_ratio"], *r["velocity_RMSE_ratios"]]) <= cfg["maximum_pair_RMSE_ratio"] for r in subset))
        verdicts.append(dict(truth_index=truth, passes_frozen_adequacy_rule=bool(passed)))
    report = dict(bundle=plan["bundle"], pairs=rows, per_truth=verdicts,
        quantile_prior_adequacy="PASS_DEVELOPMENT_ONLY" if all(r["passes_frozen_adequacy_rule"] for r in verdicts) else "NO_GO_OR_INCOMPLETE",
        next="Driver review, then record actual-data model/input semantics; no further prior repair loop.",
        actual_data_fit_started=False, next_bundle_started=False, calibration_certified=False)
    (root / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: report[k] for k in ("quantile_prior_adequacy", "per_truth", "next")}), flush=True)


if __name__ == "__main__":
    plan = json.loads(PLAN.read_text())
    if sys.argv[1:] == ["preflight"]:
        preflight(plan)
    elif sys.argv[1:] == ["compare"]:
        compare(plan)
    else:
        raise SystemExit("use preflight or compare")
