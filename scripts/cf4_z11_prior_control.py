"""Frozen prior-compatible fields versus the existing native-PM Z9 controls."""
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

from cf4_z9_tracer_response import load_mock as load_z9

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "config/cf4_z11_prior_control_plan_v1.json"


def field_statistics(rho, velocity):
    g = np.log(rho)
    centered = g - g.mean()
    sd = centered.std()
    return dict(density_spatial_SD=float(rho.std()), log_density_spatial_SD=float(sd),
                log_density_skewness=float(np.mean(centered**3) / sd**3),
                log_density_excess_kurtosis=float(np.mean(centered**4) / sd**4 - 3),
                rho_quantiles50_90_99_999=np.quantile(rho, [.5, .9, .99, .999]).tolist(),
                velocity_RMS_km_s=float(np.sqrt(np.mean(velocity**2))))


def prepare(plan):
    z9 = json.loads((ROOT / plan["data"]["Z9_plan"]).read_text())
    return load_z9(plan["data"]["Z9_extended_case"], z9)


def generate(task, plan, prepared):
    case = plan["data"]["cases"][task]
    if case["task"] != task:
        raise ValueError("case/task mismatch")
    model, design = prepared[:2]
    if model.size - model.field_size != 25 or model.tracer_curvature_sigma != .2:
        raise ValueError("Z9 extended model changed")
    seeds = [case[k] for k in ("field_seed", "train_seed", "holdout_seed", "radial_seed")]
    if len(set(seeds)) != 4:
        raise ValueError("field and noise RNG streams must differ")
    truth = np.zeros(model.size)
    truth[:model.field_size] = np.random.default_rng(case["field_seed"]).standard_normal(model.field_size)
    truth[-1] = case["generator_curvature"] / model.tracer_curvature_sigma
    _, rho, velocity = (np.asarray(a) for a in jax.jit(model.fields)(jnp.asarray(truth)))
    if not np.isfinite(rho).all() or np.any(rho <= 0) or abs(rho.mean() - 1) > 1e-10 or not np.isfinite(velocity).all():
        raise ValueError("invalid prior-compatible physical fields")
    lam, signal = (np.asarray(a) for a in jax.jit(model.observe)(jnp.asarray(rho), jnp.asarray(velocity), jnp.asarray(truth[model.field_size:])))
    counts = np.random.default_rng(case["train_seed"]).poisson(.8 * lam)
    heldcounts = np.random.default_rng(case["holdout_seed"]).poisson(.2 * lam)
    radial = signal + np.random.default_rng(case["radial_seed"]).normal(size=signal.shape) * np.sqrt(design["variance"])
    metadata = dict(case, generator="exact fitted z=0 field prior; not PM or IC", native_origin_fraction=.25,
                    covariance_training_indices=prepared[4]["covariance_training_indices"],
                    statistics=field_statistics(rho, velocity), independent_validation=False,
                    calibration_certified=False, observation_generator_shared_with_inference=True)
    return model, design, rho, velocity, metadata, counts, heldcounts, radial, np.zeros(model.size)


def arrays(result):
    return dict(truth_density=result[2] - 1, truth_velocity=result[3], counts_train=result[5],
                counts_holdout=result[6], radial_mock=result[7])


def load_mock(task, plan):
    root = Path(plan["output_root"])
    report = json.loads((root / "preflight.json").read_text())
    if report["status"] != "PASS":
        raise ValueError("Z11 preflight did not pass")
    result = list(generate(task, plan, prepare(plan)))
    with np.load(root / f"input_task_{task}.npz", allow_pickle=False) as saved:
        for key, value in arrays(result).items():
            np.testing.assert_array_equal(saved[key], value)
        result[5:8] = [saved[k].copy() for k in ("counts_train", "counts_holdout", "radial_mock")]
    return tuple(result)


def preflight(plan):
    root = Path(plan["output_root"])
    root.mkdir(parents=True, exist_ok=False)
    prepared = prepare(plan)
    results = [generate(task, plan, prepared) for task in plan["data"]["tasks"]]
    repeated = generate(0, plan, prepared)
    for key, value in arrays(results[0]).items():
        np.testing.assert_array_equal(arrays(repeated)[key], value)
    for a, b in ((0, 1), (2, 3)):
        for index in (2, 3, 7):
            np.testing.assert_array_equal(results[a][index], results[b][index])
    if np.array_equal(results[0][2], results[2][2]):
        raise ValueError("two frozen field seeds produced identical fields")
    for task, result in enumerate(results):
        if np.any(result[-1] != 0):
            raise ValueError("truth used for inference initialization")
        model, design = result[:2]
        zero = jnp.zeros(model.size)
        value, gradient = jax.jit(jax.value_and_grad(model.nlp))(zero, jnp.asarray(result[5]), jnp.asarray(result[7]))
        if not np.isfinite(float(value)) or not np.isfinite(np.asarray(gradient)).all():
            raise ValueError("invalid initial likelihood or gradient")
        changed = result[7].copy(); changed[design["holdout"]] += 1e6
        np.testing.assert_allclose(jax.jit(model.nlp)(zero, jnp.asarray(result[5]), jnp.asarray(changed)), value, rtol=1e-12)
        np.savez_compressed(root / f"input_task_{task}.npz", **arrays(result))
    report = dict(status="PASS", cases=[r[4] for r in results],
        frozen_PM_reference_statistics=field_statistics(prepared[2], prepared[3]),
        deterministic_generation=True, same_field_and_radial_data_within_gamma_pairs=True,
        distinct_field_seeds=True, no_truth_initializer=True, initial_value_gradient_and_holdout_checks=True,
        limits="Descriptive statistics of two prior draws and one previously used PM field; not calibration or a significance test.")
    (root / "preflight.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("Z11 prior-control input and gradient preflight PASS", flush=True)


def compare(plan):
    root = Path(plan["output_root"])
    z9 = json.loads((ROOT / plan["data"]["Z9_plan"]).read_text())
    rows = []
    for case in plan["data"]["cases"]:
        path = root / f"task_{case['task']}" / "result.json"
        reference = json.loads((Path(z9["output_root"]) / f"task_{case['PM_reference_task']}" / "result.json").read_text())
        row = dict(case, status="NO_RESULT_FAILED_OR_INCOMPLETE")
        if path.exists():
            result = json.loads(path.read_text())
            interval = result["tracer_curvature_quantiles025_50_975"]
            row.update(status=result["status"], curvature_quantiles025_50_975=interval,
                true_curvature_in_95_interval=interval[0] <= case["generator_curvature"] <= interval[2],
                median_minus_truth=interval[1] - case["generator_curvature"], density=result["density"],
                heldout_gain_against_own_homogeneous_baseline=result["heldout_pointwise_log_predictive_gain"],
                truth_statistics=result["truth"]["statistics"])
        row["PM_reference_curvature_quantiles025_50_975"] = reference["tracer_curvature_quantiles025_50_975"]
        row["PM_reference_density"] = reference["density"]
        rows.append(row)
    report = dict(bundle=plan["bundle"], cases=rows, disposition="DRIVER_SCIENCE_REVIEW_REQUIRED",
                  next_bundle_started=False, actual_observational_posterior=False, calibration_certified=False,
                  interpretation=plan["assessment"]["interpretation"], limitations=plan["model"]["limitations"])
    (root / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"tasks": [{"task": r["task"], "status": r["status"]} for r in rows]}), flush=True)


if __name__ == "__main__":
    plan = json.loads(PLAN.read_text())
    if sys.argv[1:] == ["preflight"]:
        preflight(plan)
    elif sys.argv[1:] == ["compare"]:
        compare(plan)
    else:
        raise SystemExit("use preflight or compare")
