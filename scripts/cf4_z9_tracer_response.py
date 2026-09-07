"""Bounded tracer-link control with independent Poisson heldout noise."""
import copy
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

from cf4_z6_native_physics import load_mock as load_native

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "config/cf4_z9_tracer_response_plan_v1.json"


def load_mock(task, plan):
    case = plan["data"]["cases"][task]
    if case["task"] != task or case["fit"] not in ("baseline", "curvature"):
        raise ValueError("invalid frozen case")
    if case["train_seed"] == case["heldout_seed"]:
        raise ValueError("training and heldout RNG streams must be separate")
    native_plan = json.loads((ROOT / plan["data"]["Z6_plan"]).read_text())
    model, design, rho, velocity, meta, _, _, radial, _ = load_native(plan["data"]["Z6_case"], native_plan)
    generator = copy.copy(model)
    generator.tracer_curvature_sigma = plan["model"]["curvature_prior_sigma"]
    generator.size += 1
    nuisance = np.zeros(25)
    nuisance[24] = case["generator_curvature"] / generator.tracer_curvature_sigma
    lam, _ = jax.jit(generator.observe)(jnp.asarray(rho), jnp.asarray(velocity), jnp.asarray(nuisance))
    lam = np.asarray(lam)
    if not np.isfinite(lam).all() or np.any(lam < 0):
        raise ValueError("invalid generated tracer intensity")
    counts = np.random.default_rng(case["train_seed"]).poisson(.8 * lam)
    holdcounts = np.random.default_rng(case["heldout_seed"]).poisson(.2 * lam)
    if case["fit"] == "curvature":
        model = generator
    meta.update(case, experiment="Z9 controlled tracer link", independent_validation=False,
                shared_coarse_observation_generator=case["fit"] == "curvature" or case["generator_curvature"] == 0,
                independent_heldout_Poisson_noise=True)
    return model, design, rho, velocity, meta, counts, holdcounts, radial, np.zeros(model.size)


def preflight(plan):
    """One-time exact pair/data check; no storage diagnostics or monitoring loop."""
    root = Path(plan["output_root"])
    root.mkdir(parents=True, exist_ok=False)
    for a, b in ((0, 1), (2, 3)):
        left, right = load_mock(a, plan), load_mock(b, plan)
        for index in (2, 3, 5, 6, 7):
            np.testing.assert_array_equal(left[index], right[index])
        baseline, extended = left[0], right[0]
        x = np.random.default_rng(901 + a).normal(scale=.05, size=baseline.size)
        y = np.concatenate((x, [0.]))
        for actual, expected in zip(jax.jit(extended.forward)(jnp.asarray(y)), jax.jit(baseline.forward)(jnp.asarray(x)), strict=True):
            np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-10)
        if extended.size != baseline.size + 1:
            raise ValueError("curvature parameter missing from posterior")
    (root / "preflight.json").write_text(json.dumps({"status": "PASS", "paired_data_identical": True,
        "zero_curvature_nested_predictions_agree": True, "actual_observational_inference": False}, indent=2) + "\n")
    print("Z9 pair/preflight checks PASS", flush=True)


def compare(plan):
    root = Path(plan["output_root"])
    summary = {"bundle": plan["bundle"], "pairs": [], "next_bundle_started": False,
               "actual_observational_posterior": False, "dx_cMpc_h": 12.,
               "disposition": "DRIVER_SCIENCE_REVIEW_THEN_USER_APPROVAL"}
    for a, b in ((0, 1), (2, 3)):
        row = {"baseline_task": a, "extended_task": b,
               "generator_curvature": plan["data"]["cases"][a]["generator_curvature"]}
        paths = [root / f"task_{t}" for t in (a, b)]
        if not all((p / "result.json").exists() for p in paths):
            row["status"] = "INCOMPLETE_OR_FAILED"
            summary["pairs"].append(row)
            continue
        results = [json.loads((p / "result.json").read_text()) for p in paths]
        with np.load(paths[0] / "mock.npz") as left, np.load(paths[1] / "mock.npz") as right:
            for key in left.files:
                np.testing.assert_array_equal(left[key], right[key])
        with np.load(paths[0] / "heldout_scores.npz") as left, np.load(paths[1] / "heldout_scores.npz") as right:
            np.testing.assert_array_equal(left["count_support"], right["count_support"])
            delta = right["count_lppd"] - left["count_lppd"]
            row["heldout_count_lppd_extended_minus_baseline"] = float(delta.sum())
            row["heldout_count_lppd_delta_by_population"] = delta.sum(axis=(1, 2, 3)).tolist()
            row["heldout_velocity_lppd_extended_minus_baseline"] = float((right["velocity_lppd"] - left["velocity_lppd"]).sum())
        row["status"] = "MECHANICS_PASS_REQUIRES_SCIENCE_REVIEW" if all(r["status"] == "MECHANICS_PASS_DEVELOPMENT_ONLY" for r in results) else "NO_GO_SAMPLER_NOT_VALIDATED"
        row["density_baseline"] = results[0]["density"]
        row["density_extended"] = results[1]["density"]
        row["tracer_curvature_quantiles025_50_975"] = results[1]["tracer_curvature_quantiles025_50_975"]
        summary["pairs"].append(row)
    summary["limits"] = "One reused development PM truth; independent noise, not independent galaxy physics. Pointwise predictive sums do not establish a significance or model evidence. Z8 uncertainty/amplitude inference was withdrawn. No automatic production promotion."
    (root / "comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"pair_status": [r["status"] for r in summary["pairs"]]}), flush=True)


if __name__ == "__main__":
    plan = json.loads(PLAN.read_text())
    if sys.argv[1:] == ["preflight"]:
        preflight(plan)
    elif sys.argv[1:] == ["compare"]:
        compare(plan)
    else:
        raise SystemExit("use preflight or compare")
