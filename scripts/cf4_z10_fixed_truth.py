"""Z9 observation posterior conditional on its exact native mock fields."""
import json
import os
from pathlib import Path
import resource
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp, xlogy

from cf4_z9_tracer_response import load_mock
from cf4_pm_calibrated_z0 import PMCalibratedFieldModel
from cf4_chunked_hmc import checked_record, make_chunks
from cf4_datum_bearing_z0_phasec_pilot import chain_diagnostics

ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class FixedTruthModel(PMCalibratedFieldModel):
    """Reuse the original observation map/nlp; remove all field coordinates."""

    def __init__(self, source, rho, velocity):
        self.__dict__.update(vars(source))
        self.size = source.size - source.field_size
        self.field_size = 0
        if self.size != 25 or source.tracer_curvature_sigma != .2:
            raise ValueError("expected frozen Z9 extended nuisance model")
        self.truth_rho = jnp.asarray(rho)
        self.truth_velocity = jnp.asarray(velocity)

    def fields(self, vector):
        return jnp.log(self.truth_rho), self.truth_rho, self.truth_velocity


def verify(model, counts, radial, design):
    """Bounded checks of the actual 25D conditional objective before sampling."""
    rng = np.random.default_rng(701)
    objective = jax.jit(model.nlp)
    gradient = jax.jit(jax.grad(model.nlp))
    for curvature_unit in (0., 1.):
        x = rng.normal(scale=.1, size=25)
        x[24] = curvature_unit
        direction = rng.normal(size=25)
        direction /= np.linalg.norm(direction)
        value = float(objective(x, counts, radial))
        grad = np.asarray(gradient(x, counts, radial))
        eps = 1e-5
        finite = float((objective(x + eps * direction, counts, radial) - objective(x - eps * direction, counts, radial)) / (2 * eps))
        np.testing.assert_allclose(np.dot(grad, direction), finite, rtol=2e-5, atol=2e-5)
        lam, predicted = (np.asarray(a) for a in model.forward(jnp.asarray(x)))
        mask = np.asarray(model.response) > 0
        mean = .8 * lam[mask]
        train = ~design["holdout"]
        reference = .5 * np.sum(x**2) + np.sum(mean - np.asarray(counts)[mask] * np.log(mean))
        reference += .5 * np.sum((np.asarray(radial)[train] - predicted[train])**2 / design["variance"][train])
        np.testing.assert_allclose(value, reference, rtol=1e-12, atol=1e-8)
        changed = np.asarray(radial).copy()
        changed[design["holdout"]] += 1e6
        np.testing.assert_allclose(objective(x, counts, changed), value, rtol=1e-12)
        np.testing.assert_allclose(gradient(x, counts, changed), grad, rtol=1e-12, atol=1e-10)
        if not np.isfinite(grad).all():
            raise ValueError("non-finite conditional gradient")


def aggregate(plan):
    root = Path(plan["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in plan["data"]["cases"]:
        path = root / f"task_{case['task']}" / "result.json"
        rows.append(json.loads(path.read_text()) if path.exists() else dict(case, status="NO_RESULT_FAILED_OR_INCOMPLETE"))
    dump(root / "comparison.json", {"bundle": plan["bundle"], "cases": rows,
         "next_bundle_started": False, "disposition": "DRIVER_SCIENCE_REVIEW_REQUIRED",
         "limit": plan["model"]["interpretation"]})
    print(json.dumps({"tasks": [{"task": r["task"], "status": r["status"]} for r in rows]}), flush=True)


def run(plan, task):
    start = time.perf_counter()
    if jax.default_backend() != "gpu" or len(jax.devices("gpu")) != 1:
        raise RuntimeError("one Slurm-allocated GPU required")
    case = plan["data"]["cases"][task]
    if case["task"] != task:
        raise ValueError("task/case mismatch")
    z9plan = json.loads((ROOT / plan["data"]["Z9_plan"]).read_text())
    original = Path(z9plan["output_root"]) / f"task_{case['Z9_task']}"
    old = json.loads((original / "result.json").read_text())
    if old["status"] != "MECHANICS_PASS_DEVELOPMENT_ONLY":
        raise ValueError("Z9 comparison did not pass its sampler gate")
    with jax.default_device(jax.devices("cpu")[0]):
        source, design, rho, vel, meta, counts, heldcounts, radial, _ = load_mock(case["Z9_task"], z9plan)
    with np.load(original / "mock.npz", allow_pickle=False) as saved:
        for key, array in (("counts_train", counts), ("counts_holdout", heldcounts), ("radial_mock", radial), ("truth_density", rho - 1), ("truth_velocity", vel)):
            np.testing.assert_array_equal(saved[key], array)
        counts, heldcounts, radial = (saved[key].copy() for key in ("counts_train", "counts_holdout", "radial_mock"))
        rho, vel = 1 + saved["truth_density"], saved["truth_velocity"].copy()
    if meta["generator_curvature"] != case["true_curvature"]:
        raise ValueError("generator curvature changed")
    model = FixedTruthModel(source, rho, vel)
    counts_j, radial_j = jnp.asarray(counts), jnp.asarray(radial)
    verify(model, counts_j, radial_j, design)
    out = Path(plan["output_root"]) / f"task_{task}"
    out.mkdir(parents=True, exist_ok=False)
    dump(out / "preflight.json", {"status": "PASS", "saved_Z9_data_identical": True,
        "conditional_nlp_gradient_and_holdout_exclusion": True, "free_dimensions": 25})
    print(f"task {task}: fixed-truth preflight PASS; sampling 25 nuisance coordinates", flush=True)
    cfg = plan["sampler"]
    initialize, warm, sample, final_step = make_chunks(lambda q: -model.nlp(q, counts_j, radial_j), 25, cfg)
    all_draws, all_traces = [], []
    for chain in range(cfg["chain_count"]):
        seed = 2026090700 + task * 100 + chain
        state, adaptation = initialize(jnp.asarray(.5 * np.random.default_rng(seed).normal(size=25)))
        draws, traces, warm_traces = [], [], []
        for phase, steps in (("warmup", cfg["warmup_steps"]), ("sampling", cfg["draws_per_chain"])):
            if phase == "sampling":
                step = final_step(adaptation)
            keys = jax.random.split(jax.random.PRNGKey(seed + (10000 if phase == "sampling" else 0)), steps)
            for offset in range(0, steps, cfg["chunk_steps"]):
                part = keys[offset:offset + cfg["chunk_steps"]]
                if phase == "warmup":
                    (state, adaptation), records = warm(state, adaptation, part)
                else:
                    state, records = sample(state, step, part)
                pos, ld, accept, div, energy, raw, used = checked_record(records, state)
                trace = np.stack((ld, accept, div, energy, raw, used), axis=-1)
                if phase == "sampling":
                    draws.append(pos); traces.append(trace)
                else:
                    warm_traces.append(trace)
                progress = dict(task=task, chain=chain, phase=phase, steps=offset + len(part),
                    acceptance=float(accept.mean()), divergences=int(div.sum()), elapsed_s=time.perf_counter() - start)
                dump(out / "progress.json", progress)
                print(json.dumps(progress), flush=True)
        draws, traces = np.concatenate(draws), np.concatenate(traces)
        np.savez_compressed(out / f"chain_{chain}.npz", nuisance_draws=draws,
                            sampling_trace=traces, warmup_trace=np.concatenate(warm_traces))
        all_draws.append(draws); all_traces.append(traces)
    draws, traces = np.asarray(all_draws), np.asarray(all_traces)
    diagnostic_values = np.concatenate((draws, traces[:, :, :1]), axis=-1)
    diag = chain_diagnostics(diagnostic_values, [f"nuisance_{i}" for i in range(25)] + ["log_posterior"])
    gate = plan["assessment"]["mechanics_gates"]
    divergence = float(traces[:, :, 2].mean())
    passed = diag["max_Rhat"] <= gate["Rhat_max"] and diag["min_bulk_ESS"] >= gate["bulk_ESS_min"] and diag["min_tail_ESS"] >= gate["tail_ESS_min"] and divergence <= gate["sampling_divergence_fraction_max"]
    gamma = draws[:, :, 24] * model.tracer_curvature_sigma
    interval = np.quantile(gamma, [.025, .5, .975])
    forward = jax.jit(model.forward)
    support = np.asarray(model.response) > 0
    held = design["holdout"]
    count_scores, radial_scores = [], []
    for q in draws[:, ::32].reshape(-1, 25):
        lam, pred = (np.asarray(a) for a in forward(jnp.asarray(q)))
        mu = .2 * lam[support]
        count_scores.append(xlogy(heldcounts[support], mu) - mu)
        radial_scores.append(-.5 * (radial[held] - pred[held])**2 / design["variance"][held])
    count_lppd = logsumexp(count_scores, axis=0) - np.log(len(count_scores))
    radial_lppd = logsumexp(radial_scores, axis=0) - np.log(len(radial_scores))
    with np.load(original / "heldout_scores.npz", allow_pickle=False) as old_scores:
        np.testing.assert_array_equal(support, old_scores["count_support"])
        count_gain = float(np.sum(count_lppd - old_scores["count_lppd"][support]))
        radial_gain = float(np.sum(radial_lppd - old_scores["velocity_lppd"]))
    flat = draws.reshape(-1, 25)
    report = dict(task=task, Z9_task=case["Z9_task"], true_curvature=case["true_curvature"],
        status="MECHANICS_PASS_CONDITIONAL_DIAGNOSTIC" if passed else "NO_GO_SAMPLER_NOT_VALIDATED",
        commit=os.environ["EXPECTED_COMMIT"], job_id=os.environ["SLURM_JOB_ID"], convergence=diag,
        sampling_divergence_fraction=divergence, sampling_acceptance=float(traces[:, :, 1].mean()),
        curvature_quantiles025_50_975=interval.tolist(), true_curvature_in_95_interval=bool(interval[0] <= case["true_curvature"] <= interval[2]),
        Z9_free_field_curvature_quantiles025_50_975=old["tracer_curvature_quantiles025_50_975"],
        nuisance_quantiles025_50_975=np.quantile(flat, [.025, .5, .975], axis=0).tolist(),
        curvature_bias_unit_correlations=[float(np.corrcoef(flat[:, 24], flat[:, i])[0, 1]) for i in range(6, 12)],
        heldout_lppd_conditional_minus_Z9=dict(count=count_gain, velocity=radial_gain, samples=len(count_scores)),
        elapsed_s=time.perf_counter() - start, peak_host_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        actual_observational_posterior=False, new_density_reconstruction=False, calibration_certified=False,
        limits=plan["model"]["interpretation"])
    dump(out / "result.json", report)
    print(json.dumps({"task": task, "status": report["status"], "curvature_interval": interval.tolist()}), flush=True)


if __name__ == "__main__":
    plan = json.loads((ROOT / "config/cf4_z10_fixed_truth_plan_v1.json").read_text())
    if sys.argv[1:] == ["aggregate"]:
        aggregate(plan)
    elif not sys.argv[1:]:
        run(plan, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        raise SystemExit("use no argument or aggregate")
