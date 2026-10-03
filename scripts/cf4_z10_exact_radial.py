"""Exact independent radial-calibration block for the fixed-truth Z10 posterior."""
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from cf4_z10_fixed_truth import ROOT, FixedTruthModel, chain_diagnostics, dump, load_mock


def gaussian_block(A, residual, variance, train):
    a = A[train] / np.sqrt(variance[train, None])
    y = residual[train] / np.sqrt(variance[train])
    precision = np.eye(4) + a.T @ a
    chol = np.linalg.cholesky(precision)
    rhs = a.T @ y
    covariance = np.linalg.solve(chol.T, np.linalg.solve(chol, np.eye(4)))
    mean = np.linalg.solve(chol.T, np.linalg.solve(chol, rhs))
    np.testing.assert_allclose(precision @ covariance, np.eye(4), atol=1e-11)
    np.testing.assert_allclose(precision @ mean, rhs, rtol=1e-11, atol=1e-11)
    return mean, covariance, precision


def energy(q, mean, precision):
    centered = q - mean
    return .5 * np.einsum("...i,ij,...j->...", centered, precision, centered)


def gates_pass(d, gate):
    return d["max_Rhat"] <= gate["Rhat_max"] and d["min_bulk_ESS"] >= gate["bulk_ESS_min"] and d["min_tail_ESS"] >= gate["tail_ESS_min"]


def process(plan, z10, case, root):
    task = case["task"]
    start = time.perf_counter()
    z9 = json.loads((ROOT / z10["data"]["Z9_plan"]).read_text())
    original = Path(z9["output_root"]) / f"task_{case['Z9_task']}"
    old_root = Path(z10["output_root"]) / f"task_{task}"
    old_result = json.loads((old_root / "result.json").read_text())
    source, design, rho, velocity, _, counts, heldcounts, radial, _ = load_mock(case["Z9_task"], z9)
    with np.load(original / "mock.npz", allow_pickle=False) as saved:
        for key, array in (("counts_train", counts), ("counts_holdout", heldcounts), ("radial_mock", radial), ("truth_density", rho - 1), ("truth_velocity", velocity)):
            np.testing.assert_array_equal(saved[key], array)
    model = FixedTruthModel(source, rho, velocity)
    A = np.asarray(model.B) * np.asarray(model.qstd)[None, :]
    base = np.asarray(model.radial_prediction(jnp.asarray(velocity), jnp.zeros(25)))
    variance, train = design["variance"], ~design["holdout"]
    mean, covariance, precision = gaussian_block(A, radial - base, variance, train)
    altered = radial.copy(); altered[~train] += 1e6
    for actual, expected in zip(gaussian_block(A, altered - base, variance, train), (mean, covariance, precision), strict=True):
        np.testing.assert_array_equal(actual, expected)

    # Check the analytic block against the unchanged original full likelihood.
    forward = jax.jit(model.forward)
    objective = jax.jit(model.nlp)
    rng = np.random.default_rng(plan["seed"] + task * 100)
    for _ in range(2):
        u = rng.normal(scale=.2, size=25)
        v = u.copy(); v[20:24] = rng.normal(size=4)
        intensity_u, signal_u = (np.asarray(a) for a in forward(jnp.asarray(u)))
        intensity_v, signal_v = (np.asarray(a) for a in forward(jnp.asarray(v)))
        np.testing.assert_array_equal(intensity_u, intensity_v)
        np.testing.assert_allclose(signal_u, base + A @ u[20:24], rtol=1e-12, atol=1e-10)
        np.testing.assert_allclose(signal_v - signal_u, A @ (v[20:24] - u[20:24]), rtol=1e-12, atol=1e-10)
        actual = float(objective(v, counts, radial) - objective(u, counts, radial))
        expected = energy(v[20:24], mean, precision) - energy(u[20:24], mean, precision)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-8)

    draws, logp, divergences = [], [], []
    for chain in range(z10["sampler"]["chain_count"]):
        with np.load(old_root / f"chain_{chain}.npz", allow_pickle=False) as saved:
            draws.append(saved["nuisance_draws"])
            logp.append(saved["sampling_trace"][:, 0])
            divergences.append(saved["sampling_trace"][:, 2])
    draws, logp = np.asarray(draws), np.asarray(logp)
    if draws.shape != (4, 1024, 25) or not np.isfinite(draws).all() or not np.isfinite(logp).all():
        raise ValueError("unexpected or non-finite saved chains")
    retained = np.r_[np.arange(20), 24]
    names = [f"nuisance_{i}" for i in range(25)]
    marginal_logp = logp + energy(draws[:, :, 20:24], mean, precision)
    marginal = chain_diagnostics(np.concatenate((draws[:, :, retained], marginal_logp[..., None]), axis=-1),
                                 [names[i] for i in retained] + ["marginal_log_posterior_up_to_constant"])
    repaired = draws.copy()
    factor = np.linalg.cholesky(covariance)
    iid_rng = np.random.default_rng(plan["seed"] + 10000 + task)
    repaired[:, :, 20:24] = mean + iid_rng.normal(size=draws[:, :, 20:24].shape) @ factor.T
    repaired_logp = marginal_logp - energy(repaired[:, :, 20:24], mean, precision)
    np.testing.assert_array_equal(repaired[:, :, retained], draws[:, :, retained])
    white = np.linalg.solve(factor, (repaired[:, :, 20:24].reshape(-1, 4) - mean).T).T
    count = len(white)
    if np.max(np.abs(white.mean(axis=0))) > 5 / np.sqrt(count) or np.max(np.abs(np.cov(white, rowvar=False) - np.eye(4))) > 6 * np.sqrt(2 / count):
        raise ValueError("IID radial draws fail fixed moment tolerance")
    for c, s in ((0, 0), (3, 1023)):
        np.testing.assert_allclose(logp[c, s], -float(objective(draws[c, s], counts, radial)), rtol=1e-11, atol=1e-7)
        np.testing.assert_allclose(repaired_logp[c, s], -float(objective(repaired[c, s], counts, radial)), rtol=1e-11, atol=1e-7)
    joint = chain_diagnostics(np.concatenate((repaired, repaired_logp[..., None]), axis=-1), names + ["log_posterior"])
    interval = np.quantile(repaired[:, :, 24] * .2, [.025, .5, .975])
    np.testing.assert_array_equal(interval, old_result["curvature_quantiles025_50_975"])
    held = ~train
    predicted = base[held] + A[held] @ mean
    predicted_variance = variance[held] + np.einsum("ni,ij,nj->n", A[held], covariance, A[held])
    # Same omission of fixed N(y; ., variance) constants as Z9/Z10 scores.
    radial_lppd = -.5 * (radial[held] - predicted)**2 / predicted_variance - .5 * np.log(predicted_variance / variance[held])
    with np.load(original / "heldout_scores.npz", allow_pickle=False) as saved:
        radial_gain = float(np.sum(radial_lppd - saved["velocity_lppd"]))
    gate = plan["assessment"]["mechanics_gates"]
    old_divergence = float(np.mean(divergences))
    passed = gates_pass(marginal, gate) and gates_pass(joint, gate) and old_divergence <= gate["sampling_divergence_fraction_max"]
    out = root / f"task_{task}"
    out.mkdir(exist_ok=False)
    np.savez_compressed(out / "posterior.npz", nuisance_draws=repaired, log_posterior=repaired_logp,
                        exact_radial_mean=mean, exact_radial_covariance=covariance,
                        heldout_radial_lppd=radial_lppd)
    report = dict(task=task, true_curvature=case["true_curvature"],
        status="PASS_EXACT_BLOCK_CONDITIONAL_DIAGNOSTIC" if passed else "NO_GO_SAMPLER_NOT_VALIDATED",
        preflight="PASS_FACTORIZATION_DATA_OBJECTIVE_HOLDOUT_AND_IID_MOMENTS",
        source_Z10_status=old_result["status"], retained_marginal_convergence=marginal, reconstructed_joint_convergence=joint,
        source_HMC_divergence_fraction=old_divergence, new_HMC_run=False, all_21_other_coordinates_unchanged=True,
        curvature_quantiles025_50_975=interval.tolist(), true_curvature_in_95_interval=bool(interval[0] <= case["true_curvature"] <= interval[2]),
        Z9_free_field_curvature_quantiles025_50_975=old_result["Z9_free_field_curvature_quantiles025_50_975"],
        exact_radial_mean=mean.tolist(), exact_radial_covariance=covariance.tolist(),
        heldout_count_lppd_gain_vs_Z9_unchanged=old_result["heldout_lppd_conditional_minus_Z9"]["count"],
        exact_heldout_radial_lppd_gain_vs_Z9=radial_gain,
        actual_observational_posterior=False, calibration_certified=False, elapsed_s=time.perf_counter() - start)
    dump(out / "result.json", report)
    print(json.dumps({"task": task, "status": report["status"], "max_Rhat": joint["max_Rhat"], "min_bulk_ESS": joint["min_bulk_ESS"]}), flush=True)
    return report


if __name__ == "__main__":
    plan = json.loads((ROOT / "config/cf4_z10_exact_radial_plan_v1.json").read_text())
    z10 = json.loads((ROOT / plan["Z10_plan"]).read_text())
    root = Path(plan["output_root"])
    root.mkdir(parents=True, exist_ok=False)
    rows = [process(plan, z10, case, root) for case in z10["data"]["cases"]]
    dump(root / "comparison.json", {"bundle": plan["bundle"], "cases": rows,
         "commit": os.environ["EXPECTED_COMMIT"], "job_id": os.environ["SLURM_JOB_ID"],
         "disposition": "DRIVER_REVIEW_REQUIRED", "next_bundle_started": False,
         "limits": plan["assessment"]["science"]})
