"""Two bounded development fits. Never opens observed galaxy counts."""
import json
import os
from pathlib import Path
import resource
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from scipy.special import xlogy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cf4_bgc_fixed_design_smoke as fixed
import cf4_datum_bearing_z0_phasec_pilot as phasec
from cf4_z0_physical_field import PhysicalFieldModel, recenter_old_pm_fields


def metrics(truth, fitted, support):
    def one(mask):
        x, y = truth[mask], fitted[mask]
        return {"correlation": float(np.corrcoef(x, y)[0, 1]),
                "RMSE": float(np.sqrt(np.mean((x - y)**2))),
                "truth_SD": float(x.std()), "fitted_SD": float(y.std())}
    return {"whole_box": one(np.ones(truth.shape, dtype=bool)), "observed_support": one(support)}


def main():
    start = time.perf_counter()
    task = int(os.environ["SLURM_ARRAY_TASK_ID"])
    plan = json.loads((ROOT / "config/cf4_z4_physical_field_plan_v1.json").read_text())
    case = plan["experiments"][task]
    out = Path(plan["output_root"]) / f"task_{task}_{os.environ['SLURM_JOB_ID']}"
    out.mkdir(parents=True, exist_ok=False)
    base = json.loads((ROOT / plan["inputs"]["base_program"]).read_text())
    with np.load(base["input_bindings"]["Phase_A_datum"]["path"], allow_pickle=False) as data:
        response = data["raw_selection_exposure"].astype(float)
    design = fixed.prepare_fixed_design(base["input_bindings"]["CF4_catalog"]["path"])
    assert "vobs" not in design
    transfer, growth = fixed.build_density_transfer(fixed.frozen_args(base["input_bindings"]["CF4_catalog"]["path"]))
    nbar, bias = phasec._published_prior_arrays(base)
    model = PhysicalFieldModel(transfer, growth, 384., response, design, nbar, bias, base["inference_model"])
    rng = np.random.default_rng(case["seed"])
    initial = np.zeros(model.size)
    if task == 0:
        truth_vector = initial.copy()
        truth_vector[:model.n**3] = rng.standard_normal(model.n**3)
        _, truth_rho, truth_v = model.fields(jnp.asarray(truth_vector))
        truth_rho, truth_v = np.asarray(truth_rho), np.asarray(truth_v)
        truth_metadata = {"kind": "matched_model", "nuisance": "fixed at prior centres"}
    else:
        with np.load(plan["inputs"]["nonlinear_truth"], allow_pickle=False) as data:
            original_rho = 1 + data["truth_coarse_density"].astype(float)
            original_v = data["truth_coarse_velocity"].astype(float)
        truth_rho, truth_v = recenter_old_pm_fields(original_rho, original_v)
        truth_metadata = {"kind": "independent_PM_fields_shared_coarse_observation_model",
                          "native_mean_density": float(original_rho.mean()),
                          "remap_mass_error": float(abs(truth_rho.sum() - original_rho.sum())),
                          "native_density_SD": float(original_rho.std()),
                          "remapped_density_SD": float(truth_rho.std()),
                          "extra_smoothing": "positive +0.25-cell mass/momentum remap"}
    observe = jax.jit(model.observe)
    intensity, signal = observe(jnp.asarray(truth_rho), jnp.asarray(truth_v), jnp.zeros(24))
    intensity, signal = np.asarray(intensity), np.asarray(signal)
    if not np.isfinite(intensity).all() or np.any(intensity < 0):
        raise ValueError("invalid mock intensity")
    counts_train = rng.poisson(.8 * intensity)
    counts_hold = rng.poisson(.2 * intensity)
    radial_data = signal + rng.normal(size=signal.size) * np.sqrt(design["variance"])
    counts_j, data_j = jnp.asarray(counts_train), jnp.asarray(radial_data)
    value_grad = jax.jit(jax.value_and_grad(model.nlp))
    evaluations = 0

    def objective(x):
        nonlocal evaluations
        value, grad = value_grad(jnp.asarray(x), counts_j, data_j)
        value, grad = float(value), np.asarray(grad)
        if not np.isfinite(value) or not np.isfinite(grad).all():
            raise FloatingPointError("non-finite MAP objective/gradient")
        evaluations += 1
        if evaluations == 1 or evaluations % 25 == 0:
            print(json.dumps({"task": task, "evaluations": evaluations, "nlp": value,
                              "gradient_inf": float(np.max(abs(grad))),
                              "elapsed_s": time.perf_counter() - start}), flush=True)
        return value, grad

    initial_value, _ = objective(initial)
    options = {k: plan["optimizer"][k] for k in ("maxiter", "maxfun", "gtol", "ftol")}
    fit = minimize(objective, initial, method="L-BFGS-B", jac=True, options=options)
    latent, rho, vel = (np.asarray(a) for a in model.fields(jnp.asarray(fit.x)))
    lam, radial_fit = (np.asarray(a) for a in jax.jit(model.forward)(jnp.asarray(fit.x)))
    lam0, radial0 = (np.asarray(a) for a in jax.jit(model.forward)(jnp.asarray(initial)))
    if not np.isfinite(rho).all() or not np.isfinite(vel).all() or rho.min() <= 0 or abs(rho.mean() - 1) > 1e-10:
        raise ValueError("invalid final physical field")
    support = response.sum(axis=0) > 0
    hold = design["holdout"]
    use = response > 0
    def count_score(mu):
        return float(np.sum(xlogy(counts_hold[use], .2 * mu[use]) - .2 * mu[use]))
    def velocity_score(pred):
        return float(-.5 * np.sum((radial_data[hold] - pred[hold])**2 / design["variance"][hold]))
    report = {"bundle": plan["bundle"], "task": task, "case": case, "truth": truth_metadata,
              "commit": os.environ["EXPECTED_COMMIT"], "Slurm_job_id": os.environ["SLURM_JOB_ID"],
              "grid_N": model.n, "dx_cMpc_h": 12., "actual_observational_inference": False,
              "optimizer": {"success": bool(fit.success), "message": str(fit.message),
                            "iterations": int(fit.nit), "evaluations": evaluations,
                            "initial_nlp": initial_value, "final_nlp": float(fit.fun),
                            "gradient_inf": float(np.max(abs(fit.jac)))},
              "physical_density": {"minimum": float(rho.min()), "mean": float(rho.mean()),
                                   **metrics(truth_rho - 1, rho - 1, support)},
              "velocity": [metrics(truth_v[a], vel[a], support) for a in range(3)],
              "heldout": {"count_log_score_gain_vs_homogeneous": count_score(lam) - count_score(lam0),
                          "velocity_log_score_gain_vs_homogeneous": velocity_score(radial_fit) - velocity_score(radial0),
                          "velocity_rows": int(hold.sum()), "count_support_cells": int(use.sum())},
              "nuisance_standardized_MAP": fit.x[model.n**3:].tolist(),
              "limitations": ["two development MAP fits, not posterior means or coverage",
                              "independent PM dynamics but shared coarse mock observation map",
                              "velocity closure remains approximate; no IC consistency shown",
                              "N32=12 cMpc/h only; no achieved LG or surroundings target resolution"],
              "elapsed_s": time.perf_counter() - start,
              "process_peak_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    np.savez_compressed(out / "fields.npz", truth_density=truth_rho - 1, truth_velocity=truth_v,
                        fitted_physical_density=rho - 1, fitted_velocity_approximation=vel,
                        fitted_log_density_latent=latent, MAP_vector=fit.x, observed_support=support)
    # Fixed shared colour scale, with no fitted spectrum or display normalization.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, field, name in zip(axes, (truth_rho, rho), ("Mock truth", "Physical-density MAP")):
        im = ax.imshow(np.log10(field[:, :, model.n // 2]).T, origin="lower", extent=(0, 384, 0, 384), vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_title(name); ax.set_xlabel("cMpc/h"); ax.set_ylabel("cMpc/h")
    fig.colorbar(im, ax=axes, label="log10(rho / mean rho)")
    fig.suptitle(f"{case['name']} — development N32, 12 cMpc/h")
    fig.savefig(out / "density_slice.png", dpi=130)
    plt.close(fig)
    (out / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
