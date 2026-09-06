"""Train a native-observable z=0 prior on a frozen disjoint development split."""
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cf4_bgc_fixed_design_smoke as fixed
import cf4_datum_bearing_z0_phasec_pilot as phasec
from cf4_pm_calibrated_z0 import PMCalibratedFieldModel, fit_covariance
from cf4_z0_physical_field import PhysicalFieldModel


def read_native(plan, index):
    arm = "ABCD"[index // 2]
    path = Path(plan["data"]["posterior_root"]) / f"posterior_v1_{index:02d}_mock_{index:02d}_seed_{2026083000+index}_arm_{arm}" / "posterior_summary.npz"
    with np.load(path, allow_pickle=False) as data:
        rho = 1 + data["truth_coarse_density"].astype(float)
        velocity = data["truth_coarse_velocity"].astype(float)
    if rho.shape != (32,32,32) or velocity.shape != (3,32,32,32) or np.any(rho <= 0) or abs(rho.mean()-1) > 1e-6:
        raise ValueError("invalid native PM averaging contract")
    return rho, velocity


def calibrate():
    plan = json.loads((ROOT / "config/cf4_z6_native_physics_plan_v1.json").read_text())
    if set(plan["data"]["training_indices"]) & set(plan["data"]["evaluation_indices"]):
        raise ValueError("training/evaluation overlap")
    root = Path(plan["output_root"])
    root.mkdir(parents=True, exist_ok=False)
    fields = [read_native(plan, i) for i in plan["data"]["training_indices"]]
    covariance, report = fit_covariance(fields, 384., plan["model"]["spectral_bins"])
    np.savez_compressed(root / "covariance.npz", **covariance)
    report.update(status="PASS_TRAINING_CONTRACT_AND_PSD", training_indices=plan["data"]["training_indices"],
                  evaluation_indices_not_read=plan["data"]["evaluation_indices"],
                  empirical_prior_not_IC=True, posterior_or_calibration_certified=False)
    (root / "calibration.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report), flush=True)


def load_mock(task, plan):
    root = Path(plan["output_root"])
    report = json.loads((root / "calibration.json").read_text())
    if report["status"] != "PASS_TRAINING_CONTRACT_AND_PSD":
        raise ValueError("calibration did not pass")
    case = plan["data"]["cases"][task]
    if case["truth_index"] in report["training_indices"]:
        raise ValueError("evaluation truth leaked into prior fitting")
    base = json.loads((ROOT / "config/cf4_datum_bearing_z0_phasec_program_v1.json").read_text())
    with np.load(base["input_bindings"]["Phase_A_datum"]["path"], allow_pickle=False) as data:
        response = data["raw_selection_exposure"].astype(float)
    design = fixed.prepare_fixed_design(base["input_bindings"]["CF4_catalog"]["path"])
    transfer, growth = fixed.build_density_transfer(fixed.frozen_args(base["input_bindings"]["CF4_catalog"]["path"]))
    nbar, bias = phasec._published_prior_arrays(base)
    args = (transfer, growth, 384., response, design, nbar, bias, base["inference_model"])
    if case["model"] == "PM_calibrated_joint":
        with np.load(root / "covariance.npz", allow_pickle=False) as data:
            covariance = {key: data[key] for key in data.files}
        model = PMCalibratedFieldModel(*args, covariance=covariance, origin_fraction=.25)
    elif case["model"] == "old_loglinear_native_origin":
        model = PhysicalFieldModel(*args, origin_fraction=.25)
    else:
        raise ValueError("unknown Z6 model")
    rho, velocity = read_native(plan, case["truth_index"])
    lam, signal = (np.asarray(x) for x in jax.jit(model.observe)(jnp.asarray(rho), jnp.asarray(velocity), jnp.zeros(24)))
    rng = np.random.default_rng(case["noise_seed"])
    counts = rng.poisson(.8 * lam)
    holdcounts = rng.poisson(.2 * lam)
    radial = signal + rng.normal(size=signal.size) * np.sqrt(design["variance"])
    meta = dict(case, native_origin_fraction=.25, posthoc_truth_interpolation=False,
                density_SD=float(rho.std()), log_density_SD=float(np.log(rho).std()),
                covariance_training_indices=report["training_indices"],
                shared_coarse_observation_generator=True, independent_validation=False)
    return model, design, rho, velocity, meta, counts, holdcounts, radial, np.zeros(model.size)


if __name__ == "__main__":
    calibrate()
