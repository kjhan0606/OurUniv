"""Actual CF4/2M++ inputs and provisional products; no invented truth."""
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
from cf4_linear_cr import prepare_bgc_catalog
from cf4_pm_calibrated_z0 import PMCalibratedFieldModel

PLAN = ROOT / "config/cf4_actual_data_preview_v1.json"


def validate_counts(arrays, expected):
    train, held, total, exposure = [arrays[k] for k in
        ("counts_train", "counts_holdout", "counts_all", "raw_selection_exposure")]
    if not (train.shape == held.shape == total.shape == exposure.shape) or train.ndim != 4 or train.shape[0] != 6:
        raise ValueError("invalid six-population array shapes")
    for a in (train, held, total):
        if not np.issubdtype(a.dtype, np.integer) or np.any(a < 0):
            raise ValueError("counts must be nonnegative integers")
    np.testing.assert_array_equal(train+held, total)
    if not np.isfinite(exposure).all() or np.any(exposure < 0) or np.any(exposure > 1+1e-12):
        raise ValueError("invalid raw selection exposure")
    if np.any((total > 0) & (exposure <= 0)):
        raise ValueError("positive observed count outside model support")
    for a, key in ((train, "training_counts"), (held, "holdout_counts"), (total, "retained_counts")):
        if int(a.sum()) != expected[key]:
            raise ValueError(f"changed actual catalog total: {key}")


def prepare(plan):
    data = plan["data"]
    base = json.loads((ROOT / data["base_settings"]).read_text())
    record = json.loads(Path(data["covariance_record"]).read_text())
    if record["training_indices"] != [1,2,3,4] or record["status"] != "PASS_TRAINING_CONTRACT_AND_PSD":
        raise ValueError("unexpected field covariance")
    count_record = json.loads(Path(data["count_result"]).read_text())
    if count_record["status"] != "PASS_PHASE_A_ACTUAL_36635_COUNT_DATUM_ORDER6_RAW_EXPOSURE":
        raise ValueError("actual count datum not accepted")
    with np.load(data["counts"], allow_pickle=False) as f:
        arrays = {k: f[k].copy() for k in ("counts_train", "counts_holdout", "counts_all", "raw_selection_exposure")}
    validate_counts(arrays, data)
    catalog = ROOT / data["CF4_catalog"]
    fixed.verify_frozen_provenance(catalog)
    args = fixed.frozen_args(catalog)
    with np.load(catalog, allow_pickle=False) as f:
        prepared = prepare_bgc_catalog(args, f)
    design = fixed.fixed_design_from_prepared(prepared)
    radial = np.array(prepared["vobs"], copy=True)
    if radial.shape != design["variance"].shape or not np.isfinite(radial).all():
        raise ValueError("invalid actual CF4 velocities")
    transfer, growth = fixed.build_density_transfer(args)
    nbar, bias = phasec._published_prior_arrays(base)
    with np.load(data["covariance"], allow_pickle=False) as f:
        covariance = {k: f[k].copy() for k in f.files}
    model = PMCalibratedFieldModel(transfer, growth, args.box_size, arrays["raw_selection_exposure"],
        design, nbar, bias, base["inference_model"], covariance=covariance, origin_fraction=.25)
    if model.n != plan["grid"]["N"] or model.size-model.field_size != 24:
        raise ValueError("unexpected grid or nuisance dimension")
    meta = dict(actual_data=True, CF4_rows=len(radial), CF4_train=int((~design["holdout"]).sum()),
        CF4_holdout=int(design["holdout"].sum()), count_total=int(arrays["counts_all"].sum()),
        count_population_totals=arrays["counts_all"].sum(axis=(1,2,3)).tolist(),
        covariance_training_indices=record["training_indices"], units=dict(position="cMpc/h", velocity="km/s"),
        field_model="baseline PM-calibrated lognormal; no tracer curvature", science_status="MODEL_STRESS_DIAGNOSTIC_NOT_CALIBRATED")
    return model, design, None, None, meta, arrays["counts_train"], arrays["counts_holdout"], radial, np.zeros(model.size)


def bound_arrays(result):
    design = result[1]
    return dict(counts_train=result[5], counts_holdout=result[6], radial_observed=result[7],
        exposure=np.asarray(result[0].response), **{f"CF4_{k}": v for k, v in design.items()})


def load_data(task, plan):
    if task != 0:
        raise ValueError("one actual-data fit only")
    root = Path(plan["output_root"])
    if json.loads((root / "preflight.json").read_text())["status"] != "PASS":
        raise ValueError("actual inputs not checked")
    result = prepare(plan)
    with np.load(root / "observations.npz", allow_pickle=False) as f:
        for key, value in bound_arrays(result).items():
            np.testing.assert_array_equal(f[key], value)
    return result


def preflight(plan):
    root = Path(plan["output_root"])
    root.mkdir(parents=True, exist_ok=False)
    result = prepare(plan)
    model, design = result[:2]
    zero = jnp.zeros(model.size)
    counts, radial = jnp.asarray(result[5]), jnp.asarray(result[7])
    value, grad = jax.jit(jax.value_and_grad(model.nlp))(zero, counts, radial)
    if not np.isfinite(float(value)) or not np.isfinite(np.asarray(grad)).all():
        raise ValueError("invalid actual-data likelihood gradient")
    changed = result[7].copy(); changed[design["holdout"]] += 1e6
    np.testing.assert_allclose(jax.jit(model.nlp)(zero, counts, jnp.asarray(changed)), value, rtol=1e-12)
    lam, prediction = (np.asarray(a) for a in jax.jit(model.forward)(zero))
    if np.any((result[5]+result[6] > 0) & (lam <= 0)) or not np.isfinite(lam).all():
        raise ValueError("actual counts cannot be predicted")
    np.savez_compressed(root / "observations.npz", **bound_arrays(result))
    report = dict(status="PASS", metadata=result[4], truth_absent=result[2] is None and result[3] is None,
        initial_nlp=float(value), heldout_velocity_exclusion=True,
        source_count_split_checks=True, source_binding_frozen=True,
        homogeneous_expected_population_counts=lam.sum(axis=(1,2,3)).tolist(),
        diagnostic_only=True, calibration_certified=False)
    (root / "preflight.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report), flush=True)


def predictive_products(out, model, design, counts, holdcounts, radial, lam_sum, lam_sq, v_sum, v_sq, number):
    lam = lam_sum/number; signal = v_sum/number
    lam_var = np.maximum(lam_sq/number-lam**2, 0)
    signal_var = np.maximum(v_sq/number-signal**2, 0)
    mu = .2*lam
    count_var = mu+.04*lam_var
    support = np.asarray(model.response) > 0
    count_z = (holdcounts-mu)/np.sqrt(np.where(support, count_var, 1))
    velocity_z = (radial-signal)/np.sqrt(design["variance"]+signal_var)
    np.savez_compressed(out / "observation_predictions.npz", counts_train=counts, counts_holdout=holdcounts,
        intensity_mean=lam, intensity_posterior_variance=lam_var, count_holdout_standardized_residual=count_z,
        radial_observed=radial, radial_mean=signal, radial_posterior_variance=signal_var,
        radial_measurement_variance=design["variance"], radial_standardized_residual=velocity_z,
        radial_positions=design["pos"], radial_holdout=design["holdout"], selection_exposure=np.asarray(model.response))
    held = design["holdout"]
    return dict(count_population_observed_train=counts.sum(axis=(1,2,3)).tolist(),
        count_population_predicted_train=(.8*lam).sum(axis=(1,2,3)).tolist(),
        count_population_observed_holdout=holdcounts.sum(axis=(1,2,3)).tolist(),
        count_population_predicted_holdout=mu.sum(axis=(1,2,3)).tolist(),
        holdout_velocity_standardized_residual_mean=float(velocity_z[held].mean()),
        holdout_velocity_standardized_residual_SD=float(velocity_z[held].std()),
        holdout_count_standardized_residual_mean=float(count_z[support].mean()),
        holdout_count_standardized_residual_SD=float(count_z[support].std()),
        residual_semantics="Moment-standardized model-predictive residuals, not independent normal deviates or a calibration certificate.")


def plot_preview(out, model, mean, sd, sample, counts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(10,8),constrained_layout=True)
    for ax, data, title, cmap, limits, origin in zip(axes.flat,
        (np.log10(1+mean), np.log10(1+sample), sd, np.log1p(counts.sum(axis=0))),
        ("Mean log10 density (z=3)", "Sample log10 density (z=3)", "Density posterior SD (z=3)", "Observed log(1+count) (z=6)"),
        ("RdBu_r", "RdBu_r", "viridis", "magma"), ((-1,1),(-1,1),(None,None),(None,None)), (.25,.25,.25,.5)):
        edge = (origin-.5)*model.box/model.n-model.box/2
        im = ax.imshow(data[:,:,16].T, origin="lower", cmap=cmap, vmin=limits[0], vmax=limits[1],
            extent=(edge,edge+model.box,edge,edge+model.box))
        ax.set_title(title); ax.set_xlabel("cMpc/h from observer"); fig.colorbar(im, ax=ax)
    fig.suptitle("Actual CF4 + 2M++: provisional model-stress preview, dx=12 cMpc/h")
    fig.savefig(out / "density_preview.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    if sys.argv[1:] != ["preflight"]:
        raise SystemExit("use preflight")
    preflight(json.loads(PLAN.read_text()))
