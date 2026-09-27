"""Single predetermined IC adjoint for live 2M++ count + FP + TF marks."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import cumulative_trapezoid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cf4_r1_particle_forward import make_dynamics, particle_grid
from cf4_r2_live_tf_joint import count_fp_tf_parts

BASE = Path("/gpfs/kjhan/CF4/z0_density")
N, BOX = 128, 384.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--linked", action="store_true",
                        help="use the secure-singleton count-point TF bridge")
    args = parser.parse_args()
    out = BASE / ("r2_live_tf_matched_point_control_v1" if args.linked
                  else "r2_live_tf_joint_control_v1")
    if not os.environ.get("SLURM_JOB_ID") or jax.default_backend() != "gpu":
        raise RuntimeError("Slurm GPU allocation required")
    if out.exists():
        raise FileExistsError(out)
    started = time.monotonic()
    geometry_path = BASE / "r2_hierarchical_field_geometry_v1/geometry_q257.npz"
    tf_path = BASE / ("r2_tf_matched_point_bridge_v1/tf_groups_linked.npz" if args.linked
                      else "r2_tf_source_link_v1/tf_only_groups.npz")
    anchor_path = BASE / "r2_cross_method_anchors_v1/anchors.npz"
    count_path = BASE / "r2_inclusive_count_diagnostic_v1/inclusive_counts_3_sparse.npz"
    selection_path = BASE / "r2_common_selection_128_v1/selection_3.h5"
    state_path = BASE / "r2_hierarchical_field_pilot_v1/initial_chain0.npz"
    with np.load(geometry_path, allow_pickle=False) as f:
        fp = {k:jnp.asarray(f[k]) for k in f.files if k not in ("method_names", "group_labels")}
        if list(f["method_names"]) != ["sbf", "snII", "snIa", "tf"]:
            raise ValueError("FP anchor method order changed")
    with np.load(tf_path, allow_pickle=False) as f:
        tf_host = {k:f[k].copy() for k in f.files}
    with np.load(anchor_path, allow_pickle=False) as f:
        if np.intersect1d(tf_host["group_pgc"], f["CF4_group"]).size:
            raise ValueError("TF-only group overlaps an existing FP non-FP anchor")
    if len(tf_host["group_pgc"]) != 8502 or len(np.unique(tf_host["group_pgc"])) != 8502:
        raise ValueError("TF source group count/identity changed")
    tf = {k:jnp.asarray(v) for k,v in tf_host.items() if k != "group_pgc"}
    with np.load(count_path, allow_pickle=False) as f:
        keys = np.asarray(f["parent_keys"], dtype=np.int32)
        counts = np.asarray(f["parent_counts"], dtype=np.int32)
    if len(keys) != 45776 or int(counts.sum()) != 57238:
        raise ValueError("2M++ inclusive-count ownership changed")
    with h5py.File(selection_path) as f:
        if f.attrs["status"] != "INTEGRATION_COMPLETE_NOT_CALIBRATED":
            raise ValueError("source integrated-selection status changed")
        exposure = np.empty((6, N, N, N), dtype=np.float64)
        for lo in range(0, N, 4):
            exposure[:, lo:lo+4] = f["selection_shells"][:, :, lo:lo+4].sum(axis=1, dtype=np.float64)
    if not np.isfinite(exposure).all() or np.any(exposure.reshape(-1)[keys] <= 0):
        raise ValueError("observed count has no positive selection support")
    common_cfg = json.loads((ROOT / "config/cf4_r2_common_cosmology_v1.json").read_text())
    common, published = common_cfg["common_cosmology"], common_cfg["published_prior"]
    nbar = np.asarray(published["original_mean_count_per_cell_bright_first"])*(
        (BOX/N)/published["original_cell_cMpc_h"])**3
    bias = jnp.asarray(published["linear_regime_bias_bright_first"])
    rate_cfg = json.loads((ROOT / "config/cf4_r2_rate_nuisance_v1.json").read_text())
    shape = jnp.full(6, rate_cfg["rate_shape"])
    frozen = json.loads((ROOT / "config/cf4_datum_bearing_z0_phasec_program_v1.json").read_text())["inference_model"]
    fog = jnp.asarray(frozen["FoG_prior_median_km_s"])
    redshift = jnp.asarray(frozen["fixed_redshift_error_km_s"])
    sd = jnp.asarray([.004, 1., 1., 1., 1., 2.])
    base = json.loads((ROOT / "config/cf4_r1_particle_entry_v1.json").read_text())
    settings = {k:base[k] for k in ("cosmology", "a_start", "a_stop", "a_nbody_maxstep")}
    settings.update(n=N, box_cMpc_h=BOX)
    if settings["cosmology"]["h"] != common["h"] or settings["cosmology"]["Om"] != common["Omega_m"]:
        raise ValueError("PM/observation cosmology mismatch")
    size = N**3
    with np.load(state_path, allow_pickle=False) as f:
        ic, hyper, group = (jnp.asarray(f[k]) for k in
            ("white_ic", "white_hyper", "white_training_group"))
    if ic.shape != (size,) or hyper.shape != (8,) or group.shape != fp["train_group_index"].shape:
        raise ValueError("predeclared saved IC/nuisance shapes changed")
    roots, qweight = np.polynomial.legendre.leggauss(257)
    nodes = (roots+1.)*.5*(191.-.1)+.1
    qweight = qweight*.5*(191.-.1)
    ztable = np.linspace(0., .2, 20001)
    dtable = 2997.92458*cumulative_trapezoid(1/np.sqrt(.31*(1+ztable)**3+.69), ztable, initial=0.)
    zcos = np.interp(nodes, dtable, ztable)
    e, k, c = jnp.asarray(exposure), jnp.asarray(keys), jnp.asarray(counts)
    nodes, qweight, zcos = (jnp.asarray(x) for x in (nodes, qweight, zcos))
    evolve, _initial, conf, _cosmo, particle_mass = make_dynamics(settings)
    particle_masses = jnp.full((size,), particle_mass)

    def factors(ic_arg):
        pos, vel = evolve(ic_arg)
        field = particle_grid(pos, vel, particle_masses, conf)
        return count_fp_tf_parts(field["rho"],
            jnp.moveaxis(field["mean_velocity_km_s"], -1, 0),
            ic_arg, hyper, group, fp, tf, sd, e, k, c,
            jnp.asarray(nbar), shape, bias, fog, redshift,
            nodes, qweight, zcos, box=BOX,
            hubble=common["H0_km_s_Mpc"], h=common["h"])

    def total_with_aux(ic_arg):
        parts, unit = factors(ic_arg)
        return parts.sum(), (parts, jnp.min(jnp.take(unit.reshape(-1), k)))

    (total, (parts, min_unit)), gradient = jax.jit(
        jax.value_and_grad(total_with_aux, has_aux=True))(ic)
    gradient.block_until_ready()
    tf_gradient = np.asarray(jax.jit(jax.grad(lambda x: factors(x)[0][2]))(ic))
    direction = np.random.default_rng(2026092817).standard_normal(size)
    direction /= np.linalg.norm(direction)
    tangent = float(np.dot(tf_gradient, direction))
    epsilon = 2e-5
    tf_score = jax.jit(lambda x: factors(x)[0][2])
    finite = float((tf_score(ic+epsilon*direction)-tf_score(ic-epsilon*direction))/(2*epsilon))
    if not np.isfinite(np.asarray(parts)).all() or not np.isfinite(np.asarray(gradient)).all():
        raise FloatingPointError("nonfinite combined field factors/gradient")
    if not np.isfinite(tf_gradient).all() or float(min_unit) <= 0:
        raise FloatingPointError("TF gradient or occupied count support invalid")
    relative = abs(tangent-finite)/max(1., abs(tangent), abs(finite))
    if relative > .01:
        raise AssertionError(f"TF IC directional derivative discrepancy {relative:g}")
    baseline = None
    if args.linked:
        baseline_path = BASE / "r2_live_tf_joint_control_v1/result.json"
        baseline = json.loads(baseline_path.read_text())
        if baseline["saved_state"] != str(state_path) or baseline["observed_count_points"] != int(counts.sum()):
            raise ValueError("baseline state/count datum changed")
    report = dict(classification=("R2_LIVE_COUNT_FP_TF_MATCHED_POINT_PARTIAL_TARGET_CONTROL"
                                  if args.linked else "R2_LIVE_COUNT_FP_TF_PARTIAL_TARGET_NUMERICAL_CONTROL"),
        job_id=os.environ["SLURM_JOB_ID"], saved_state=str(state_path),
        factors=dict(zip(("count", "FP", "TF", "white_prior"), map(float, np.asarray(parts)))),
        total=float(total), observed_count_cells=len(keys), observed_count_points=int(counts.sum()),
        TF_training_groups=int((~tf_host["holdout"]).sum()),
        TF_holdout_groups=int(tf_host["holdout"].sum()),
        occupied_min_unit_intensity=float(min_unit),
        TF_IC_directional_reverse=tangent, TF_IC_directional_finite_difference=finite,
        TF_IC_directional_relative_discrepancy=relative,
        TF_same_relative_zero_as_FP_anchors=True,
        TF_secure_singleton_point_conditional=bool(args.linked),
        TF_secure_singleton_training_groups=int(np.count_nonzero(
            (tf_host.get("point_population", np.full(len(tf_host["group_pgc"]), -1)) >= 0)
            & ~tf_host["holdout"])),
        TF_factor_change_from_old_same_state=(float(parts[2])-baseline["factors"]["TF"]
                                               if baseline is not None else None),
        TF_redshift_conditioned_not_multiplied=True,
        calibration_claim=False, sampler_run=False, R2_posterior=False,
        TF_group_inclusion_and_covariance_calibrated=False,
        source_paths=[str(p) for p in (geometry_path, tf_path, anchor_path,
                                         count_path, selection_path, state_path)],
        code_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (Path(__file__), ROOT / "src/cf4_r2_live_tf_joint.py",
                               ROOT / "src/cf4_r2_tf_group_marks.py")},
        runtime_seconds=time.monotonic()-started,
        host_peak_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    out.mkdir(parents=True)
    (out / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
