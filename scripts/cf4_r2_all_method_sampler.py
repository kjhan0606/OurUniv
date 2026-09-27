"""Bounded count+FP+TF same-current-field HMC equilibration test."""

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
from cf4_chunked_hmc import make_chunks
from cf4_r1_particle_forward import make_dynamics, particle_grid
from cf4_r2_live_tf_joint import count_fp_tf_parts

BASE = Path("/gpfs/kjhan/CF4/z0_density")
OUT = BASE / "r2_all_method_sampler_pilot_v2_matched"
N, BOX = 128, 384.
WARMUP, RETAIN = 192, 128
SEEDS = (2026092911, 2026092919)


def checked(record, state):
    """Finite accepted trajectory; rejected divergent energy may be infinite."""
    a = tuple(np.asarray(item) for item in record)
    for index in (0, 1, 2, 5, 6, 7):
        if not np.isfinite(a[index]).all():
            raise FloatingPointError(f"nonfinite accepted/step record {index}")
    if not np.isfinite(np.asarray(state.logdensity_grad)).all():
        raise FloatingPointError("nonfinite accepted state gradient")
    bad = ~np.isfinite(a[4])
    if np.any(bad & ~a[3]):
        raise FloatingPointError("nonfinite non-divergent proposal energy")
    return a, int(bad.sum())


def main():
    if not os.environ.get("SLURM_JOB_ID") or jax.default_backend() != "gpu":
        raise RuntimeError("Slurm GPU allocation required")
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.monotonic()
    geometry_path = BASE / "r2_hierarchical_field_geometry_v1/geometry_q257.npz"
    tf_path = BASE / "r2_tf_matched_point_bridge_v1/tf_groups_linked.npz"
    anchor_path = BASE / "r2_cross_method_anchors_v1/anchors.npz"
    count_path = BASE / "r2_inclusive_count_diagnostic_v1/inclusive_counts_3_sparse.npz"
    selection_path = BASE / "r2_common_selection_128_v1/selection_3.h5"
    starts = [BASE / f"r2_joint_nuisance_pilot_v2/chain{chain}_terminal_white_state.npz"
              for chain in range(2)]
    with np.load(geometry_path, allow_pickle=False) as f:
        fp = {k: jnp.asarray(f[k]) for k in f.files if k not in ("method_names", "group_labels")}
        if list(f["method_names"]) != ["sbf", "snII", "snIa", "tf"]:
            raise ValueError("FP anchor method order changed")
    with np.load(tf_path, allow_pickle=False) as f:
        tf_host = {k: f[k].copy() for k in f.files}
    with np.load(anchor_path, allow_pickle=False) as f:
        if np.intersect1d(tf_host["group_pgc"], f["CF4_group"]).size:
            raise ValueError("TF-only group overlaps an existing FP anchor")
    if len(tf_host["group_pgc"]) != 8502 or int((~tf_host["holdout"]).sum()) != 6745:
        raise ValueError("TF catalogue identity/split changed")
    matched = tf_host["point_population"] >= 0
    if int(matched.sum()) != 3108 or int(np.count_nonzero(matched & ~tf_host["holdout"])) != 2484:
        raise ValueError("secure singleton TF subset changed")
    tf_host = {key: value[matched] for key, value in tf_host.items()}
    tf = {k: jnp.asarray(v) for k, v in tf_host.items() if k != "group_pgc"}
    with np.load(count_path, allow_pickle=False) as f:
        keys = np.asarray(f["parent_keys"], dtype=np.int32)
        counts = np.asarray(f["parent_counts"], dtype=np.int32)
    if len(keys) != 45776 or int(counts.sum()) != 57238:
        raise ValueError("inclusive count datum changed")
    with h5py.File(selection_path) as f:
        if f.attrs["status"] != "INTEGRATION_COMPLETE_NOT_CALIBRATED":
            raise ValueError("selection source status changed")
        exposure = np.empty((6, N, N, N), dtype=np.float64)
        for lo in range(0, N, 4):
            exposure[:, lo:lo+4] = f["selection_shells"][:, :, lo:lo+4].sum(axis=1, dtype=np.float64)
    if not np.isfinite(exposure).all() or np.any(exposure.reshape(-1)[keys] <= 0):
        raise ValueError("observed count selection support failure")
    cfg = json.loads((ROOT / "config/cf4_r2_common_cosmology_v1.json").read_text())
    common, published = cfg["common_cosmology"], cfg["published_prior"]
    nbar = np.asarray(published["original_mean_count_per_cell_bright_first"]) * (
        (BOX/N)/published["original_cell_cMpc_h"])**3
    bias0 = jnp.asarray(published["linear_regime_bias_bright_first"])
    rate_cfg = json.loads((ROOT / "config/cf4_r2_rate_nuisance_v1.json").read_text())
    shape = jnp.full(6, rate_cfg["rate_shape"])
    frozen = json.loads((ROOT / "config/cf4_datum_bearing_z0_phasec_program_v1.json").read_text())["inference_model"]
    fog0 = jnp.asarray(frozen["FoG_prior_median_km_s"])
    redshift = jnp.asarray(frozen["fixed_redshift_error_km_s"])
    calibration_sd = jnp.asarray([.004, 1., 1., 1., 1., 2.])
    base = json.loads((ROOT / "config/cf4_r1_particle_entry_v1.json").read_text())
    settings = {k: base[k] for k in ("cosmology", "a_start", "a_stop", "a_nbody_maxstep")}
    settings.update(n=N, box_cMpc_h=BOX)
    if settings["cosmology"]["h"] != common["h"] or settings["cosmology"]["Om"] != common["Omega_m"]:
        raise ValueError("PM/observation cosmology mismatch")
    roots, weights = np.polynomial.legendre.leggauss(257)
    nodes = (roots+1.)*.5*(191.-.1)+.1
    weights = weights*.5*(191.-.1)
    ztable = np.linspace(0., .2, 20001)
    dtable = 2997.92458*cumulative_trapezoid(1/np.sqrt(.31*(1+ztable)**3+.69), ztable, initial=0.)
    zcos = np.interp(nodes, dtable, ztable)
    nodes, weights, zcos = (jnp.asarray(v) for v in (nodes, weights, zcos))
    e, k, c = jnp.asarray(exposure), jnp.asarray(keys), jnp.asarray(counts)
    size, nh, ng, nt = N**3, 8, len(fp["train_group_index"]), 7
    dimension = size+nh+ng+nt
    evolve, _initial, conf, _cosmo, particle_mass = make_dynamics(settings)
    mass = jnp.full((size,), particle_mass)

    def split(x):
        return (x[:size], x[size:size+nh], x[size+nh:size+nh+ng], x[size+nh+ng:])

    def factors(x):
        ic, hyper, group, nuisance = split(x)
        pos, vel = evolve(ic)
        field = particle_grid(pos, vel, mass, conf)
        parts, unit = count_fp_tf_parts(field["rho"],
            jnp.moveaxis(field["mean_velocity_km_s"], -1, 0), ic,
            hyper, group, fp, tf, calibration_sd, e, k, c,
            jnp.asarray(nbar), shape, bias0*jnp.exp(.5*nuisance[:6]),
            fog0*jnp.exp(.7*nuisance[6]), redshift,
            nodes, weights, zcos, box=BOX, hubble=common["H0_km_s_Mpc"],
            h=common["h"])
        parts = parts.at[3].add(-.5*jnp.vdot(nuisance, nuisance))
        return parts, jnp.min(jnp.take(unit.reshape(-1), k))

    target = lambda x: factors(x)[0].sum()
    factor_fn = jax.jit(factors)
    hmc = dict(initial_step_size=.003, maximum_step_size=.05,
               target_acceptance=.8, divergence_threshold=1000.,
               integration_steps=6, integration_steps_range=[4, 8])
    initialize, warm, sample, final = make_chunks(target, dimension, hmc, record_steps=True)
    report = dict(classification="R2_ALL_METHOD_PARTIAL_TARGET_SAMPLER_TEST",
        status="STARTED", job_id=os.environ["SLURM_JOB_ID"], N=N, box_cMpc_h=BOX,
        count_points=int(counts.sum()), count_occupied_cells=len(keys),
        FP_training_groups=ng, TF_training_groups=int((~tf_host["holdout"]).sum()),
        TF_holdout_groups=int(tf_host["holdout"].sum()),
        TF_unmatched_or_ambiguous_groups_not_in_target=5394,
        warmup_per_chain=WARMUP, retained_per_chain=RETAIN,
        source_calibrated=False, group_inclusion_calibrated=False,
        posterior_converged=False, R2_delivery=False,
        starts_are_prior_partial_target_terminal_states=True,
        TF_secure_matched_groups_share_count_bias_and_FoG=True,
        sampler=hmc, chains=[],
        source_paths=list(map(str, (geometry_path, tf_path, anchor_path, count_path,
                                    selection_path, *starts))),
        code_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), ROOT / "src/cf4_r2_live_tf_joint.py",
                      ROOT / "src/cf4_r2_tf_group_marks.py", ROOT / "src/cf4_chunked_hmc.py")})
    OUT.mkdir(parents=True)
    traces = {}

    def write():
        report["runtime_seconds"] = time.monotonic()-started
        report["host_peak_GiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
        (OUT / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        if traces:
            np.savez_compressed(OUT / "transition_summaries.npz", **traces)

    def deadline():
        if time.monotonic()-started > 9300:
            raise TimeoutError("bounded 155-minute application budget exhausted")

    write()
    try:
        for chain, source in enumerate(starts):
            deadline()
            with np.load(source, allow_pickle=False) as saved:
                old = [saved[name] for name in ("white_ic", "white_hyper",
                                                 "white_training_group", "white_tracer")]
            if (old[0].shape != (size,) or old[1].shape != (nh,) or
                    old[2].shape != (ng,) or old[3].shape != (7,)):
                raise ValueError(f"saved chain{chain} state shape changed")
            vector = jnp.asarray(np.concatenate(old))
            parts, support = factor_fn(vector)
            if not np.isfinite(np.asarray(parts)).all() or float(support) <= 0:
                raise FloatingPointError(f"chain{chain} initial target/support invalid")
            state, adaptation = initialize(vector)
            row = dict(chain=chain, start=str(source), initial_factors=np.asarray(parts).tolist(),
                       initial_occupied_min_unit_intensity=float(support),
                       warmup_completed=0, retained_completed=0,
                       rejected_nonfinite_proposal_energies=0)
            report["chains"].append(row)
            key = jax.random.PRNGKey(SEEDS[chain])
            key_warm, key_sample = jax.random.split(key)
            keysets = (jax.random.split(key_warm, WARMUP),
                       jax.random.split(key_sample, RETAIN))
            buffers = {name: [] for name in ("summary", "acceptance", "divergent",
                                            "step", "leapfrog_steps", "phase")}
            for phase, length in enumerate((WARMUP, RETAIN)):
                if phase:
                    step = final(adaptation)
                    row["sampling_step"] = float(step)
                for offset in range(0, length, 4):
                    deadline()
                    keys_chunk = keysets[phase][offset:offset+4]
                    if phase:
                        state, raw = sample(state, step, keys_chunk)
                    else:
                        (state, adaptation), raw = warm(state, adaptation, keys_chunk)
                    a, bad = checked(raw, state)
                    row["rejected_nonfinite_proposal_energies"] += bad
                    x = a[0]
                    summary = np.column_stack((a[1], np.mean(x[:, :size]**2, axis=1),
                        x[:, :4], x[:, size:size+nh],
                        np.mean(x[:, size+nh:size+nh+ng], axis=1), x[:, -nt:]))
                    for name, values in zip(buffers, (summary, a[2], a[3], a[6],
                            a[7], np.full(4, phase, dtype=np.int8))):
                        buffers[name].append(values)
                    for name, values in buffers.items():
                        traces[f"chain{chain}_{name}"] = np.concatenate(values)
                    row["retained_completed" if phase else "warmup_completed"] = offset+4
                    write()
                    print(f"chain{chain} phase{phase} {offset+4}/{length} elapsed={time.monotonic()-started:.1f}s",
                          flush=True)
            parts, support = factor_fn(state.position)
            if not np.isfinite(np.asarray(parts)).all() or float(support) <= 0:
                raise FloatingPointError(f"chain{chain} endpoint target/support invalid")
            mask = traces[f"chain{chain}_phase"] == 1
            row.update(endpoint_factors=np.asarray(parts).tolist(),
                endpoint_occupied_min_unit_intensity=float(support),
                retained_acceptance_mean=float(traces[f"chain{chain}_acceptance"][mask].mean()),
                retained_divergences=int(traces[f"chain{chain}_divergent"][mask].sum()))
            final_x = np.asarray(state.position)
            np.savez_compressed(OUT / f"chain{chain}_terminal_white_state.npz",
                white_ic=final_x[:size], white_hyper=final_x[size:size+nh],
                white_training_group=final_x[size+nh:size+nh+ng],
                white_tracer=final_x[-nt:])
            pos, vel = evolve(jnp.asarray(final_x[:size]))
            field = particle_grid(pos, vel, mass, conf)
            rho = np.asarray(field["rho"])
            if not np.isfinite(rho).all() or abs(float(rho.mean())-1.) > 1e-7:
                raise FloatingPointError("endpoint density mass conservation failed")
            block = rho.reshape(16, 8, 16, 8, 16, 8).mean(axis=(1, 3, 5))
            np.savez_compressed(OUT / f"chain{chain}_terminal_density_block24.npz",
                rho_block24=block.astype(np.float32), block_cMpc_h=24., box_cMpc_h=BOX)
            write()
        report["status"] = "COMPLETE_PARTIAL_TARGET_CHAINS_NOT_YET_ASSESSED"
    except Exception as exc:
        report.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write()
        print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
