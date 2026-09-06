"""Bounded Z4-data physical posterior; no production or next-bundle launch."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cf4_chunked_hmc import checked_record, make_chunks
import cf4_datum_bearing_z0_phasec_pilot as phasec
from cf4_z4_physical_recovery import load_mock, metrics


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    begin = time.perf_counter()
    plan = json.loads((ROOT / os.environ.get("CF4_POSTERIOR_PLAN", "config/cf4_z5_bias_posterior_plan_v1.json")).read_text())
    root = Path(plan["output_root"])
    if len(sys.argv) > 1 and sys.argv[1] == "aggregate":
        root.mkdir(parents=True, exist_ok=True)
        rows = []
        for task in plan["data"]["tasks"]:
            path = root / f"task_{task}" / "result.json"
            rows.append(json.loads(path.read_text()) if path.exists() else {"task": task, "status": "NO_RESULT_FAILED_OR_INCOMPLETE"})
        dump(root / "aggregate.json", {"bundle": plan["bundle"], "tasks": rows,
             "next_bundle_started": False, "disposition": "DRIVER_REVIEW_THEN_USER_APPROVAL_REQUIRED"})
        print(json.dumps({"tasks": [{"task": r["task"], "status": r["status"]} for r in rows]}), flush=True)
        if plan["bundle"] == "Z7-DATA-SOURCE-STRUCTURE-RECOVERY":
            from cf4_z7_information_sources import compare
            compare(plan)
        return
    task = int(os.environ["SLURM_ARRAY_TASK_ID"])
    out = root / f"task_{task}"
    out.mkdir(parents=True, exist_ok=False)
    if jax.default_backend() != "gpu" or len(jax.devices("gpu")) != 1:
        raise RuntimeError("one allocated Slurm GPU is required")
    cfg = plan["sampler"]
    # CPU mock generation preserves the previous discrete Poisson RNG draw.
    with jax.default_device(jax.devices("cpu")[0]):
        if plan["bundle"] == "Z7-DATA-SOURCE-STRUCTURE-RECOVERY":
            from cf4_z7_information_sources import load_mock as load_channels
            model, design, truth_rho, truth_v, truth_meta, counts, holdcounts, radial, candidate = load_channels(task, plan)
        elif plan["bundle"] == "Z6-NATIVE-PM-Z0-JOINT-PRIOR":
            from cf4_z6_native_physics import load_mock as load_native
            model, design, truth_rho, truth_v, truth_meta, counts, holdcounts, radial, candidate = load_native(task, plan)
        else:
            model, design, truth_rho, truth_v, truth_meta, counts, holdcounts, radial = load_mock(task)
            with np.load(plan["data"]["conditional_start_files"][task], allow_pickle=False) as data:
                candidate = data["MAP_vector"].copy()
        initial_nlp = float(jax.jit(model.nlp)(jnp.zeros(model.size), jnp.asarray(counts), jnp.asarray(radial)))
    if "initial_objective_reference" in plan["data"] and abs(initial_nlp - plan["data"]["initial_objective_reference"][task]) > plan["data"]["initial_objective_absolute_tolerance"]:
        raise ValueError("regenerated Z4 datum does not match the fixed reference")
    np.savez_compressed(out / "mock.npz", counts_train=counts, counts_holdout=holdcounts,
                        radial_mock=radial, truth_density=truth_rho - 1, truth_velocity=truth_v)
    counts_j, radial_j = jnp.asarray(counts), jnp.asarray(radial)
    logdensity = lambda x: -model.nlp(x, counts_j, radial_j)
    initialize, warm, sample, final_step = make_chunks(logdensity, model.size, cfg)
    field_size = model.field_size
    probes = np.concatenate([np.array([0, 1, 31, 32, 1024, 4096, 16384, 32767]) + offset
                             for offset in range(0, field_size, model.n**3)])
    names = [f"nuisance_{i}" for i in range(24)] + [f"white_{i}" for i in probes] + ["white_RMS", "logdensity"]
    saved, traces, projections = [], [], []
    for chain in range(cfg["chain_count"]):
        seed = 2026090600 + 100 * task + chain
        rng = np.random.default_rng(seed)
        position = candidate + .5 * rng.standard_normal(model.size)
        state, adaptation = initialize(jnp.asarray(position))
        chain_saved, chain_trace, chain_projection, warm_trace = [], [], [], []
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
                pos, ld, accept, div, energy, raw_step, used_step = checked_record(records, state)
                trace = np.stack((ld, accept, div, energy, raw_step, used_step), axis=-1)
                if phase == "warmup":
                    warm_trace.append(trace)
                else:
                    chain_saved.append(pos[::cfg["stored_field_thinning"]].copy())
                    chain_trace.append(trace)
                    chain_projection.append(np.column_stack((pos[:, field_size:], pos[:, probes],
                        np.sqrt(np.mean(pos[:, :field_size]**2, axis=1)), ld)))
                progress = {"task": task, "chain": chain, "phase": phase, "steps": offset + len(part),
                            "acceptance": float(accept.mean()), "divergences": int(div.sum()),
                            "step": float(used_step[-1]), "elapsed_s": time.perf_counter() - begin}
                print(json.dumps(progress), flush=True)
                dump(out / "progress.json", progress)
        chain_saved = np.concatenate(chain_saved)
        chain_trace = np.concatenate(chain_trace)
        chain_projection = np.concatenate(chain_projection)
        np.savez_compressed(out / f"chain_{chain}.npz", retained_vectors=chain_saved,
                            sampling_trace=chain_trace, warmup_trace=np.concatenate(warm_trace),
                            projections=chain_projection, projection_names=np.array(names))
        saved.append(chain_saved); traces.append(chain_trace); projections.append(chain_projection)
    draws, traces, projections = np.array(saved), np.array(traces), np.array(projections)
    convergence = phasec.chain_diagnostics(projections, names)
    support = np.asarray(model.response).sum(axis=0) > 0
    field_fn = jax.jit(model.fields)
    density_samples = np.empty((draws.shape[0], draws.shape[1], model.n, model.n, model.n), dtype=np.float32)
    velocity_sum = np.zeros((3, model.n, model.n, model.n)); velocity_square = velocity_sum.copy()
    density_rms = np.empty(draws.shape[:2])
    for c in range(len(draws)):
        for s in range(draws.shape[1]):
            _, rho, vel = (np.asarray(x) for x in field_fn(jnp.asarray(draws[c, s])))
            if not np.isfinite(rho).all() or rho.min() <= 0 or abs(rho.mean() - 1) > 1e-10 or not np.isfinite(vel).all():
                raise ValueError("invalid derived physical posterior field")
            density_samples[c, s] = rho - 1
            density_rms[c, s] = rho.std()
            velocity_sum += vel; velocity_square += vel**2
    count_draws = draws.shape[0] * draws.shape[1]
    density_flat = density_samples.reshape((-1, model.n, model.n, model.n))
    mean = density_flat.mean(axis=0, dtype=np.float64)
    sd = density_flat.std(axis=0, dtype=np.float64)
    quantiles = np.quantile(density_flat, [.025, .16, .84, .975], axis=0)
    mean_v = velocity_sum / count_draws
    sd_v = np.sqrt(np.maximum(velocity_square / count_draws - mean_v**2, 0))
    field_convergence = phasec.chain_diagnostics(density_rms[..., None], ["physical_density_RMS"])
    density_summary = metrics(truth_rho - 1, mean, support)
    for name, mask in (("whole_box", np.ones(support.shape, bool)), ("observed_support", support)):
        truth = (truth_rho - 1)[mask]
        density_summary[name].update(coverage68=float(np.mean((truth >= quantiles[1][mask]) & (truth <= quantiles[2][mask]))),
                                     coverage95=float(np.mean((truth >= quantiles[0][mask]) & (truth <= quantiles[3][mask]))),
                                     mean_posterior_SD=float(sd[mask].mean()))
    forward = jax.jit(model.forward)
    support_pop = np.asarray(model.response) > 0
    held = design["holdout"]
    def scores(x):
        lam, pred = (np.asarray(v) for v in forward(jnp.asarray(x)))
        mu = .2 * lam[support_pop]
        return xlogy(holdcounts[support_pop], mu) - mu, -.5 * (radial[held] - pred[held])**2 / design["variance"][held]
    count_scores, velocity_scores = [], []
    for x in draws[:, ::32].reshape(-1, model.size):
        a, b = scores(x); count_scores.append(a); velocity_scores.append(b)
    zero_count, zero_v = scores(np.zeros(model.size))
    count_gain = float(np.sum(logsumexp(count_scores, axis=0) - np.log(len(count_scores)) - zero_count))
    velocity_gain = float(np.sum(logsumexp(velocity_scores, axis=0) - np.log(len(velocity_scores)) - zero_v))
    nuisance = projections[:, :, :24]
    gate = plan["assessment"]["mechanics_gates"]
    worst_rhat = max(convergence["max_Rhat"], field_convergence["max_Rhat"])
    min_bulk = min(convergence["min_bulk_ESS"], field_convergence["min_bulk_ESS"])
    min_tail = min(convergence["min_tail_ESS"], field_convergence["min_tail_ESS"])
    divergence = float(traces[:, :, 2].mean())
    passed = worst_rhat <= gate["Rhat_max"] and min_bulk >= gate["bulk_ESS_min"] and min_tail >= gate["tail_ESS_min"] and divergence <= gate["sampling_divergence_fraction_max"]
    report = {"task": task, "status": "MECHANICS_PASS_DEVELOPMENT_ONLY" if passed else "NO_GO_SAMPLER_NOT_VALIDATED",
              "commit": os.environ["EXPECTED_COMMIT"], "job_id": os.environ["SLURM_JOB_ID"],
              "GPU": str(jax.devices("gpu")[0]), "truth": truth_meta, "initial_reference_nlp": initial_nlp,
              "convergence": convergence, "density_RMS_convergence": field_convergence,
              "sampling_divergence_fraction": divergence, "sampling_mean_acceptance": float(traces[:, :, 1].mean()),
              "bias_unit_mean": nuisance[:, :, 6:12].mean(axis=(0, 1)).tolist(),
              "bias_unit_quantiles025_50_975": np.quantile(nuisance[:, :, 6:12], [.025, .5, .975], axis=(0, 1)).tolist(),
              "density": density_summary, "velocity": [metrics(truth_v[a], mean_v[a], support) for a in range(3)],
              "heldout_pointwise_log_predictive_gain": {"count": count_gain, "velocity": velocity_gain, "posterior_samples": len(count_scores)},
              "elapsed_s": time.perf_counter() - begin, "peak_host_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              "actual_observational_posterior": False, "calibration_certified": False, "dx_cMpc_h": 12.}
    np.savez_compressed(out / "posterior_fields.npz", density_mean=mean, density_SD=sd, density_quantiles=quantiles,
                        velocity_mean_approximation=mean_v, velocity_SD_approximation=sd_v,
                        truth_density=truth_rho-1, truth_velocity=truth_v, observed_support=support)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    edge = (model.origin_fraction - .5) * model.box / model.n
    for ax, rho, title in zip(axes, (truth_rho, 1 + mean), ("Development truth", "Joint posterior mean (provisional)")):
        im = ax.imshow(np.log10(rho[:, :, 16]).T, origin="lower", vmin=-1, vmax=1, cmap="RdBu_r", extent=(edge, model.box+edge, edge, model.box+edge))
        ax.set_title(title); ax.set_xlabel("cMpc/h")
    fig.colorbar(im, ax=axes, label="log10(rho / mean rho)")
    fig.savefig(out / "density_slice.png", dpi=130); plt.close(fig)
    dump(out / "result.json", report)
    print(json.dumps({"task": task, "status": report["status"], "density": density_summary,
                      "max_Rhat": worst_rhat, "min_bulk_ESS": min_bulk, "min_tail_ESS": min_tail}), flush=True)


if __name__ == "__main__":
    main()
