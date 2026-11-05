#!/usr/bin/env python3
"""Bounded Slurm timing for the finite LF transfer used by raw FP marks.

This isolates the changed transfer kernel; it is not a full target or PMWD
benchmark. Times include compiled forward-plus-gradient evaluations after one
warm compilation for each fixed source batch.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cf4_r2_marked_tracer_jax import source_mark_transfer


OUT = Path(os.environ.get(
    "CF4_R2_OUT_DIR",
    "/gpfs/kjhan/CF4/z0_density/r2_lf_transfer_cost_20261002_v1",
))
SIZES = (4096, 65536, 262144)
REPEATS = 5
SEED = 20261002


def _sync(tree):
    return jax.tree_util.tree_map(
        lambda value: value.block_until_ready(), tree
    )


def _case(size: int, rng: np.random.Generator) -> dict:
    true_modulus = jnp.asarray(rng.uniform(33.0, 38.0, size))
    observed_modulus = true_modulus + jnp.asarray(
        rng.normal(0.0, 0.12, size)
    )
    true_redshift = jnp.asarray(rng.uniform(0.002, 0.060, size))
    observed_redshift = jnp.maximum(
        0.0,
        true_redshift + jnp.asarray(rng.normal(0.0, 0.001, size)),
    )
    source_weight = jnp.asarray(rng.uniform(0.1, 1.0, size))

    def objective(parameters):
        mstar, alpha = parameters
        transfer = source_mark_transfer(
            true_modulus,
            observed_modulus,
            true_redshift,
            observed_redshift,
            mstar=mstar,
            alpha=alpha,
            finite_reference_interval=(-25.0, -21.0),
        )[0]
        return jnp.sum(transfer * source_weight[None, :]) / size

    value_and_grad = jax.jit(jax.value_and_grad(objective))
    parameters = jnp.asarray((-23.28, -1.12))
    t0 = time.perf_counter()
    first = _sync(value_and_grad(parameters))
    compile_and_first_seconds = time.perf_counter() - t0
    if not np.isfinite(np.asarray(first[0])).all() or not np.isfinite(
        np.asarray(first[1])
    ).all():
        raise FloatingPointError(f"nonfinite benchmark result at n={size}")

    timings = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        result = _sync(value_and_grad(parameters))
        timings.append(time.perf_counter() - t0)
        if not np.isfinite(np.asarray(result[0])).all():
            raise FloatingPointError(f"nonfinite repeated result at n={size}")

    median = float(np.median(timings))
    return {
        "source_count": size,
        "compile_plus_first_seconds": compile_and_first_seconds,
        "warm_forward_gradient_seconds": timings,
        "warm_median_seconds": median,
        "source_throughput_per_second": size / median,
        "mean_value": float(first[0]),
        "gradient_mstar_alpha": np.asarray(first[1]).tolist(),
        "population_profiled": 0,
        "finite_reference_interval": [-25.0, -21.0],
        "alpha_profiled": -1.12,
    }


def main() -> None:
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU backend required, got {jax.default_backend()}")
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    results = [_case(size, rng) for size in SIZES]
    payload = {
        "status": "FINITE_LF_TRANSFER_KERNEL_COST_ONLY",
        "seed": SEED,
        "repeats_after_compile": REPEATS,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "jax_version": jax.__version__,
        "python": platform.python_version(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "cases": results,
        "limits": [
            "synthetic deterministic inputs; no saved field or observed score",
            "one observed population only",
            "does not include PMWD, raw-mark support construction, or full target adjoint",
            "not a posterior, calibration, heldout result, or science pass",
        ],
    }
    destination = OUT / "result.json"
    temporary = OUT / "result.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(destination)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
