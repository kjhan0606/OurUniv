#!/usr/bin/env python3
"""Allocated-GPU API smoke for the replacement identity-HMC implementation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cf4_phasec_sampler_mechanics_pilot as mechanics


def main() -> int:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
        raise RuntimeError("identity-HMC smoke requires one allocated GPU")

    controller = {
        "mechanics": {
            "MAP": {
                "maximum_iterations": 16,
                "maximum_line_search_steps": 10,
                "objective_relative_tolerance": 1e-10,
            },
            "sampler": {
                "chain_count": 2,
                "warmup_steps": 8,
                "posterior_draws_per_chain": 8,
                "integration_steps": 3,
                "initial_step_size": 0.02,
                "target_acceptance_rate": 0.8,
                "chain_initial_jitter_std": 0.05,
                "divergence_energy_threshold": 1000.0,
            },
        },
        "rng_tags": {"chain_initialization": 843927706},
    }

    def standard_normal_nlp(vector):
        return 0.5 * jnp.sum(vector**2)

    draws, diagnostics = mechanics.run_identity_hmc(
        standard_normal_nlp,
        np.zeros(8, dtype=np.float64),
        controller,
        2026090101,
    )
    if draws.shape != (2, 8, 8) or not np.all(np.isfinite(draws)):
        raise RuntimeError("identity-HMC smoke returned invalid draws")
    for key in ("step_size", "acceptance_rate", "energy", "logdensity"):
        if not np.all(np.isfinite(np.asarray(diagnostics[key]))):
            raise RuntimeError(f"identity-HMC smoke returned invalid {key}")
    print(
        json.dumps(
            {
                "status": "PASS",
                "backend": jax.default_backend(),
                "devices": [str(device) for device in jax.devices()],
                "draw_shape": list(draws.shape),
                "step_size": np.asarray(diagnostics["step_size"]).tolist(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
