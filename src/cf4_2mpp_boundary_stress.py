"""Deterministic local boundary-stress fixture for the GH convergence gate."""

from __future__ import annotations

import numpy as np


def boundary_stress_case() -> dict[str, object]:
    """Return fixed edge/cell-boundary sources and broad but finite FoG widths.

    This is a numerical stress fixture only.  It is not an observational truth
    catalogue and carries no permission to promote a likelihood or resolution
    frontier.
    """

    positions = np.array(
        [
            [0.0001, 0.0001, 0.0001],
            [5.9999, 5.9999, 5.9999],
            [0.375, 1.125, 2.375],
            [2.625, 3.875, 4.625],
            [3.0001, 2.9999, 3.0002],
            [1.4999, 4.4999, 5.4999],
        ],
        dtype=np.float64,
    )
    velocities = np.array(
        [
            [300.0, -250.0, 200.0],
            [-300.0, 250.0, -200.0],
            [250.0, -200.0, 150.0],
            [-250.0, 200.0, -150.0],
            [400.0, 400.0, -400.0],
            [-400.0, -400.0, 400.0],
        ],
        dtype=np.float64,
    )
    masses = np.array(
        [np.linspace(0.2, 1.2, 6) * (1.0 + 0.1 * population) for population in range(6)],
        dtype=np.float64,
    )
    indices = np.indices((8, 8, 8)).sum(axis=0)
    exposure = np.array(
        [0.25 + 0.75 * ((indices + population) % 5) / 4.0 for population in range(6)],
        dtype=np.float64,
    )
    return {
        "case_id": "RSD_FOG_boundary_v1",
        "positions": positions,
        "velocities": velocities,
        "masses": masses,
        "exposure": exposure,
        "kwargs": {
            "observer": np.array([3.0, 3.0, 3.0], dtype=np.float64),
            "box_size_cMpc_h": 6.0,
            "hubble_km_s_Mpc": 100.0,
            "little_h": 0.746,
            "scale_factor": 1.0,
            "sigma_fog_km_s": np.array([80.0, 120.0, 160.0, 200.0, 240.0, 300.0], dtype=np.float64),
            "sigma_redshift_km_s": np.array([30.0, 40.0, 50.0, 60.0, 70.0, 80.0], dtype=np.float64),
        },
    }


def nonboundary_stress_case() -> dict[str, object]:
    """Return the paired fixture with all sources away from cell boundaries."""

    case = boundary_stress_case()
    case["positions"] = np.array(
        [
            [0.731, 1.367, 2.413],
            [5.217, 4.631, 3.887],
            [1.743, 2.219, 4.157],
            [2.941, 5.083, 0.677],
            [4.271, 0.913, 1.589],
            [3.413, 3.731, 5.347],
        ],
        dtype=np.float64,
    )
    case["case_id"] = "RSD_FOG_nonboundary_v1"
    return case
