import numpy as np

from src.cf4_parent_p1 import (
    DensityScorer,
    bootes_metrics,
    observer_environment_metrics,
)


def test_bootes_metrics_returns_gate_row():
    scorer = DensityScorer(-0.2 * np.ones((16, 16, 16), np.float32), 2.0, 2.0)
    spec = {
        "sgl_deg": 0.0,
        "sgb_deg": 0.0,
        "distance_mpc_h": 4.0,
        "profile_radii_mpc_h": [2.0, 4.0],
        "require_negative_mean_at_radii_mpc_h": [2.0, 4.0],
        "maximum_center_shell_percentile": 100.0,
    }
    assert bootes_metrics(spec, scorer)["pass"]


def test_observer_environment_converts_density_to_excess_mass():
    scorer = DensityScorer(0.1 * np.ones((16, 16, 16), np.float32), 2.0, 2.0)
    spec = {
        "local_sheet_radius_mpc_h": 5.0,
        "minimum_local_sheet_mean_delta": -0.5,
        "maximum_excess_mass_msun_h": {"5.0": 1.0e13, "8.0": 5.0e13},
    }
    row = observer_environment_metrics(spec, scorer, omega_m=0.31)
    assert row["pass"]
    assert row["spheres"]["5.0"]["excess_mass_msun_h"] > 0.0


def test_density_scorer_observer_offset_changes_relative_origin():
    field = np.zeros((16, 16, 16), np.float32)
    field[10, 8, 8] = 3.0
    centred = DensityScorer(field, 2.0, 2.0)
    shifted = DensityScorer(field, 2.0, 2.0, observer_offset=np.array([4.0, 0.0, 0.0]))
    assert centred.value(np.zeros(3)) == 0.0
    assert shifted.value(np.zeros(3)) == 3.0
