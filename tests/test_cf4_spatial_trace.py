import numpy as np

from scripts.cf4_trace_ramses_spatial_region import periodic_radius_mask


def test_periodic_radius_selection_wraps_across_box_face() -> None:
    position = np.array([[0.1, 5.0, 5.0], [9.7, 5.0, 5.0], [8.0, 5.0, 5.0]])
    selected = periodic_radius_mask(position, np.array([9.9, 5.0, 5.0]), 10.0, 0.5)
    assert selected.tolist() == [True, True, False]


def test_spatial_trace_runner_is_trace_only_and_uses_environment_sphere() -> None:
    text = open("scripts/run_cf4_lg_b3292_s40349_spatial_trace_v1.sbatch").read()
    assert "--radius 5.0" in text
    assert "--base-level 9" in text
    assert "--buffer 1.5" in text
    assert 'parent_promoted"] is False' in text
    assert 'm33_resolved"] is False' in text
