import numpy as np
import pytest

from hong2021_camels_density_audit import (
    compare_log_fields,
    deposit_particle_counts,
    field_summary,
    isotropic_power_bands,
    log_density,
    smooth_ngp_as_expected_cic,
)


@pytest.mark.parametrize("assignment", ["ngp", "cic"])
def test_particle_assignment_conserves_periodically(assignment):
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.125, 0.25, 0.75],
            [0.999999, 0.999999, 0.999999],
            [1.0, 1.0, 1.0],
        ]
    )
    counts = deposit_particle_counts(
        coordinates, grid=4, box_mpc_h=1.0, assignment=assignment
    )
    assert counts.shape == (4, 4, 4)
    assert counts.sum() == pytest.approx(4.0, abs=1e-13)
    assert np.all(counts >= 0)


def test_ngp_matches_floor_cell_definition():
    coordinates = np.array(
        [
            [0.01, 0.01, 0.01],
            [0.24, 0.24, 0.24],
            [0.26, 0.26, 0.26],
        ]
    )
    counts = deposit_particle_counts(
        coordinates, grid=4, box_mpc_h=1.0, assignment="ngp"
    )
    assert counts[0, 0, 0] == 2
    assert counts[1, 1, 1] == 1
    assert np.count_nonzero(counts) == 2


def test_cic_cell_center_is_exact_and_boundary_wraps():
    coordinates = np.array([[0.125, 0.125, 0.125], [0.0, 0.0, 0.0]])
    counts = deposit_particle_counts(
        coordinates, grid=4, box_mpc_h=1.0, assignment="cic"
    )
    assert counts[0, 0, 0] == pytest.approx(1.125)
    boundary_cells = np.ix_([0, 3], [0, 3], [0, 3])
    assert counts[boundary_cells].sum() == pytest.approx(2.0)
    assert np.count_nonzero(counts) == 8


def test_assignment_rejects_invalid_payload():
    with pytest.raises(ValueError, match="shape"):
        deposit_particle_counts(np.zeros((2, 2)), grid=4, box_mpc_h=1)
    with pytest.raises(ValueError, match="non-finite"):
        deposit_particle_counts(np.array([[0.0, np.nan, 0.0]]))
    with pytest.raises(ValueError, match="outside"):
        deposit_particle_counts(np.array([[0.0, -1.0, 0.0]]), box_mpc_h=1)
    with pytest.raises(ValueError, match="unsupported"):
        deposit_particle_counts(np.zeros((1, 3)), assignment="tsc")


def test_field_summary_reports_void_floor():
    field = np.array([[[0.0, 1.0], [2.0, 5.0]]])
    result = field_summary(field)
    assert result["zero_cells"] == 1
    assert result["zero_volume_fraction"] == 0.25
    assert result["minimum_positive"] == 1.0
    assert result["mean"] == 2.0


def test_log_density_requires_positive_values():
    field = np.ones((2, 2, 2))
    field[0, 0, 0] = 0
    with pytest.raises(ValueError, match="positive"):
        log_density(field)
    transformed = log_density(field, zero_floor_count=0.5)
    assert np.isfinite(transformed).all()


def test_log_comparison_is_identity_for_same_field():
    coordinate = np.arange(80, dtype=np.float64)
    field = 1.0 + (
        coordinate[:, None, None]
        + 2 * coordinate[None, :, None]
        + 3 * coordinate[None, None, :]
    )
    result = compare_log_fields(field, field)
    assert result["voxel_pearson_r"] == pytest.approx(1.0)
    assert result["candidate_minus_reference_rms_dex"] == 0.0
    assert all(value == 1.0 for value in result["candidate_over_reference_power"].values())


def test_power_bands_reject_wrong_grid():
    with pytest.raises(ValueError, match="expected"):
        isotropic_power_bands(np.ones((8, 8, 8)))


def test_expected_cic_smoothing_is_periodic_and_mass_conserving():
    counts = np.zeros((4, 4, 4))
    counts[0, 0, 0] = 1.0
    smoothed = smooth_ngp_as_expected_cic(counts)
    assert smoothed.sum() == pytest.approx(1.0, abs=1e-14)
    assert smoothed[0, 0, 0] == pytest.approx(0.75**3)
    assert smoothed[3, 0, 0] == pytest.approx(0.125 * 0.75**2)
    assert np.count_nonzero(smoothed) == 27


def test_expected_cic_smoothing_rejects_invalid_field():
    with pytest.raises(ValueError, match="cubic"):
        smooth_ngp_as_expected_cic(np.ones((2, 3, 2)))
    counts = np.ones((2, 2, 2))
    counts[0, 0, 0] = -1
    with pytest.raises(ValueError, match="non-negative"):
        smooth_ngp_as_expected_cic(counts)
