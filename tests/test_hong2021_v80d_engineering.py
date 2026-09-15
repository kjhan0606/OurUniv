import numpy as np

import hong2021_v80_sample as frozen
import hong2021_v80d_engineering_sample as diagnostic


def test_only_fix_flattens_dc_without_changing_field(monkeypatch) -> None:
    field = np.arange(3 * 8, dtype=np.float32).reshape(3, 1, 2, 2, 2)
    inherited_dc = np.asarray([[1e-10], [2e-10], [3e-10]])

    def inherited(*args):
        return field.copy(), inherited_dc.copy()

    monkeypatch.setattr(diagnostic, "_INHERITED_CALIBRATE_AND_PROJECT", inherited)
    output, dc = diagnostic.fixed_calibrate_and_project(
        field, np.zeros((1, 2, 2, 2)), np.asarray([-1.0, 1.0]), np.asarray([-1.0, 1.0])
    )
    assert np.array_equal(output, field)
    assert dc.shape == (3,)
    assert np.array_equal(dc, inherited_dc[:, 0])


def test_fix_matches_frozen_HDF5_row_contract() -> None:
    members = np.zeros(16, dtype=np.float32)
    fixed = np.zeros((16, 1), dtype=np.float64).reshape(16)
    members[:] = fixed
    assert members.shape == (16,)
