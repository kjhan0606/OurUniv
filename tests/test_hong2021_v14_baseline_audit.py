import numpy as np
import pytest

from hong2021_v14_baseline_audit import centered_fourier_band_rms


def test_fourier_bands_exhaust_centered_residual_power():
    coordinate = np.arange(64, dtype=np.float64)
    field = (
        0.2
        + np.sin(2 * np.pi * coordinate[:, None, None] / 64)
        + 0.5 * np.cos(10 * 2 * np.pi * coordinate[None, :, None] / 64)
    )
    field = np.broadcast_to(field, (64, 64, 64)).copy()
    dc, bands, centered = centered_fourier_band_rms(field, voxel_mpc_h=0.3125)
    assert dc == pytest.approx(0.2)
    assert bands.shape == (4,)
    assert np.square(bands).sum() == pytest.approx(np.mean(centered**2), rel=1e-6)
    assert bands[0] == pytest.approx(np.sqrt(0.5), rel=1e-6)
    assert bands[2] == pytest.approx(np.sqrt(0.125), rel=1e-6)
    assert bands[1] < 1e-12
    assert bands[3] < 1e-12


def test_fourier_diagnostic_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="cubic"):
        centered_fourier_band_rms(np.ones((3, 4, 3)), voxel_mpc_h=1)
    with pytest.raises(ValueError, match="increase"):
        centered_fourier_band_rms(
            np.ones((4, 4, 4)), voxel_mpc_h=1, edges=(0, 2, 1)
        )
