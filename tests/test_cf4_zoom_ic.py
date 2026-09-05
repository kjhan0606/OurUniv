import numpy as np

from src.cf4_zoom_ic import fourier_resample_field


def test_fourier_resample_physical_field_preserves_shared_modes():
    rng = np.random.default_rng(17)
    source = rng.standard_normal((12, 12, 12)).astype(np.float32)
    target = fourier_resample_field(source, 8)
    source_fft = np.fft.rfftn(source)
    target_fft = np.fft.rfftn(target)
    scale = (8.0 / 12.0) ** 3
    np.testing.assert_allclose(target_fft[1:4, 1:4, 1:4],
                               source_fft[1:4, 1:4, 1:4] * scale,
                               rtol=3e-6, atol=3e-5)
