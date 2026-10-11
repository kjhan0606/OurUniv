import numpy as np

from hong2021_transfer_likelihood import (
    binned_spectra,
    derived,
    k_geometry,
    tukey,
    windows,
)


def test_tukey_is_symmetric_and_flat_inside():
    value = tukey(64, 0.25)
    np.testing.assert_allclose(value, value[::-1])
    assert value[0] == 0.0
    assert value[32] == 1.0


def test_exact_linear_predictor_has_exact_transfer_and_zero_noise():
    generator = np.random.default_rng(7)
    truth = generator.normal(size=(32, 32, 32))
    prediction = 0.73 * truth
    names, analysis_windows = windows(32, 1.0, 0.25)
    edges = np.asarray([0.2, 0.5, 1.0, 2.0, 3.1])
    index, valid, geometry = k_geometry(32, 1.0, edges)
    assert len(names) == 4
    assert np.all(geometry[:, 1] > 0)
    spectra = binned_spectra(
        truth, prediction, analysis_windows, index, valid, len(edges) - 1
    )
    result = derived(spectra)
    np.testing.assert_allclose(result["transfer"], 0.73, atol=1e-12)
    np.testing.assert_allclose(result["noise_power"], 0.0, atol=1e-10)
    np.testing.assert_allclose(result["coherence_squared"], 1.0, atol=1e-12)
