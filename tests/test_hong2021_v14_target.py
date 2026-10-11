import numpy as np
import pytest

from hong2021_v14_target import log_cic_target


def test_log_cic_target_preserves_finite_values_outside_legacy_guard() -> None:
    density = np.array([1.0e-5, 32.768, 3.0e6], dtype=np.float32)
    actual = log_cic_target(density, 32.768)
    expected = np.log10(density / 32.768) / 4.5
    np.testing.assert_allclose(actual, expected, rtol=1.0e-7, atol=0.0)
    assert actual[0] < -1.0
    assert actual[-1] > 1.0


@pytest.mark.parametrize("bad", (0.0, -1.0, np.nan, np.inf))
def test_log_cic_target_rejects_nonpositive_or_nonfinite_density(bad: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        log_cic_target(np.array([1.0, bad]), 1.0)
