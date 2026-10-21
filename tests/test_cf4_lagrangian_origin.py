import numpy as np

from src.cf4_lagrangian_origin import periodic_delta


def test_periodic_delta_uses_short_wrapped_displacement():
    got = periodic_delta(np.array([1.0, 383.0, 10.0]), np.array([383.0, 1.0, 12.0]), 384.0)
    np.testing.assert_allclose(got, [2.0, -2.0, -2.0])
