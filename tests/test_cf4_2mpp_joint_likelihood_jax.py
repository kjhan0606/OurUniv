from __future__ import annotations

import numpy as np
import pytest

import cf4_2mpp_joint_likelihood_jax as jax_kernel
from cf4_2mpp_joint_likelihood_local import (
    LikelihoodInputError,
    observer_centred_spherical_rsd,
    predict_selected_intensity,
)


def _case():
    positions = np.array([[0.2, 0.4, 0.7], [2.1, 1.3, 4.8], [5.2, 5.5, 0.9]], dtype=np.float64)
    velocities = np.array([[30.0, -10.0, 5.0], [-20.0, 4.0, 11.0], [3.0, 14.0, -8.0]], dtype=np.float64)
    masses = np.full((6, 3), 0.5, dtype=np.float64)
    exposure = np.full((6, 8, 8, 8), 0.8, dtype=np.float64)
    kwargs = dict(
        observer=np.array([3.0, 3.0, 3.0]), box_size_cMpc_h=6.0,
        hubble_km_s_Mpc=100.0, little_h=0.746, scale_factor=1.0,
        sigma_fog_km_s=np.full(6, 20.0), sigma_redshift_km_s=np.full(6, 10.0),
    )
    return positions, velocities, masses, exposure, kwargs


def test_jax_optional_import_is_explicitly_fail_closed_when_unavailable():
    if jax_kernel.jax is not None:
        pytest.skip("JAX is installed; comparison tests cover the available path")
    with pytest.raises(jax_kernel.JaxUnavailable):
        jax_kernel.poisson_log_likelihood_jax(np.ones((1,)), np.ones((1,)))


def test_checked_jax_bridge_rejects_independent_redshift_factor_before_kernel():
    counts = np.zeros((6, 2, 2, 2), dtype=np.int64)
    intensity = np.ones_like(counts, dtype=np.float64)
    with pytest.raises(LikelihoodInputError, match=r"independent 2M\+\+ redshift factor"):
        jax_kernel.joint_log_likelihood_jax_checked(
            counts,
            intensity,
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
            np.array([1.0, 1.0]),
            np.array([0, 0], dtype=np.int64),
            np.array([0.5]),
            secure_object_ids=["A", "B"],
            independent_twompp_redshift_ids=["external-factor"],
        )


def test_checked_jax_bridge_requires_source_bound_manifest_for_arbitrary_ids():
    counts = np.zeros((6, 2, 2, 2), dtype=np.int64)
    intensity = np.ones_like(counts, dtype=np.float64)
    with pytest.raises(LikelihoodInputError, match="source-bound crossmatch manifest"):
        jax_kernel.joint_log_likelihood_jax_checked(
            counts,
            intensity,
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
            np.array([1.0, 1.0]),
            np.array([0, 0], dtype=np.int64),
            np.array([0.5]),
            secure_object_ids=["arbitrary-a", "arbitrary-b"],
            twompp_object_ids=["2mpp:a", "2mpp:b"],
        )


@pytest.mark.skipif(jax_kernel.jax is None, reason="JAX is not installed")
def test_jax_intensity_matches_numpy_oracle_and_has_finite_gradient():
    jax_kernel.jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    positions, velocities, masses, exposure, kwargs = _case()
    expected = predict_selected_intensity(positions, velocities, masses, exposure, **kwargs)
    actual = jax_kernel.predict_selected_intensity_jax(
        jnp.asarray(positions), jnp.asarray(velocities), jnp.asarray(masses),
        jnp.asarray(exposure), **{key: jnp.asarray(value) if isinstance(value, np.ndarray) else value for key, value in kwargs.items()}
    )
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-12, atol=1e-12)
    value, gradient = jax_kernel.jax.value_and_grad(
        lambda alpha: jnp.sum(jax_kernel.predict_selected_intensity_jax(
            jnp.asarray(positions), jnp.asarray(velocities),
            jnp.asarray(masses), jnp.asarray(exposure),
            **{**kwargs, "sigma_fog_km_s": jnp.asarray(kwargs["sigma_fog_km_s"]), "sigma_redshift_km_s": jnp.asarray(kwargs["sigma_redshift_km_s"])},
        ) * alpha[:, None, None, None])
    )(jnp.ones((6,), dtype=jnp.float64))
    assert np.isfinite(float(value))
    assert np.all(np.isfinite(np.asarray(gradient)))


@pytest.mark.skipif(jax_kernel.jax is None, reason="JAX is not installed")
def test_jax_rsd_conversion_matches_numpy_units():
    jax_kernel.jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    positions, velocities, _, _, kwargs = _case()
    np_result = observer_centred_spherical_rsd(positions, velocities, kwargs["observer"], 6.0, 100.0, little_h=0.746, scale_factor=1.0)
    shifted, displacement, _ = jax_kernel.observer_centred_spherical_rsd_jax(
        jnp.asarray(positions), jnp.asarray(velocities), jnp.asarray(kwargs["observer"]),
        6.0, 100.0, little_h=0.746, scale_factor=1.0,
    )
    np.testing.assert_allclose(np.asarray(shifted), np_result.positions, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(displacement), np_result.coherent_displacement_cMpc_h, rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(jax_kernel.jax is None, reason="JAX is not installed")
def test_zero_exposure_and_observer_coincident_inputs_remain_finite():
    jax_kernel.jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    positions, velocities, masses, exposure, kwargs = _case()
    positions[0] = kwargs["observer"]
    shifted, displacement, rhat = jax_kernel.observer_centred_spherical_rsd_jax(
        jnp.asarray(positions), jnp.asarray(velocities), jnp.asarray(kwargs["observer"]),
        6.0, 100.0, little_h=.746, scale_factor=1.0)
    assert np.isfinite(np.asarray(shifted)).all() and np.isfinite(np.asarray(displacement)).all()
    counts = jnp.zeros((2,))
    intensity = jnp.asarray([0.0, 1.0])
    value, gradient = jax_kernel.jax.value_and_grad(lambda z: jax_kernel.poisson_log_likelihood_jax(counts, z).sum())(intensity)
    assert np.isfinite(float(value)) and np.isfinite(np.asarray(gradient)).all()


@pytest.mark.skipif(jax_kernel.jax is None, reason="JAX is not installed")
def test_rhat_uses_shifted_position_across_periodic_wrap():
    """JAX stochastic LOS must match the NumPy oracle after coherent RSD."""
    jax_kernel.jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    observer = np.array([3.0, 3.0, 3.0])
    positions = np.array([[5.9, 3.0, 3.0], [0.1, 3.0, 3.0]], dtype=np.float64)
    velocities = np.array([[120.0, 0.0, 0.0], [-120.0, 0.0, 0.0]], dtype=np.float64)
    shifted, displacement, rhat = jax_kernel.observer_centred_spherical_rsd_jax(
        jnp.asarray(positions), jnp.asarray(velocities), jnp.asarray(observer),
        6.0, 100.0, little_h=0.746, scale_factor=1.0,
    )
    shifted_np = np.asarray(shifted)
    relative = (shifted_np - observer + 3.0) % 6.0 - 3.0
    expected = relative / np.linalg.norm(relative, axis=1)[:, None]
    np.testing.assert_allclose(np.asarray(rhat), expected, rtol=1e-12, atol=1e-12)
    assert np.isfinite(np.asarray(displacement)).all()
