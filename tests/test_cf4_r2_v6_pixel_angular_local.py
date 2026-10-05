import healpy as hp
import numpy as np
import pytest

from cf4_r2_v6_pixel_angular_geometry import (
    OBSERVER,
    pixel_completeness,
    source_volume_rule,
)


def test_active_gl2_source_volume_rule_is_normalized_and_cell_local():
    offsets, weights = source_volume_rule()
    assert offsets.shape == (8, 3)
    assert weights.shape == (8,)
    assert np.isclose(weights.sum(), 1.0, rtol=0.0, atol=2e-15)
    assert np.all(np.abs(offsets) < 0.75)
    assert np.allclose(offsets.mean(axis=0), 0.0, rtol=0.0, atol=1e-15)


def test_pixel_completeness_uses_same_rings_pixel_for_both_maps():
    nside = 1
    pixels = hp.nside2npix(nside)
    maps = [np.linspace(0.0, 1.0, pixels), np.linspace(1.0, 0.0, pixels)]
    points = np.asarray([OBSERVER + [1.0, 0.0, 0.0],
                         OBSERVER + [0.0, -1.0, 0.0]])
    actual = pixel_completeness(points, maps, np.eye(3), nside=nside)
    pix = hp.vec2pix(nside, [1.0, 0.0], [0.0, -1.0], [0.0, 0.0], nest=False)
    expected = np.stack([m[pix] for m in maps])
    assert np.array_equal(actual, expected)


def test_pixel_completeness_rejects_observer_coincident_direction():
    maps = [np.ones(12), np.ones(12)]
    with pytest.raises(ValueError, match="observer-coincident"):
        pixel_completeness(np.asarray([OBSERVER]), maps, np.eye(3), nside=1)
