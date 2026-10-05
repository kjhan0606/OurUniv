import healpy as hp
import numpy as np
import pytest

from cf4_r2_v6_pixel_angular_geometry import (
    OBSERVER,
    flatten_cell_indices,
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


def test_flattened_ids_index_their_declared_cells_not_candidate_ranks():
    ijk = np.asarray([[2, 1, 4], [0, 0, 1]], dtype=np.int32)
    flat_ids = flatten_cell_indices(ijk, n=5)
    index_coded_field = np.arange(5**3, dtype=np.int64).reshape(5, 5, 5)

    assert np.array_equal(flat_ids, np.asarray([59, 1]))
    assert np.array_equal(index_coded_field.reshape(-1)[flat_ids],
                          index_coded_field[ijk[:, 0], ijk[:, 1], ijk[:, 2]])


def test_known_n256_control_has_the_full_grid_flattened_id():
    ijk = np.asarray([[126, 124, 139]], dtype=np.int32)
    assert flatten_cell_indices(ijk, n=256).tolist() == [8_289_419]


def test_flattened_ids_reject_out_of_grid_cells():
    with pytest.raises(ValueError, match="outside the grid"):
        flatten_cell_indices(np.asarray([[256, 0, 0]], dtype=np.int32), n=256)
