import sys
from pathlib import Path

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hong2021_data import (  # noqa: E402
    cyclic_flip_transform,
    grid_uncertainty_aware_observables,
)
from hong2021_model import Hong2021Net  # noqa: E402
from hong2021_prepare_tng import (  # noqa: E402
    choose_spatial_split,
    extract_periodic_cube,
)


def test_all_24_paper_augmentations_preserve_values() -> None:
    cube = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    outputs = [
        cyclic_flip_transform(cube, permutation, flips)
        for permutation in range(3)
        for flips in range(8)
    ]
    assert len(outputs) == 24
    for output in outputs:
        np.testing.assert_array_equal(np.sort(output.ravel()), cube.ravel())


def test_precision_weighted_velocity_statistics() -> None:
    position = np.array([[1.0, 0.0, 1.0], [1.01, 0.0, 1.01]])
    velocity = np.array([[100.0, 0.0, 100.0], [200.0, 0.0, 200.0]])
    error = np.array([100.0, 200.0])
    count, mean, sigma_mean, scatter = grid_uncertainty_aware_observables(
        position, velocity, error, grid=8, box_mpc_h=4.0, mask_abs_b_deg=0.0
    )
    cell = np.unravel_index(np.argmax(count), count.shape)
    assert count[cell] == 2
    np.testing.assert_allclose(mean[cell], 169.705627, rtol=1e-6)
    np.testing.assert_allclose(sigma_mean[cell], 89.442719, rtol=1e-6)
    np.testing.assert_allclose(scatter[cell], 100.0, rtol=1e-6)


def test_three_channel_extension_preserves_grid_shape() -> None:
    model = Hong2021Net(in_channels=3, channels=(2, 4, 8, 16, 32))
    model.eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 64, 64, 64))
    assert output.shape == (1, 1, 64, 64, 64)


def test_periodic_cube_extraction_wraps_all_axes() -> None:
    field = np.arange(8**3).reshape(8, 8, 8)
    cube = extract_periodic_cube(field, np.array([6, 7, 5]), size=4)
    expected = field[np.ix_([6, 7, 0, 1], [7, 0, 1, 2], [5, 6, 7, 0])]
    np.testing.assert_array_equal(cube, expected)


def test_spatial_split_has_no_cross_cube_overlap() -> None:
    generator = np.random.default_rng(7)
    validation_cluster = generator.uniform(4.0, 8.0, size=(8, 3))
    training_cluster = generator.uniform(34.0, 38.0, size=(12, 3))
    positions = np.concatenate((validation_cluster, training_cluster))
    training, validation, metadata = choose_spatial_split(
        positions, n_train=10, n_validation=6, seed=9
    )
    delta = np.abs(
        positions[training][:, None, :] - positions[validation][None, :, :]
    )
    delta = np.minimum(delta, 75.0 - delta)
    assert not np.any(np.all(delta < 20.0, axis=2))
    assert metadata["available_nonoverlapping_training_centers"] >= 10
