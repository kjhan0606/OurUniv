import sys
from pathlib import Path

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hong2021_data import (  # noqa: E402
    cyclic_flip_transform,
    grid_uncertainty_aware_observables,
)
from hong2021_augmentation import (  # noqa: E402
    CUBE_ISOMETRIES,
    apply_cube_isometry,
    is_mirror_isometry,
)
from hong2021_balanced_split import distribution_metrics  # noqa: E402
from hong2021_evaluate import (  # noqa: E402
    OpenBoundaryTwoPoint,
    ks_statistic,
    summarize_ks,
)
from hong2021_model import Hong2021Net, group_count  # noqa: E402
from hong2021_joint_split_search import balance_objective  # noqa: E402
from hong2021_paper_visual import central_slab, slab_mean, slab_sum  # noqa: E402
from hong2021_prepare_tng import (  # noqa: E402
    choose_spatial_split,
    extract_periodic_cube,
)
from hong2021_spectral_diagnostics import (  # noqa: E402
    contrast_pair,
    fourier_diagnostics,
)
from hong2021_residual_diffusion import (  # noqa: E402
    ConditionalResidualUNet,
    cosine_beta_schedule,
    highpass_mask,
    highpass_numpy,
    radial_geometry,
)
from hong2021_residual_v6 import (  # noqa: E402
    edm_coefficients,
    karras_sigmas,
    laplacian_residual,
)
from hong2021_train import apply_input_preprocessing  # noqa: E402


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


def test_full_cube_augmentation_has_24_rotations_and_24_mirrors() -> None:
    cube = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    outputs = [
        apply_cube_isometry(cube, permutation, reflections)
        for permutation, reflections in CUBE_ISOMETRIES
    ]
    assert len(outputs) == 48
    assert sum(
        is_mirror_isometry(permutation, reflections)
        for permutation, reflections in CUBE_ISOMETRIES
    ) == 24
    assert len({output.tobytes() for output in outputs}) == 48
    for output in outputs:
        np.testing.assert_array_equal(np.sort(output.ravel()), cube.ravel())


def test_cube_isometry_transforms_all_channels_identically() -> None:
    cube = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    channels = np.stack((cube, cube + 1000))
    transformed = apply_cube_isometry(
        channels, (2, 0, 1), (True, False, True)
    )
    np.testing.assert_array_equal(transformed[1] - transformed[0], 1000)


def test_residual_highpass_removes_constant_and_preserves_small_scale() -> None:
    mask = highpass_mask(16, 0.3125, 2.0, 4.0)
    constant = np.ones((16, 16, 16), dtype=np.float32)
    np.testing.assert_allclose(highpass_numpy(constant, mask), 0.0, atol=1e-6)
    checker = np.indices((16, 16, 16)).sum(axis=0) % 2
    filtered = highpass_numpy(checker.astype(np.float32), mask)
    assert float(filtered.std()) > 0.45
    assert abs(float(filtered.mean())) < 1e-6


def test_radial_geometry_is_invariant_under_all_cube_isometries() -> None:
    radius = radial_geometry(16)
    for permutation, reflections in CUBE_ISOMETRIES:
        np.testing.assert_array_equal(
            apply_cube_isometry(radius, permutation, reflections), radius
        )


def test_residual_diffusion_model_preserves_grid_shape() -> None:
    model = ConditionalResidualUNet(condition_channels=4, base_channels=4)
    residual = torch.zeros((2, 1, 16, 16, 16))
    condition = torch.zeros((2, 4, 16, 16, 16))
    time = torch.tensor([0.1, 0.9])
    assert model(residual, condition, time).shape == residual.shape


def test_cosine_diffusion_schedule_is_valid() -> None:
    beta = cosine_beta_schedule(20)
    assert beta.shape == (20,)
    assert torch.all((beta > 0) & (beta < 1))
    alpha_bar = torch.cumprod(1.0 - beta, dim=0)
    assert torch.all(alpha_bar[1:] < alpha_bar[:-1])


def test_laplacian_residual_is_nonperiodic_and_removes_constant() -> None:
    constant = np.ones((1, 16, 16, 16), dtype=np.float32)
    np.testing.assert_allclose(laplacian_residual(constant, 2.0), 0.0, atol=1e-7)
    impulse = np.zeros_like(constant)
    impulse[0, 0, 8, 8] = 1.0
    residual = laplacian_residual(impulse, 2.0)
    assert abs(float(residual[0, -1, 8, 8])) < 1e-5
    assert abs(float(residual.mean())) < 1e-7


def test_edm_preconditioning_has_correct_small_noise_limit() -> None:
    sigma = torch.tensor([1.0e-6, 1.0, 10.0])
    c_skip, c_out, c_in, c_noise = edm_coefficients(sigma, sigma_data=1.0)
    assert c_skip[0] > 0.999
    assert c_out[0] < 1.0e-5
    assert torch.isfinite(c_in).all()
    assert torch.isfinite(c_noise).all()


def test_karras_sigma_schedule_is_monotone_and_ends_at_zero() -> None:
    sigma = karras_sigmas(20, 0.002, 40.0, 7.0, torch.device("cpu"))
    assert sigma.shape == (21,)
    torch.testing.assert_close(sigma[0], torch.tensor(40.0))
    assert sigma[-2] > 0
    torch.testing.assert_close(sigma[-1], torch.tensor(0.0))
    assert torch.all(sigma[1:] <= sigma[:-1])


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


def test_occupied_input_standardization_preserves_empty_cells() -> None:
    value = np.array(
        [
            [[[0.0, 2.0], [4.0, 0.0]]],
            [[[0.0, 100.0], [0.0, 999.0]]],
        ],
        dtype=np.float32,
    )
    spec = {
        "mode": "occupied_standardized",
        "count_scale": 2.0,
        "velocity_occupied_mean_kms": 50.0,
        "velocity_occupied_std_kms": 25.0,
    }
    result = apply_input_preprocessing(value, spec)
    np.testing.assert_allclose(result[0], [[[0.0, 1.0], [2.0, 0.0]]])
    np.testing.assert_allclose(result[1], [[[0.0, 2.0], [-2.0, 0.0]]])


def test_group_count_uses_largest_divisor_at_most_32() -> None:
    assert group_count(2048) == 32
    assert group_count(128) == 32
    assert group_count(130) == 26
    assert group_count(7) == 7


def test_groupnorm_is_independent_of_batch_companions_and_mode() -> None:
    torch.manual_seed(19)
    model = Hong2021Net(
        channels=(2, 4, 8, 16, 32),
        normalization="group",
    )
    target = torch.randn(1, 2, 64, 64, 64)
    companions = torch.randn(2, 2, 64, 64, 64)
    with torch.inference_mode():
        model.train()
        alone_train = model(target)
        with_companions = model(torch.cat((target, companions)))[0:1]
        model.eval()
        alone_eval = model(target)
    torch.testing.assert_close(alone_train, with_companions)
    torch.testing.assert_close(alone_train, alone_eval)


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


def test_open_boundary_two_point_constant_mean_density_is_zero() -> None:
    estimator = OpenBoundaryTwoPoint(grid=4, voxel_mpc_h=1.0, rmax_mpc_h=2.0)
    result = estimator(np.zeros((4, 4, 4)))
    np.testing.assert_allclose(result, 0.0, atol=1e-14)


def test_open_boundary_two_point_matches_direct_zero_lag() -> None:
    field = np.arange(4**3, dtype=np.float64).reshape(4, 4, 4) / 10.0
    estimator = OpenBoundaryTwoPoint(grid=4, voxel_mpc_h=1.0, rmax_mpc_h=2.0)
    result = estimator(field)
    np.testing.assert_allclose(result[0], np.mean(field**2), rtol=1e-12)


def test_ks_summary_recovers_identical_and_disjoint_samples() -> None:
    radius = (np.arange(32) + 0.5) * 0.3125
    truth = np.tile(np.arange(5, dtype=np.float64)[:, None], (1, 32))
    identical = summarize_ks(truth, truth.copy(), radius)
    assert identical["mean_0_10_mpc_h"] == 0.0
    shifted = summarize_ks(truth, truth + 10.0, radius)
    assert shifted["mean_0_10_mpc_h"] == 1.0
    assert ks_statistic(np.arange(5.0), np.arange(5.0)) == 0.0


def test_paper_visual_central_slab_projection() -> None:
    cube = np.arange(6**3).reshape(6, 6, 6)
    np.testing.assert_array_equal(central_slab(cube, axis=0, cells=2), cube[2:4])
    np.testing.assert_array_equal(slab_sum(cube, axis=0, cells=2), cube[2:4].sum(0))
    np.testing.assert_allclose(slab_mean(cube, axis=2, cells=4), cube[:, :, 1:5].mean(2))


def test_split_balance_metrics_identical_distributions() -> None:
    values = np.arange(24, dtype=np.float64).reshape(4, 6)
    metrics = distribution_metrics(values, values.copy())
    np.testing.assert_allclose(metrics["standardized_mean_difference"], 0.0)
    np.testing.assert_allclose(metrics["ks_distance"], 0.0)
    assert metrics["all_abs_smd_below_0.25"]
    assert balance_objective(values, values.copy()) == 0.0


def test_spectral_diagnostics_identical_fields() -> None:
    generator = np.random.default_rng(13)
    density = np.exp(generator.normal(size=(2, 8, 8, 8))).astype(np.float32)
    result = fourier_diagnostics(density, density.copy(), voxel_mpc_h=0.5)
    np.testing.assert_allclose(result["transfer_sqrt_Ppred_over_Ptruth"], 1.0)
    np.testing.assert_allclose(result["cross_correlation_r"], 1.0)
    np.testing.assert_allclose(result["residual_to_truth_power"], 0.0)


def test_two_point_contrast_own_cube_means_are_zero_mean() -> None:
    truth = np.arange(1, 9, dtype=np.float64).reshape(2, 2, 2)
    prediction = truth * 3.0
    truth_delta, prediction_delta = contrast_pair(
        truth,
        prediction,
        "own_cube_mean",
        float(truth.mean()),
        float(prediction.mean()),
    )
    np.testing.assert_allclose(truth_delta.mean(), 0.0, atol=1e-15)
    np.testing.assert_allclose(prediction_delta.mean(), 0.0, atol=1e-15)
