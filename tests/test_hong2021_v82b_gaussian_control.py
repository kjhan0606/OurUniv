from pathlib import Path

import numpy as np

import hong2021_v82b_gaussian_control as control


def test_loo_periodic_power_excludes_current_query() -> None:
    grid = 8
    shells = control.PeriodicShells(grid, 1.0)
    x = np.arange(grid, dtype=np.float64)[:, None, None]
    fields = np.stack(
        [np.broadcast_to(np.sin((index + 1) * 2 * np.pi * x / grid), (grid,) * 3) for index in range(4)]
    )
    power, sums = control.loo_periodic_mode_power(fields, shells)
    expected = (sums.sum(axis=0)[None] - sums) / (3 * shells.count[None])
    expected[:, 0] = 0.0
    np.testing.assert_allclose(power, expected)


def test_gaussian_draw_is_deterministic_and_has_exact_zero_DC() -> None:
    shells = control.PeriodicShells(8, 1.0)
    mode_power = np.linspace(0.0, 2.0, shells.shell_count)
    first = control.gaussian_residual_draws(
        np.random.default_rng(9), mode_power, shells, members=3
    )
    second = control.gaussian_residual_draws(
        np.random.default_rng(9), mode_power, shells, members=3
    )
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first.mean(axis=(-3, -2, -1)), 0.0, atol=1e-15)


def test_loo_quartile_scales_preserve_equal_weight_global_variance() -> None:
    sums = np.asarray([[1.0, 4.0, 9.0, 16.0], [2.0, 8.0, 18.0, 32.0]])
    counts = np.full((2, 4), 10)
    scale, variance = control.loo_quartile_scales(sums, counts)
    np.testing.assert_allclose(np.mean(np.square(scale), axis=1), 1.0)
    assert np.all(variance > 0.0)


def test_quartile_scale_is_paired_and_restores_DC() -> None:
    residual = np.ones((2, 4, 4, 4), dtype=np.float64)
    mean = np.arange(64, dtype=np.float64).reshape(4, 4, 4)
    scaled = control.apply_quartile_scale(
        residual, mean, np.asarray([0.5, 1.0, 1.5, 2.0])
    )
    np.testing.assert_allclose(scaled.mean(axis=(-3, -2, -1)), 0.0)
    np.testing.assert_array_equal(scaled[0], scaled[1])


def _decision_payload(stationary: float, quartile: float) -> dict:
    return {
        domain: {
            "baselines": {
                "stationary": {
                    "exact_rank_coverage_null": {
                        "conditional_p_values": {
                            "rank_tv": stationary,
                            "coverage_deviation": stationary,
                        }
                    }
                },
                "quartile_scale": {
                    "exact_rank_coverage_null": {
                        "conditional_p_values": {
                            "rank_tv": quartile,
                            "coverage_deviation": quartile,
                        }
                    }
                },
            }
        }
        for domain in control.DOMAIN_ORDER
    }


def test_decision_requires_nonGaussian_model_only_when_both_controls_fail() -> None:
    assert control.decide(_decision_payload(0.5, 0.5))[
        "conditional_flow_or_equivalent_nonGaussian_model_warranted"
    ] is False
    conditional = control.decide(_decision_payload(0.001, 0.5))
    assert conditional["branch"].startswith("conditional_scale_required")
    failed = control.decide(_decision_payload(0.001, 0.001))
    assert failed["conditional_flow_or_equivalent_nonGaussian_model_warranted"] is True


def test_source_does_not_train_or_write_HDF5() -> None:
    source = Path(control.__file__).read_text()
    assert "torch" not in source
    assert "create_dataset" not in source
    assert 'h5py.File(path, "r")' in source
