from pathlib import Path

import numpy as np

import hong2021_v82c_location_scale_control as control


def test_loo_quartile_location_excludes_current_query() -> None:
    sums = np.asarray([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    counts = np.full((2, 4), 2)
    location = control.loo_quartile_location(sums, counts)
    np.testing.assert_allclose(location[0], sums[1] / 2)
    np.testing.assert_allclose(location[1], sums[0] / 2)


def test_location_scale_adds_ordered_location_and_restores_DC() -> None:
    residual = np.zeros((2, 4, 4, 4), dtype=np.float64)
    mean = np.arange(64, dtype=np.float64).reshape(4, 4, 4)
    generated = control.apply_quartile_location_scale(
        residual,
        mean,
        np.asarray([1.0, 2.0, 3.0, 4.0]),
        np.ones(4),
    )
    np.testing.assert_allclose(generated.mean(axis=(-3, -2, -1)), 0.0)
    assert generated[0, 0, 0, 0] < generated[0, -1, -1, -1]
    np.testing.assert_array_equal(generated[0], generated[1])


def test_decision_requires_all_six_values_above_reference() -> None:
    passing = {
        domain: {
            "exact_rank_coverage_null": {
                "conditional_p_values": {"rank_tv": 0.5, "coverage_deviation": 0.5}
            }
        }
        for domain in control.DOMAIN_ORDER
    }
    assert control.decide(passing)["branch"].endswith("no_flow")
    passing["TNG100"]["exact_rank_coverage_null"]["conditional_p_values"][
        "rank_tv"
    ] = 0.001
    failed = control.decide(passing)
    assert failed["conditional_flow_or_equivalent_nonGaussian_model_warranted"] is True


def test_source_is_consumed_read_only_without_training() -> None:
    source = Path(control.__file__).read_text()
    assert "torch" not in source
    assert "create_dataset" not in source
    assert 'h5py.File(path, "r")' in source
    assert "V82B_quartile_scale_histogram_crosscheck_pass" in source
