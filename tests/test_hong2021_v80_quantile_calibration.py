import hashlib
import json
from pathlib import Path

import numpy as np

import hong2021_v80_quantile_calibration as calibration


REPO = Path(__file__).resolve().parents[1]
PROGRAM = REPO / "config/hong2021_v80_consumed_development_quantile_calibration_program.json"


def test_program_is_byte_bound_and_has_no_V79_payload_authorization() -> None:
    assert hashlib.sha256(PROGRAM.read_bytes()).hexdigest() == calibration.PROGRAM_SHA256
    program = json.loads(PROGRAM.read_text())
    assert program["schema"] == calibration.PROGRAM_SCHEMA
    assert program["status"] == calibration.PROGRAM_STATUS
    assert program["frozen_calibration_algorithm"]["histogram_bins"] == 65536
    assert program["authorization"]["read_any_V79_selected_input_or_target_during_calibration"] is False
    assert program["authorization"]["candidate_design_change_after_training_diagnostics"] is False


def test_histogram_conserves_values_and_rejects_out_of_range() -> None:
    edges = calibration.histogram_edges()
    counts = np.zeros(calibration.BINS, dtype=np.int64)
    values = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])
    calibration.add_histogram(counts, values, edges)
    assert counts.sum() == len(values)
    try:
        calibration.add_histogram(counts, np.asarray([2.01]), edges)
    except ValueError as error:
        assert "outside frozen range" in str(error)
    else:
        raise AssertionError("out-of-range calibration value was accepted")


def test_monotone_map_matches_shifted_empirical_distribution(monkeypatch) -> None:
    monkeypatch.setattr(calibration, "BINS", 16)
    monkeypatch.setattr(calibration, "MINIMUM_Y", -2.0)
    monkeypatch.setattr(calibration, "MAXIMUM_Y", 2.0)
    edges = calibration.histogram_edges()
    source = np.zeros(16, dtype=np.int64)
    truth = np.zeros(16, dtype=np.int64)
    source[4:8] = [1, 2, 2, 1]
    truth[6:10] = [1, 2, 2, 1]
    x, y = calibration.fit_monotone_map(source, truth, edges)
    assert np.all(np.diff(x) > 0)
    assert np.all(np.diff(y) >= 0)
    assert np.allclose(y - x, 0.5)


def test_map_uses_unit_slope_extrapolation_without_clipping() -> None:
    source = np.asarray([-1.0, 0.0, 1.0])
    mapped = np.asarray([-0.5, 0.0, 0.5])
    values = np.asarray([-2.0, -1.0, 0.5, 1.0, 2.0])
    result = calibration.apply_monotone_map(values, source, mapped)
    assert np.allclose(result, [-1.5, -0.5, 0.25, 0.5, 1.5])
    assert result[0] != result[1]
    assert result[-1] != result[-2]


def test_map_and_project_restores_each_cube_dc() -> None:
    rng = np.random.default_rng(80001)
    mean = rng.normal(scale=0.05, size=(1, 4, 4, 4))
    sample = mean + rng.normal(scale=0.1, size=(3, 1, 4, 4, 4))
    source = np.asarray([-1.0, 0.0, 1.0])
    mapped = np.asarray([-0.8, 0.1, 0.9])
    output, maximum = calibration.map_and_project(sample, mean, source, mapped)
    residual = output.astype(np.float64) - mean
    assert output.dtype == np.float32
    assert maximum <= 1e-8
    assert np.max(np.abs(residual.mean(axis=(-3, -2, -1)))) <= 1e-8


def test_histogram_quantile_is_ordered() -> None:
    edges = np.linspace(-2.0, 2.0, 9)
    counts = np.asarray([0, 1, 2, 3, 4, 3, 2, 1])
    values = [calibration.histogram_quantile(counts, edges, q) for q in (0.1, 0.5, 0.9)]
    assert values[0] < values[1] < values[2]
