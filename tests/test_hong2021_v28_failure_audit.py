from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from hong2021_v18_init import sha256_file
from hong2021_v28_failure_audit import (
    AUDIT_PROGRAM_SHA256,
    PhysicalTailAccumulator,
    classify,
    compare_physical_tails,
)


REPO = Path(__file__).parents[1]


def test_v28_failure_audit_program_hash_is_frozen():
    assert (
        sha256_file(REPO / "config/hong2021_v28_failure_mechanism_audit.json")
        == AUDIT_PROGRAM_SHA256
    )


def test_streaming_tail_summary_matches_numpy_definition():
    rng = np.random.default_rng(29)
    y = rng.normal(size=1_100_000).astype(np.float32) / 20.0
    accumulator = PhysicalTailAccumulator()
    for row in np.array_split(y, 11):
        accumulator.update(row)
    report = accumulator.report()
    log10rho = 4.5 * y.astype(np.float64)
    delta = np.power(10.0, log10rho) - 1.0
    assert np.isclose(report["q99_999_log10rho"], np.quantile(log10rho, 0.99999))
    assert report["maximum_log10rho"] == log10rho.max()
    assert np.isclose(report["mean_delta_squared"], np.square(delta).mean())


def test_physical_tail_comparison_uses_unchanged_thresholds():
    reference = {
        "q99_999_log10rho": 3.0,
        "maximum_log10rho": 4.0,
        "mean_delta_squared": 100.0,
    }
    passing = {
        "q99_999_log10rho": 3.09,
        "maximum_log10rho": 4.29,
        "mean_delta_squared": 149.0,
    }
    failing = {**passing, "mean_delta_squared": 151.0}
    assert compare_physical_tails(reference, passing)["pass"] is True
    assert compare_physical_tails(reference, failing)["pass"] is False


def _domain_row():
    passed = {"pass": True}
    return {
        "self_reconstruction_vs_truth": copy.deepcopy(passed),
        "cross_generated_vs_selected_donor_truths": copy.deepcopy(passed),
        "selected_donor_truths_vs_validation_truth": copy.deepcopy(passed),
        "two_point_improves_deterministic_all_scales": True,
    }


def test_failure_classification_order_is_frozen():
    rows = {name: _domain_row() for name in ("tng", "simba_dev", "swift_dev")}
    rows["tng"]["selected_donor_truths_vs_validation_truth"]["pass"] = False
    rows["tng"]["cross_generated_vs_selected_donor_truths"]["pass"] = False
    rows["tng"]["self_reconstruction_vs_truth"]["pass"] = False
    assert classify(rows)["class"].startswith("V21_V14_representation")
    rows["tng"]["self_reconstruction_vs_truth"]["pass"] = True
    assert classify(rows)["class"].startswith("cross_condition_inverse")
    rows["tng"]["cross_generated_vs_selected_donor_truths"]["pass"] = True
    assert classify(rows)["class"].startswith("target_free_condition_matching")
    rows["tng"]["selected_donor_truths_vs_validation_truth"]["pass"] = True
    rows["tng"]["two_point_improves_deterministic_all_scales"] = False
    assert classify(rows)["class"].endswith("limits_2PCF")
