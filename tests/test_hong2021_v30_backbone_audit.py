import hashlib
from pathlib import Path

import numpy as np

from hong2021_v30_backbone_audit import (
    PROGRAM_SHA256,
    StreamingPearson,
    block_mean,
    block_sum,
    classify,
    fourier_masks,
    tail_diagnostics,
)


REPO = Path(__file__).resolve().parents[1]


def test_v30_frozen_program_hash_and_firewall():
    path = REPO / "config/hong2021_v30_backbone_condition_audit.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PROGRAM_SHA256
    text = path.read_text().lower()
    assert '"astrid_access": "forbidden"' in text
    assert '"historical_eagle_access": "forbidden"' in text
    assert '"posthoc_ak": false' in text


def test_streaming_pearson_and_blocks():
    value = np.arange(64, dtype=np.float64).reshape(4, 4, 4)
    assert np.array_equal(block_sum(value, 2), value.reshape(2, 2, 2, 2, 2, 2).sum((1, 3, 5)))
    assert np.allclose(block_mean(value, 2), block_sum(value, 2) / 8.0)
    statistic = StreamingPearson()
    statistic.add(value, 3.0 * value + 7.0)
    result = statistic.result()
    assert result["n"] == 64
    assert np.isclose(result["pearson"], 1.0)


def test_fourier_masks_exhaust_non_dc_modes():
    edges = np.asarray([0.0, 0.5, 1.0, 2.0, 4.0, np.inf])
    masks = fourier_masks(16, 0.3125, edges)
    total = sum(mask.astype(np.int8) for mask in masks)
    assert int(total.sum()) == 16**3 - 1
    assert int(total.max()) == 1


def test_tail_diagnostic_identity_and_extreme_inflation():
    truth = np.linspace(-0.4, 0.4, 100_000, dtype=np.float64)
    exact = tail_diagnostics(truth, truth.copy())
    assert abs(exact["delta_q99_999_dex"]) < 1.0e-12
    assert abs(exact["generated_max_above_truth_max_dex"]) < 1.0e-12
    assert np.isclose(exact["generated_over_truth_mean_delta_squared"], 1.0)
    assert exact["Q3_pass"] and exact["Q4_pass"]
    inflated = truth.copy()
    inflated[-10:] += 1.0
    failed = tail_diagnostics(truth, inflated)
    assert not failed["Q3_pass"]
    assert not failed["Q4_pass"]


def test_all_v30_classification_branches_are_fixed():
    assert classify(True, False)[0] == "deterministic_backbone_and_local_residual_separation_adequate"
    assert classify(False, True)[0] == "deterministic_backbone_underfit_and_local_residual_coupling_unmodeled"
    assert classify(False, False)[0] == "deterministic_current_density_backbone_underfit"
    assert classify(True, True)[0] == "global_backbone_improves_but_local_residual_coupling_is_unmodeled"
