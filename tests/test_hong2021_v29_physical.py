from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hong2021_residual_evaluate import CENTERED_SCHEMAS
from hong2021_v18_init import sha256_file
from hong2021_v29_physical import (
    DESIGN_AUDIT_SHA256,
    ENSEMBLE_SCHEMA,
    REGISTRY_SHA256,
    centered_donor_residual,
    transport_residual,
)


REPO = Path(__file__).parents[1]


def test_v29_registry_hash_single_change_and_firewall():
    path = REPO / "config/hong2021_v29_development_program.json"
    assert sha256_file(path) == REGISTRY_SHA256
    registry = json.loads(path.read_text())
    assert registry["design_audit"]["sha256"] == DESIGN_AUDIT_SHA256
    assert registry["approval_firewall"]["Astrid_attempts_remaining"] == 1
    assert ENSEMBLE_SCHEMA in CENTERED_SCHEMAS


def test_centered_physical_residual_has_zero_dc_and_exact_definition():
    rng = np.random.default_rng(31)
    target = rng.normal(size=(1, 8, 8, 8)).astype(np.float32)
    mean = rng.normal(size=(1, 8, 8, 8)).astype(np.float32)
    residual = centered_donor_residual(target, mean, 0.2)
    raw = target - (mean + np.float32(0.2))
    expected = raw.astype(np.float64) - raw.astype(np.float64).mean()
    assert np.max(np.abs(residual.mean(axis=(-3, -2, -1)))) < 1.0e-8
    assert np.allclose(residual, expected.astype(np.float32))


def test_transport_preserves_residual_values_and_query_baseline():
    residual = np.arange(4**3, dtype=np.float32).reshape(1, 4, 4, 4)
    residual -= residual.mean()
    baseline = np.full((1, 4, 4, 4), 3.0, dtype=np.float32)
    sample = transport_residual(residual, baseline, 17)
    assert np.allclose(np.sort((sample - baseline).ravel()), np.sort(residual.ravel()))
    assert np.max(np.abs((sample - baseline).mean())) < 1.0e-7
