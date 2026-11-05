import hashlib
from pathlib import Path

import numpy as np

from hong2021_residual_evaluate import CENTERED_SCHEMAS
from hong2021_v28_empirical import DOMAIN_ORDER
from hong2021_v31_copula import (
    BACKBONE_BINS,
    REGISTRY_SHA256,
    conditional_forward,
    conditional_inverse,
    equal_source_weighted_quantile,
    lattice_slices,
    quantile_levels,
    strict_monotonic,
    transport_conditional_residual,
)


REPO = Path(__file__).resolve().parents[1]


def test_v31_registry_hash_and_firewall():
    path = REPO / "config/hong2021_v31_development_program.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == REGISTRY_SHA256
    text = path.read_text().lower()
    assert '"posthoc_ak": false' in text
    assert '"astrid_access": "forbidden' in text
    assert '"historical_eagle_access": "forbidden"' in text


def test_equal_source_quantile_is_not_object_count_weighted():
    rows = {
        DOMAIN_ORDER[0]: np.zeros(10),
        DOMAIN_ORDER[1]: np.full(1000, 10.0),
        DOMAIN_ORDER[2]: np.full(10, 20.0),
    }
    values = equal_source_weighted_quantile(rows, np.asarray([0.1, 0.5, 0.9]))
    assert values[0] == 0.0
    assert values[1] == 10.0
    assert values[2] == 20.0


def test_quantile_levels_and_monotonic_repair():
    levels = quantile_levels()
    assert levels[0] == 0.0 and levels[-1] == 1.0
    assert np.any(np.isclose(levels, 1.0e-7))
    repaired = strict_monotonic(np.asarray([-2.0, 0.0, 0.0, 2.0]), -2.0, 2.0)
    assert np.all(np.diff(repaired) > 0)
    assert repaired[0] == -2.0 and repaired[-1] == 2.0


def test_lattice_has_exact_32_cubed_voxels_and_cycles_offsets():
    cube = np.zeros((64, 64, 64))
    selections = [lattice_slices(index) for index in range(8)]
    assert all(cube[value].size == 32**3 for value in selections)
    offsets = {tuple(axis.start for axis in value) for value in selections}
    assert len(offsets) == 8


def _toy_model():
    levels = quantile_levels()
    edges = np.linspace(-4.0, 4.0, BACKBONE_BINS + 1)
    table = np.stack([levels + 0.01 * index for index in range(BACKBONE_BINS)])
    return {
        "backbone_edges": edges,
        "quantile_levels": levels,
        "residual_quantiles": table,
    }


def test_conditional_forward_inverse_and_transport_dc():
    model = _toy_model()
    rng = np.random.default_rng(31)
    backbone = rng.uniform(-3.9, 3.9, size=(1, 8, 8, 8))
    bins = np.searchsorted(model["backbone_edges"][1:-1], backbone, side="right")
    residual = rng.uniform(0.05, 0.95, size=backbone.shape) + 0.01 * bins
    uniform = conditional_forward(residual, backbone, model)
    decoded = conditional_inverse(uniform, backbone, model)
    assert np.max(np.abs(decoded - residual)) < 2.0e-7
    truth = backbone + residual
    sample, dc = transport_conditional_residual(truth, backbone, backbone, 0, model)
    assert np.isfinite(sample).all()
    assert dc < 1.0e-12
    assert abs(float((sample - backbone).mean())) < 2.0e-7


def test_v31_schema_is_evaluable_as_centered_residual():
    assert "hong2021-v31-physical-conditional-copula-ensemble-v1" in CENTERED_SCHEMAS
