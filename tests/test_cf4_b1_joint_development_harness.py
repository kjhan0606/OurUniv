from __future__ import annotations

import numpy as np
import pytest

from cf4_b1_joint_development_harness import (
    run_joint_harness,
    source_bound_joint_factor_probe,
    spherical_rsd_fog_probe,
    tsc_adjoint_probe,
    validate_selection_support,
)


def test_joint_harness_passes_all_declared_local_probes():
    result = run_joint_harness()
    assert result["status"] == "PASS"
    assert all(probe.get("status") in ("PASS", None) for probe in result["probes"].values())
    assert result["probes"]["source_bound_joint_factor"]["independent_redshift_rejected"] is True
    assert result["validation_seeds_opened"] is False


def test_selection_support_rejects_positive_count_outside_support():
    counts = np.zeros((6, 2, 2, 2), dtype=np.int64)
    exposure = np.ones_like(counts, dtype=np.float64)
    counts[0, 0, 0, 0] = 1
    exposure[0, 0, 0, 0] = 0.0
    with pytest.raises(ValueError, match="zero selection exposure"):
        validate_selection_support(counts, exposure)


def test_individual_joint_probes_are_finite_and_conservative():
    assert tsc_adjoint_probe()["inner_product_abs_error"] <= 1.0e-12
    assert spherical_rsd_fog_probe()["mass_relative_error_max"] <= 1.0e-12
    assert source_bound_joint_factor_probe()["joint_log_likelihood_finite"] is True
