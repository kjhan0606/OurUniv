from pathlib import Path

import numpy as np
import pytest

import hong2021_v80dr2_report_only as report


def _domain_row(domain: str) -> dict:
    return {
        "candidate_ensemble": Path(f"/{domain}/candidate.h5"),
        "candidate_ensemble_sha256": domain * 64,
        "control_ensemble": Path(f"/{domain}/control.h5"),
        "control_ensemble_sha256": "c" * 64,
        "candidate_metrics_path": Path(f"/{domain}/candidate_metrics.json"),
        "candidate_metrics_sha256": "m" * 64,
        "control_metrics_path": Path(f"/{domain}/control_metrics.json"),
        "control_metrics_sha256": "n" * 64,
        "candidate_metrics": {"arm": "candidate"},
        "control_metrics": {"arm": "control"},
        "numerical_and_pairing": {"residual_DC_pass": True},
    }


def test_domain_row_matches_exact_frozen_helper_contract(monkeypatch) -> None:
    monkeypatch.setattr(report.v79, "validate_ensemble_pair", lambda *args: {"residual_DC_pass": True})
    monkeypatch.setattr(
        report.v79,
        "load_metrics",
        lambda path, ensemble, indices: {"path": str(path), "indices": indices},
    )
    row = report.build_domain_row(
        Path("/candidate.h5"),
        "a" * 64,
        Path("/control.h5"),
        "b" * 64,
        Path("/candidate_metrics.json"),
        "c" * 64,
        Path("/control_metrics.json"),
        "d" * 64,
        [1, 2, 3],
        {
            "candidate_expected_attrs": {},
            "control_expected_attrs": {},
            "pairing": {},
        },
    )
    assert set(row) == report.DOMAIN_ROW_KEYS
    assert row["candidate_ensemble_sha256"] == "a" * 64
    assert row["control_ensemble_sha256"] == "b" * 64


def test_formula_contract_reaches_both_helpers_and_global_combiner(monkeypatch) -> None:
    domains = {domain: _domain_row(domain) for domain in report.DOMAIN_ORDER}
    seen = {}

    def physical(rows, references):
        seen["physical"] = rows
        return 0.4, {"block_p": 0.4}

    def rank(rows):
        seen["rank_sha"] = [
            rows[domain]["candidate_ensemble_sha256"]
            for domain in report.DOMAIN_ORDER
        ]
        return 0.3, {"block_p": 0.3}

    monkeypatch.setattr(report.v79, "physical_energy_observation", physical)
    monkeypatch.setattr(report.v79, "rank_coverage_observation", rank)
    monkeypatch.setattr(
        report.v79.v78,
        "global_p_value",
        lambda physical_p, rank_p: np.asarray([0.2]),
    )
    output = report.formula_diagnostic(domains, {})
    assert seen["physical"] is domains
    assert len(seen["rank_sha"]) == 3
    assert output["formula_global_p"] == 0.2
    assert output["would_pass_formula_if_prospective"] is True
    assert output["may_be_reported_as_a_V79_pass"] is False


def test_formula_contract_refuses_the_exact_prior_missing_key() -> None:
    domains = {domain: _domain_row(domain) for domain in report.DOMAIN_ORDER}
    del domains["TNG100"]["candidate_ensemble_sha256"]
    with pytest.raises(ValueError, match="exact V79 helper contract"):
        report.formula_diagnostic(domains, {})


def test_report_source_has_no_mutating_or_reexecution_imports() -> None:
    source = Path(report.__file__).read_text()
    assert "hong2021_v80dr_metadata_recovery" not in source
    assert "hong2021_v80_evaluate" not in source
    assert "hong2021_v80_manifest" not in source
    assert "hong2021_v80_sample as base_sample" in source
    assert "hong2021_v79_complete_gate as v79" in source
