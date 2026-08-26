from __future__ import annotations

import hashlib
import copy
import json
import math
from pathlib import Path
import stat
import sys

import numpy as np
import pytest

import cf4_lg_highk_terminal_aggregation as terminal


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_cf4_lg_highk_terminal_aggregation_v1 as independent  # noqa: E402
CONFIG = ROOT / "config/cf4_lg_highk_terminal_aggregation_v1.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text())


def _rows(*, repeated_parent: bool = False) -> list[dict]:
    return [{
        "schedule_index": index,
        "parent_seed": 1 if repeated_parent else index,
        "bridge_group": index // 64,
        "geometry_key": [index, 0, 0, 0, 0, 0],
        "fine_field_seed": 1000 + index,
        "posterior_weight": 1.0 / 256.0,
        "hard_pair_ids": [[0, 1]] if index < 16 else [],
        "loose_pair_ids": [[0, 1], [2, 3]] if index < 16 else [],
        "intersection_pair_ids": [[0, 1]] if index < 16 else [],
    } for index in range(256)]


def _pair(index: int, score: float = 0.0, passed: bool = True) -> dict:
    return {
        "schedule_index": index, "parent_seed": index,
        "bridge_group": index // 64, "geometry_key": [index, 0, 0, 0, 0, 0],
        "fine_field_seed": 1000 + index, "halo_i": 0, "halo_j": 1,
        "midpoint_mpc_h": [192.0, 192.0, 192.0],
        "log_likelihood": score, "p1_pass": passed,
    }


def test_full_loose_denominator_and_equal_row_prior() -> None:
    rows = _rows()
    pairs = [_pair(0, 0.0), _pair(1, math.log(2.0))]
    result = terminal.aggregate_terminal(rows, pairs, _config())
    first, second = result["rows"][:2]
    assert first["row_log_evidence"] == {"finite": True, "value": -math.log(2.0)}
    assert second["row_log_evidence"] == {"finite": True, "value": 0.0}
    assert math.isclose(second["normalized_weight"], 2.0 / 3.0)
    assert math.isclose(first["normalized_weight"], 1.0 / 3.0)
    assert math.isclose(
        second["unnormalized_row_log_weight"]["value"]
        - second["row_log_evidence"]["value"], -math.log(256.0),
    )


def test_zero_eligible_rows_are_closed_and_json_finite() -> None:
    result = terminal.aggregate_terminal(_rows(), [], _config())
    assert result["status"] == "complete_scientific_fail_terminal_aggregation_closed"
    assert result["jointly_eligible_rows"] == 0
    assert result["normalized_row_weight_ESS"] == 0.0
    assert all(row["normalized_weight"] == 0.0 for row in result["rows"])
    assert all(row["row_log_evidence"] == {"finite": False, "value": None}
               for row in result["rows"])
    json.dumps(result, allow_nan=False)


def test_repeated_rows_are_not_deduplicated_and_parent_gate_is_grouped() -> None:
    rows = _rows(repeated_parent=True)
    pairs = [_pair(index) for index in range(8)]
    for pair in pairs:
        pair["parent_seed"] = 1
    result = terminal.aggregate_terminal(rows, pairs, _config())
    assert result["jointly_eligible_rows"] == 8
    assert math.isclose(result["normalized_row_weight_ESS"], 8.0)
    assert result["grouped_support"]["parent"] == {"ESS": 1.0, "maximum_weight": 1.0}
    assert result["checks"]["minimum_parent_weight_ESS"] is False


def test_bridge_and_geometry_support_gates_are_independent() -> None:
    rows = _rows()
    pairs = [_pair(index) for index in range(8)]
    result = terminal.aggregate_terminal(rows, pairs, _config())
    assert result["checks"]["minimum_geometry_key_weight_ESS"] is True
    assert result["checks"]["minimum_bridge_group_weight_ESS"] is False
    assert result["grouped_support"]["bridge"]["maximum_weight"] == 1.0


def test_pair_mismatch_fails_before_P1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    row = tmp_path / "row_000"
    row.mkdir()
    np.savez(row / "halos.npz", halo_pos=np.zeros((2, 3)), halo_vel=np.zeros((2, 3)), halo_mass=np.ones(2))
    (row / "result.json").write_text(json.dumps({
        "hard_p2_pairs": [],
        "z0_likelihood": {"n_candidate_pairs": 0, "log_likelihood": -math.inf,
                          "best_pair": None, "candidate_pairs": []},
        "parent_seed": 1, "group_id": 0, "geometry_key": [0] * 6,
        "fine_field_seed": 2, "posterior_weight": 1 / 256,
    }))
    hard_pair = {"halo_i": 0, "halo_j": 1, "ranking_score": 0.0}
    monkeypatch.setattr(terminal, "find_pairs", lambda *args, **kwargs: [dict(hard_pair)])
    monkeypatch.setattr(terminal, "rank_score", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(terminal, "score_catalog", lambda *args, **kwargs: {
        "n_candidate_pairs": 0, "log_likelihood": -math.inf,
        "best_pair": None, "candidate_pairs": [],
    })
    with pytest.raises(RuntimeError, match="hard"):
        terminal._recompute_catalogues(
            production_root=tmp_path,
            hard_config={"screen": {}, "m33_subpeak_gate": {}, "ranking": {}},
            likelihood_program={}, box_size=384.0,
        )


def test_exact_all_five_P1_gates_are_required(monkeypatch: pytest.MonkeyPatch) -> None:
    science = {
        "clusters": {}, "secondary_cluster_anchors": {}, "local_void": {},
        "bootes_void": {}, "observer_environment": {}, "n_gates_passed": 4,
        "pass": True,
    }
    calls = 0

    def fake_score(*args, **kwargs):
        nonlocal calls
        calls += 1
        gates = {name: True for name in terminal.FIVE_P1_GATES}
        if calls == 2:
            gates.pop("ObserverEnvironment")
        return {**science, "gates": gates}

    monkeypatch.setattr(terminal, "_load_parent_field", lambda *args: (np.zeros((2, 2, 2)), 2.0, {"Om": 0.3}))
    monkeypatch.setattr(terminal, "_make_parent_forward", lambda *args: object())
    monkeypatch.setattr(terminal, "_forward_parent_density", lambda *args, **kwargs: (np.zeros((2, 2, 2)), 2.0, 0.3))
    monkeypatch.setattr(terminal, "score_member", fake_score)
    zero = {**science, "gates": {name: True for name in terminal.FIVE_P1_GATES}}
    with pytest.raises(RuntimeError, match="exact five"):
        terminal.evaluate_pair_recentered_p1(
            intersections=[_pair(0)],
            parent_entries={0: {"seed": 0}}, p1_config={}, legacy_rows={0: zero},
            box_size=384.0,
        )


def test_frozen_config_forbids_v8_correction_and_downstream() -> None:
    config = terminal.load_terminal_config(CONFIG)
    assert set(config["reuse_policy"]["v8_allowed_sections"]) == {
        "candidate_preselection", "z0_likelihood",
    }
    assert config["reuse_policy"]["v8_proposal_importance_correction_allowed"] is False
    assert config["authorization"]["automatic_promotion"] is False
    assert config["authorization"]["RAMSES"] is False
    assert "parent-centered P1 prefilter" in config["forbidden"]


def test_pinned_input_validation_is_read_only(tmp_path: Path) -> None:
    path = tmp_path / "input.bin"
    path.write_bytes(b"immutable-input")
    before = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert terminal._pinned({"path": str(path), "sha256": digest}, "synthetic") == path
    after = path.stat()
    assert path.read_bytes() == b"immutable-input"
    assert (before.st_size, before.st_mtime_ns, before.st_mode) == (
        after.st_size, after.st_mtime_ns, after.st_mode,
    )


def test_input_mutation_between_evaluation_and_seal_is_rejected(tmp_path: Path) -> None:
    row = tmp_path / "row_000"
    row.mkdir()
    result, halos, parent = row / "result.json", row / "halos.npz", tmp_path / "parent.npz"
    result.write_bytes(b"result")
    halos.write_bytes(b"halos")
    parent.write_bytes(b"parent")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "rows": [{"schedule_index": 0, "result_sha256": digest(result),
                  "halo_catalogue_sha256": digest(halos)}],
        "parent_fields": [{"parent_seed": 1, "path": str(parent),
                           "sha256": digest(parent)}],
    }
    terminal.verify_manifest_inputs_unchanged(manifest, tmp_path)
    halos.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="changed during aggregation"):
        terminal.verify_manifest_inputs_unchanged(manifest, tmp_path)


def test_anchor_rejects_count_or_identity_drift() -> None:
    rows = _rows()
    config = _config()
    with pytest.raises(RuntimeError, match="anchors changed"):
        terminal._assert_observed_anchors(rows, [_pair(index) for index in range(18)], config)


def test_make_forward_is_created_once_for_twelve_parents(monkeypatch: pytest.MonkeyPatch) -> None:
    intersections = []
    for index in range(12):
        pair = _pair(index)
        pair["parent_seed"] = index
        intersections.append(pair)
    entries = {index: {"seed": index} for index in range(12)}
    gates = {name: True for name in terminal.FIVE_P1_GATES}
    metrics = {
        "clusters": {}, "secondary_cluster_anchors": {}, "local_void": {},
        "bootes_void": {}, "observer_environment": {}, "gates": gates,
        "n_gates_passed": 5, "pass": True,
    }
    legacy = {index: metrics for index in range(12)}
    make_calls = []
    monkeypatch.setattr(terminal, "_load_parent_field", lambda *args: (
        np.zeros((2, 2, 2)), 2.0, {"Om": 0.3},
    ))
    monkeypatch.setattr(terminal, "_make_parent_forward", lambda cosmology: make_calls.append(dict(cosmology)) or object())
    monkeypatch.setattr(terminal, "_forward_parent_density", lambda *args, **kwargs: (
        np.zeros((2, 2, 2)), 2.0, 0.3,
    ))
    monkeypatch.setattr(terminal, "score_member", lambda *args, **kwargs: metrics)
    output = terminal.evaluate_pair_recentered_p1(
        intersections=intersections, parent_entries=entries, p1_config={},
        legacy_rows=legacy, box_size=384.0,
    )
    assert len(output) == 12
    assert make_calls == [{"Om": 0.3}]


def test_finite_log_weight_defines_eligibility_even_after_underflow() -> None:
    rows = _rows()
    pairs = [_pair(0, 0.0), _pair(1, -1000.0)]
    result = terminal.aggregate_terminal(rows, pairs, _config())
    assert result["jointly_eligible_rows"] == 2
    assert result["rows"][1]["normalized_weight"] == 0.0


def test_independent_checker_rejects_correlated_producer_forgery(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = _rows()
    pairs = [_pair(index) for index in range(8)]
    honest = independent.independent_aggregate(rows, pairs, _config())
    forged = copy.deepcopy(honest)
    forged["rows"][0]["row_log_evidence"]["value"] = 0.0
    monkeypatch.setattr(terminal, "aggregate_terminal", lambda *args, **kwargs: forged)
    independently_recomputed = independent.independent_aggregate(rows, pairs, _config())
    with pytest.raises(RuntimeError, match="values differ"):
        independent._equal(forged, independently_recomputed, "forged")


def test_independent_checker_matches_honest_producer() -> None:
    rows = _rows()
    pairs = [_pair(index, -0.25 * index) for index in range(12)]
    produced = terminal.aggregate_terminal(rows, pairs, _config())
    checked = independent.independent_aggregate(rows, pairs, _config())
    independent._equal(produced, checked, "honest")


def _fake_payloads() -> dict[str, dict]:
    return {
        "input_manifest.json": {"schema": terminal.INPUT_SCHEMA},
        "pair_recentered_p1.json": {"schema": terminal.P1_SCHEMA},
        "terminal_result.json": {"schema": terminal.RESULT_SCHEMA},
    }


def test_durable_seal_and_no_overwrite_publish(tmp_path: Path) -> None:
    staging = tmp_path / ".staging"
    staging.mkdir(mode=0o700)
    terminal._seal(staging, CONFIG, _fake_payloads(), {"test": True})
    assert stat.S_IMODE(staging.stat().st_mode) == 0o555
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o444 for path in staging.iterdir())
    final = tmp_path / "final"
    terminal._publish_no_replace(staging, final)
    assert final.is_dir() and not staging.exists()

    second = tmp_path / ".second"
    second.mkdir(mode=0o700)
    terminal._seal(second, CONFIG, _fake_payloads(), {"test": True})
    with pytest.raises(FileExistsError, match="replace"):
        terminal._publish_no_replace(second, final)
    assert second.exists()


def test_partial_staging_never_becomes_canonical(tmp_path: Path) -> None:
    partial = tmp_path / ".partial"
    partial.mkdir(mode=0o555)
    final = tmp_path / "final"
    with pytest.raises(RuntimeError, match="partial or unsealed"):
        terminal._publish_no_replace(partial, final)
    assert partial.exists() and not final.exists()


def test_measured_catalogue_bounds_and_likelihood_are_narrow() -> None:
    assert not hasattr(terminal, "_values_close")
    terminal._serialization_close("separation_mpc_h", 1.0, 1.0 + 3e-5, "pair.sep")
    with pytest.raises(RuntimeError, match="ULP bound"):
        terminal._serialization_close("separation_mpc_h", 1.0, 1.001, "pair.sep")
    assert terminal.LIKELIHOOD_ATOL <= 1e-8


def test_checker_rejects_correlated_reseal_with_changed_p1_likelihood(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    rows = _rows()
    intersections = []
    pairs = []
    gates = {name: True for name in terminal.FIVE_P1_GATES}
    for index in range(18):
        rows[index]["hard_pair_ids"] = [[0, 1]]
        rows[index]["loose_pair_ids"] = [[0, 1]]
        rows[index]["intersection_pair_ids"] = [[0, 1]]
        manifest_pair = {
            "schedule_index": index, "parent_seed": index % 12,
            "bridge_group": index // 64,
            "geometry_key": rows[index]["geometry_key"],
            "fine_field_seed": rows[index]["fine_field_seed"],
            "halo_i": 0, "halo_j": 1,
            "midpoint_mpc_h": [192.0, 192.0, 192.0],
            "log_likelihood": 0.0,
        }
        intersections.append(manifest_pair)
        pairs.append({
            **manifest_pair, "observer_offset_mpc_h": [0.0, 0.0, 0.0],
            "p1_gates": gates, "p1_pass": True,
            "p1_metrics": {"gates": gates, "pass": True},
        })
    # Correlated forgery: change P1 likelihood, then recompute the terminal
    # result and every outer seal so only the independent manifest binding can
    # detect the substitution.
    pairs[0]["log_likelihood"] = 1.0
    input_value = {
        "schema": terminal.INPUT_SCHEMA, "rows": rows,
        "hard_loose_same_identity_pairs": intersections, "parent_fields": [],
    }
    p1_value = {
        "schema": terminal.P1_SCHEMA, "pair_count": 18,
        "unique_parent_count": 12,
        "exact_gate_names": list(terminal.FIVE_P1_GATES), "pairs": pairs,
    }
    result_value = independent.independent_aggregate(rows, pairs, config)
    result_value.update({
        "config_sha256": independent.sha256_file(CONFIG),
        "seconds": 1.0,
    })
    output = tmp_path / "sealed"
    output.mkdir(mode=0o700)

    def write(name: str, value: dict) -> None:
        (output / name).write_bytes(independent.canonical_bytes(value))
        (output / name).chmod(0o444)

    write("input_manifest.json", input_value)
    write("pair_recentered_p1.json", p1_value)
    result_value["input_manifest_sha256"] = independent.sha256_file(output / "input_manifest.json")
    result_value["pair_recentered_p1_sha256"] = independent.sha256_file(output / "pair_recentered_p1.json")
    write("terminal_result.json", result_value)
    schemas = {
        "input_manifest.json": terminal.INPUT_SCHEMA,
        "pair_recentered_p1.json": terminal.P1_SCHEMA,
        "terminal_result.json": terminal.RESULT_SCHEMA,
    }
    manifest = {
        "schema": terminal.MANIFEST_SCHEMA, "status": "sealed",
        "config": str(CONFIG.resolve()),
        "config_sha256": independent.sha256_file(CONFIG),
        "runtime": {"correlated": "resealed"},
        "files": [{
            "name": name, "sha256": independent.sha256_file(output / name),
            "size_bytes": (output / name).stat().st_size, "schema": schemas[name],
        } for name in sorted(schemas)],
    }
    write("manifest.json", manifest)
    write("COMPLETE", {
        "schema": terminal.COMPLETE_SCHEMA, "status": "complete",
        "manifest_sha256": independent.sha256_file(output / "manifest.json"),
    })
    output.chmod(0o555)
    monkeypatch.setattr(independent, "_verify_runtime", lambda *args: None)
    monkeypatch.setattr(independent, "_verify_inputs", lambda *args: None)
    monkeypatch.setattr(independent, "_verify_anchors", lambda *args: None)
    with pytest.raises(RuntimeError, match="log_likelihood"):
        independent.check_output(CONFIG, output)
