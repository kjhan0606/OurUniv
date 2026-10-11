from __future__ import annotations

import copy
import inspect
import json
import os
from pathlib import Path
import shutil
import stat

import pytest

import cf4_lg_highk_terminal_publication_repair as repair


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/cf4_lg_highk_terminal_publication_repair_program_v1.json"
SEALED = Path("/gpfs/kjhan/CF4/recon/linear_cr/.lg_highk_terminal_aggregation_v1.305221.a51b6c9282d34950a3ca52287c115c99.staging")


def _config() -> dict:
    return repair.load_config(CONFIG)


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "sealed.staging"
    source.mkdir()
    for name in repair.ARTIFACT_NAMES:
        shutil.copyfile(SEALED / name, source / name)
        (source / name).chmod(0o444)
    source.chmod(0o555)
    return source


def _publish(tmp_path: Path, *, checker_hook=repair._run_private_checker):
    source, target = _source(tmp_path), tmp_path / "canonical"
    result = repair._publish_fixture(
        authority=repair._FIXTURE_TOKEN, source=source, target=target,
        checker_hook=checker_hook,
    )
    assert result["scientific_status"] == "complete_scientific_fail_terminal_aggregation_closed"
    return source, target


def test_config_is_exact_canonical_and_reserves_absent_grant_and_result() -> None:
    config = _config()
    assert len(config) == 15 and set(config) == repair.CONFIG_KEYS
    assert config["resources"]["node"] == "grammar-debug"
    for field in ("grant_path", "future_result_record_path"):
        path = ROOT / config["execution"][field]
        assert not path.exists() and not path.is_symlink()


def test_arbitrary_config_path_and_public_overrides_are_impossible(tmp_path: Path) -> None:
    copied = tmp_path / "program.json"
    copied.write_bytes(CONFIG.read_bytes())
    with pytest.raises(RuntimeError, match="canonical"):
        repair.load_config(copied)
    assert list(inspect.signature(repair.publish).parameters) == ["config_path"]
    assert "source_override" not in inspect.getsource(repair.publish)
    assert "target_override" not in inspect.getsource(repair.publish)


def _lineage_snapshot() -> dict:
    config = _config()
    implementation = "1" * 40
    implementation_paths = config["lineage"]["required_exact_added_paths"]
    grant_paths = config["lineage"]["grant_commit"]["required_exact_added_paths"]
    return {
        "config": config, "head": "2" * 40, "upstream": "2" * 40,
        "grant_parents": [implementation], "implementation_parents": [repair.EXPECTED_PARENT],
        "implementation_rows": [("A", path) for path in implementation_paths],
        "grant_rows": [("A", path) for path in grant_paths],
        "implementation_modes": {path: "100644" for path in implementation_paths},
        "grant_modes": {path: "100644" for path in grant_paths}, "clean": True,
    }


def test_lineage_requires_parent_exact_six_then_exact_one_grant() -> None:
    snapshot = _lineage_snapshot()
    assert repair.validate_lineage_values(**snapshot) == "1" * 40
    mutations = []
    wrong_parent = copy.deepcopy(snapshot)
    wrong_parent["implementation_parents"] = ["0" * 40]
    mutations.append(wrong_parent)
    extra_impl = copy.deepcopy(snapshot)
    extra_impl["implementation_rows"].append(("A", "README.md"))
    mutations.append(extra_impl)
    extra_grant = copy.deepcopy(snapshot)
    extra_grant["grant_rows"].append(("A", "README.md"))
    mutations.append(extra_grant)
    dirty = copy.deepcopy(snapshot)
    dirty["clean"] = False
    mutations.append(dirty)
    for changed in mutations:
        with pytest.raises(RuntimeError):
            repair.validate_lineage_values(**changed)


@pytest.mark.parametrize("field", ["implementation_rows", "grant_rows"])
@pytest.mark.parametrize("status_code", ["M", "D", "R"])
def test_lineage_rejects_non_additive_rows(field: str, status_code: str) -> None:
    snapshot = _lineage_snapshot()
    snapshot[field][0] = (status_code, snapshot[field][0][1])
    with pytest.raises(RuntimeError, match="exact"):
        repair.validate_lineage_values(**snapshot)


def _grant(config: dict, implementation: str, rows: list[dict]) -> dict:
    return {
        "schema": repair.GRANT_SCHEMA, "status": "authorized_publication_only",
        "date": "2026-08-27", "purpose": "authorize byte-only publication",
        "lineage": {"required_parent_commit": repair.EXPECTED_PARENT},
        "program": {"path": str(repair.CANONICAL_CONFIG.relative_to(ROOT)),
                    "sha256": repair.EXPECTED_CONFIG_SHA},
        "implementation": {"commit": implementation, "files": rows},
        "source_staging": str(repair.EXPECTED_SOURCE),
        "source_artifacts": [{key: row[key] for key in ("name", "size_bytes", "sha256")}
                             for row in config["source_artifacts"]],
        "canonical_target": str(repair.EXPECTED_TARGET),
        "authorization": {"publication_only": True, "scientific_recomputation": False,
                          "overwrite": False, "retry": False, "cleanup": False,
                          "promotion": False},
    }


def test_grant_binds_program_implementation_files_source_target_and_flags() -> None:
    config, implementation = _config(), "1" * 40
    rows = [{"path": path, "mode": "100644", "sha256": "a" * 64}
            for path in config["lineage"]["required_exact_added_paths"]]
    grant = _grant(config, implementation, rows)
    repair.validate_grant_values(config, {"implementation": implementation}, grant, rows)
    for mutation in (
        lambda value: value["program"].update(sha256="0" * 64),
        lambda value: value["implementation"].update(commit="0" * 40),
        lambda value: value.update(source_staging="/wrong"),
        lambda value: value.update(canonical_target="/wrong"),
        lambda value: value["authorization"].update(retry=True),
    ):
        changed = copy.deepcopy(grant)
        mutation(changed)
        with pytest.raises(PermissionError, match="bindings"):
            repair.validate_grant_values(config, {"implementation": implementation}, changed, rows)


def test_fixture_token_and_canonical_paths_are_fail_closed(tmp_path: Path) -> None:
    source = _source(tmp_path)
    with pytest.raises(PermissionError, match="authority"):
        repair._publish_fixture(authority=object(), source=source, target=tmp_path / "bad")
    with pytest.raises(RuntimeError, match="canonical"):
        repair._publish_fixture(
            authority=repair._FIXTURE_TOKEN, source=repair.EXPECTED_SOURCE,
            target=tmp_path / "bad",
        )
    with pytest.raises(RuntimeError, match="canonical"):
        repair._publish_fixture(
            authority=repair._FIXTURE_TOKEN, source=source, target=repair.EXPECTED_TARGET,
        )


def test_private_checker_runs_while_mode_0700_then_target_becomes_0555(tmp_path: Path) -> None:
    observed = []

    def checker_hook(config_path: Path, source_fd: int, parent_fd: int,
                     target_fd: int, target_name: str) -> None:
        observed.append(stat.S_IMODE(os.fstat(target_fd).st_mode))
        repair._run_private_checker(config_path, source_fd, parent_fd, target_fd, target_name)

    source, target = _publish(tmp_path, checker_hook=checker_hook)
    assert observed == [0o700]
    assert stat.S_IMODE(target.stat().st_mode) == 0o555
    for name in repair.ARTIFACT_NAMES:
        assert (source / name).read_bytes() == (target / name).read_bytes()
        assert (source / name).stat().st_ino != (target / name).stat().st_ino
        assert (source / name).stat().st_nlink == (target / name).stat().st_nlink == 1


def test_injected_checker_failure_leaves_complete_private_target(tmp_path: Path) -> None:
    def fail_checker(*args) -> None:
        raise RuntimeError("injected checker failure")

    source, target = _source(tmp_path), tmp_path / "canonical"
    with pytest.raises(RuntimeError, match="injected checker"):
        repair._publish_fixture(authority=repair._FIXTURE_TOKEN, source=source,
                                target=target, checker_hook=fail_checker)
    assert stat.S_IMODE(target.stat().st_mode) == 0o700
    assert sorted(path.name for path in target.iterdir()) == sorted(repair.ARTIFACT_NAMES)
    assert (target / "COMPLETE").is_file()


def test_target_replacement_is_detected_without_touching_replacement(tmp_path: Path) -> None:
    source, target = _source(tmp_path), tmp_path / "canonical"

    def replace_target(config_path: Path, source_fd: int, parent_fd: int,
                       target_fd: int, target_name: str) -> None:
        os.rename(target_name, "held-original", src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        os.mkdir(target_name, 0o700, dir_fd=parent_fd)

    with pytest.raises(RuntimeError, match="no longer binds"):
        repair._publish_fixture(authority=repair._FIXTURE_TOKEN, source=source,
                                target=target, checker_hook=replace_target)
    replacement = target
    original = tmp_path / "held-original"
    assert stat.S_IMODE(replacement.stat().st_mode) == 0o700
    assert list(replacement.iterdir()) == []
    assert stat.S_IMODE(original.stat().st_mode) == 0o700
    assert (original / "COMPLETE").is_file()


def test_copy_failure_leaves_private_partial_and_retry_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, target = _source(tmp_path), tmp_path / "canonical"
    original, calls = repair._copy_artifact, 0

    def fail_second(fd: int, name: str, payload: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("copy failure")
        original(fd, name, payload)

    monkeypatch.setattr(repair, "_copy_artifact", fail_second)
    with pytest.raises(OSError, match="copy failure"):
        repair._publish_fixture(authority=repair._FIXTURE_TOKEN, source=source, target=target)
    assert stat.S_IMODE(target.stat().st_mode) == 0o700
    with pytest.raises(FileExistsError, match="no retry"):
        repair._publish_fixture(authority=repair._FIXTURE_TOKEN, source=source, target=target)


def test_source_mutation_detected_before_checker_and_target_stays_private(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, target = _source(tmp_path), tmp_path / "canonical"
    original, calls = repair._copy_artifact, 0

    def mutate(fd: int, name: str, payload: bytes) -> None:
        nonlocal calls
        original(fd, name, payload)
        calls += 1
        if calls == 1:
            source.chmod(0o755)
            victim = source / "COMPLETE"
            victim.chmod(0o644)
            victim.write_bytes(victim.read_bytes() + b" ")
            victim.chmod(0o444)
            source.chmod(0o555)

    monkeypatch.setattr(repair, "_copy_artifact", mutate)
    with pytest.raises(RuntimeError, match="source artifact changed"):
        repair._publish_fixture(authority=repair._FIXTURE_TOKEN, source=source, target=target)
    assert stat.S_IMODE(target.stat().st_mode) == 0o700


def test_existing_target_is_never_overwritten(tmp_path: Path) -> None:
    source, target = _source(tmp_path), tmp_path / "canonical"
    target.mkdir()
    marker = target / "marker"
    marker.write_text("keep")
    with pytest.raises(FileExistsError, match="no retry"):
        repair._publish_fixture(authority=repair._FIXTURE_TOKEN, source=source, target=target)
    assert marker.read_text() == "keep"


def test_actual_staging_read_only_preflight() -> None:
    assert repair.validate_source_only(CONFIG)["artifacts"] == 5
