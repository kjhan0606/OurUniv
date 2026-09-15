from __future__ import annotations

import base64
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zlib

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cf4_lg_unconstrained_p1_reference as producer


def checker_module():
    path = ROOT / "scripts/check_cf4_lg_unconstrained_p1_reference_v1.py"
    spec = importlib.util.spec_from_file_location("reference_checker_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def design():
    return json.loads((ROOT / "config/cf4_lg_unconstrained_p1_reference_design_v1.json").read_bytes())


def assign_dotted(root, dotted, value):
    target = root
    parts = dotted.split(".")
    cursor = 0
    while cursor < len(parts) - 1:
        part = parts[cursor]
        target = target.setdefault(part, {})
        cursor += 1
        if part in {"mean_delta_profile", "spheres"}:
            decimal_key = ".".join(parts[cursor:cursor + 2])
            cursor += 2
            if cursor == len(parts):
                target[decimal_key] = value
                return
            target = target.setdefault(decimal_key, {})
    target[parts[cursor]] = value


def synthetic_score(contract, margins=None):
    margins = margins or {}
    score = {}
    component_pass = {}
    for item in contract["components"]:
        margin = float(margins.get(item["id"], 0.25))
        sign = 1 if item["comparison"] in {"GE", "GT"} else -1
        value = float(item["threshold"]) + sign * float(item["denominator"]) * margin
        if item["id"] == "LocalVoid.n_underdense":
            value = int(value)
        assign_dotted(score, item["value_path"], value)
        component_pass[item["id"]] = margin >= 0 if item["comparison"] in {"GE", "LE"} else margin > 0
    gates = {g: all(component_pass[x["id"]] for x in contract["components"] if x["gate"] == g)
             for g in producer.GATES}
    score["gates"] = gates; score["pass"] = all(gates.values()); score["n_gates_passed"] = sum(gates.values())
    return score


def fake_specs():
    return [
        {"stage": "initial_white_field", "field_name": "initial_white_field_sha256", "domain_tag": "tag:i"},
        {"stage": "unsmoothed_z0_cic_density", "field_name": "unsmoothed_z0_cic_density_sha256", "domain_tag": "tag:d"},
        {"stage": "smoothed_delta_scorer_input", "field_name": "smoothed_delta_scorer_input_sha256", "domain_tag": "tag:s"},
    ]


def fake_row(index, derived, score, specs=None):
    specs = specs or fake_specs(); digest = f"{index + 1:064x}"
    row = {"schema": producer.ROW_SCHEMA, "reference_index": index, "batch_index": index // 16,
           "within_batch_index": index % 16, "seed_digest_sha256": digest, "seed_uint64": index + 10,
           "jax_key_words": [0, index + 10], "score_member": score,
           "seed_manifest_sha256": "a" * 64, "grant_sha256": "b" * 64,
           "grant_commit": "c" * 40, "runtime_pins_sha256": "d" * 64,
           "implementation_commit": "e" * 40,
           "pareto": {"dominates_count": 0, "dominated_by_count": 0, "nondominated": True}, **derived}
    receipts = []
    for stage_i, spec in enumerate(specs):
        value = f"{100000 + stage_i * 1000 + index:064x}"
        row[spec["field_name"]] = value
        receipt = {"stage": spec["stage"], "field_name": spec["field_name"], "domain_tag": spec["domain_tag"],
                   "frame_version": "field-frame-v1", "dtype": "<f4", "shape": [192, 192, 192],
                   "reference_index": index, "seed_digest_sha256": digest, "producer_sha256": value,
                   "independent_checker_sha256": value, "pass": True}
        if stage_i == 2: receipt["post_score_sha256"] = value
        receipts.append(receipt)
    row["streaming_receipts"] = receipts
    return row


def test_import_and_test_only_never_import_jax_pmwd():
    assert "jax" not in sys.modules and "pmwd" not in sys.modules
    program = producer.load_program(ROOT / "config/cf4_lg_unconstrained_p1_reference_program_v1.json")
    manifest = producer.load_seed_manifest(program)
    assert len(manifest["seed_derivation"]["rows"]) == 768
    result = subprocess.run([sys.executable, str(ROOT / "src/cf4_lg_unconstrained_p1_reference.py"),
                             "--config", str(ROOT / "config/cf4_lg_unconstrained_p1_reference_program_v1.json"),
                             "--test-only"], check=True, text=True, stdout=subprocess.PIPE)
    assert result.stdout.strip() == "TEST_ONLY_PASS_NO_SCIENCE_IMPORTS"


def test_independent_field_framers_match_small_canonical_array():
    array = np.arange(8, dtype="<f4").reshape(2, 2, 2)
    assert producer.producer_frame_hash(array, "domain", (2, 2, 2)) == \
        checker_module().checker_frame_live_array(array, "domain", (2, 2, 2))


@pytest.mark.parametrize("bad", [
    np.arange(8, dtype="<f8").reshape(2, 2, 2),
    np.asfortranarray(np.arange(8, dtype="<f4").reshape(2, 2, 2)),
    np.arange(4, dtype="<f4").reshape(2, 2),
])
def test_correlated_forgery_dtype_order_shape_rejected(bad):
    with pytest.raises(RuntimeError): checker_module().checker_frame_live_array(bad, "domain", (2, 2, 2))


def test_margin_zero_rules_and_exact_five_gates(design):
    contract = design["margin_and_joint_diagnostics_contract"]
    score = synthetic_score(contract)
    row = producer.derive_member(score, contract)
    assert row["all_five_pass"] and list(row["gates"]) == list(producer.GATES)
    strict = synthetic_score(contract, {"Virgo.target_delta_positive": 0.0})
    strict["gates"]["Virgo"] = False; strict["pass"] = False; strict["n_gates_passed"] = 4
    result = producer.derive_member(strict, contract)
    assert not result["gates"]["Virgo"] and result["exactly_four_of_five"]
    inclusive = synthetic_score(contract, {"Virgo.target_shell_percentile": 0.0})
    assert producer.derive_member(inclusive, contract)["gates"]["Virgo"]


def test_realistic_decimal_mapping_keys_use_independent_longest_resolvers(design):
    contract = design["margin_and_joint_diagnostics_contract"]
    score = synthetic_score(contract)
    assert set(score["bootes_void"]["mean_delta_profile"]) == {"12.0", "24.0"}
    assert set(score["observer_environment"]["spheres"]) == {"5.0", "8.0"}
    expected = producer.derive_member(score, contract)
    assert checker_module().independent_derive(score, contract) == expected


def test_correlated_forgery_seed_changed_reused_receipts_rejected(design):
    c = design["margin_and_joint_diagnostics_contract"]; score = synthetic_score(c)
    row = fake_row(0, producer.derive_member(score, c), score)
    seed_row = [0, row["seed_digest_sha256"], row["seed_uint64"], 0, row["seed_uint64"]]
    producer.validate_row_binding(row, seed_row)
    forged = copy.deepcopy(row); forged["seed_uint64"] += 1
    with pytest.raises(RuntimeError): producer.validate_row_binding(forged, seed_row)


def test_correlated_forgery_swapped_stage_hash_rejected(design):
    c = design["margin_and_joint_diagnostics_contract"]; score = synthetic_score(c)
    row = fake_row(0, producer.derive_member(score, c), score)
    row["streaming_receipts"][0]["domain_tag"], row["streaming_receipts"][1]["domain_tag"] = \
        row["streaming_receipts"][1]["domain_tag"], row["streaming_receipts"][0]["domain_tag"]
    with pytest.raises(RuntimeError): checker_module().verify_receipts(row, fake_specs())


def test_correlated_forgery_score_changed_with_hash_retained_rejected(design):
    c = design["margin_and_joint_diagnostics_contract"]; score = synthetic_score(c)
    derived = producer.derive_member(score, c); row = fake_row(0, derived, score)
    seed_row = [0, row["seed_digest_sha256"], row["seed_uint64"], 0, row["seed_uint64"]]
    checker_module().verify_member(row, seed_row, c, fake_specs())
    forged = copy.deepcopy(row); forged["score_member"]["clusters"]["Virgo"]["target_delta"] += 1.0
    with pytest.raises(RuntimeError): checker_module().verify_member(forged, seed_row, c, fake_specs())


def test_correlated_forgery_runtime_binding_rejected(design):
    c = design["margin_and_joint_diagnostics_contract"]; score = synthetic_score(c)
    row = fake_row(0, producer.derive_member(score, c), score)
    seed_row = [0, row["seed_digest_sha256"], row["seed_uint64"], 0, row["seed_uint64"]]
    binding = {name: row[name] for name in ("seed_manifest_sha256", "grant_sha256", "grant_commit",
                                            "runtime_pins_sha256", "implementation_commit")}
    checker_module().verify_member(row, seed_row, c, fake_specs(), binding)
    forged = copy.deepcopy(row); forged["grant_sha256"] = "f" * 64
    with pytest.raises(RuntimeError):
        checker_module().verify_member(forged, seed_row, c, fake_specs(), binding)


def test_live_scorer_mutation_rejected():
    array = np.arange(8, dtype="<f4").reshape(2, 2, 2)
    spec = {"stage": "smoothed_delta_scorer_input", "field_name": "x", "domain_tag": "tag"}
    def mutating(x):
        x.setflags(write=True); x.flat[0] += 1
        return {}
    with pytest.raises(RuntimeError):
        producer.score_with_integrity(array, mutating, (), spec, 0, "a" * 64, (2, 2, 2))


def test_initial_and_density_handoffs_become_read_only():
    for stage, field in (("initial_white_field", "initial_white_field_sha256"),
                         ("unsmoothed_z0_cic_density", "unsmoothed_z0_cic_density_sha256")):
        array = np.arange(8, dtype="<f4").reshape(2, 2, 2)
        producer.streaming_receipt(array, {"stage": stage, "field_name": field,
                                           "domain_tag": f"tag:{stage}"},
                                   0, "a" * 64, (2, 2, 2))
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = 99


def test_correlated_forgery_missing_duplicate_cross_stage_rejected(design):
    c = design["margin_and_joint_diagnostics_contract"]; rows = []
    for i in range(3):
        score = synthetic_score(c); rows.append(fake_row(i, producer.derive_member(score, c), score))
    producer.validate_stage_hash_sets(rows, 3)
    rows[1]["initial_white_field_sha256"] = rows[0]["initial_white_field_sha256"]
    with pytest.raises(RuntimeError): producer.validate_stage_hash_sets(rows, 3)


def test_producer_and_checker_summaries_are_exact_on_synthetic_rows(design):
    c = design["margin_and_joint_diagnostics_contract"]; rows = []
    for i in range(4):
        score = synthetic_score(c, {"Virgo.target_delta_positive": 0.1 + i * 0.1})
        rows.append(fake_row(i, producer.derive_member(score, c), score))
    intersections={"seed_uint64":[],"jax_key_words":[]}
    left = producer.build_summary(copy.deepcopy(rows), c["component_order"],intersections)
    right = checker_module().independent_summary(copy.deepcopy(rows), c["component_order"],intersections)
    assert left == right
    assert len(left["patterns"]) == 32 and len(left["cofailure"]["count"]) == 5


def test_lineage_pure_validator_rejects_wrong_parent():
    program = json.loads((ROOT / "config/cf4_lg_unconstrained_p1_reference_program_v1.json").read_bytes())
    paths = program["lineage"]["implementation_exact_added_paths"]
    grant = {"implementation": {"commit": "impl", "files": [
        {"path": p, "mode": "100644", "sha256": "h" + str(i)} for i, p in enumerate(paths)]}}
    modes = {p: "100644" for p in paths}; hashes = {p: "h" + str(i) for i, p in enumerate(paths)}
    kwargs = dict(head="grant", upstream="grant", head_parents=["impl"],
                  implementation_parents=[program["lineage"]["required_parent_commit"]],
                  implementation_rows=[("A", p) for p in paths],
                  grant_rows=[("A", program["lineage"]["future_grant_path"])],
                  implementation_modes=modes, implementation_hashes=hashes)
    producer.validate_lineage_values(program, grant, **kwargs)
    kwargs["implementation_parents"] = ["wrong"]
    with pytest.raises(RuntimeError): producer.validate_lineage_values(program, grant, **kwargs)


def test_single_writer_publish_and_existing_refusal(tmp_path):
    first = tmp_path / "first"; first.mkdir(); final = tmp_path / "final"
    producer._publish_single_writer(first, final)
    assert final.is_dir() and not first.exists()
    second = tmp_path / "second"; second.mkdir()
    with pytest.raises(FileExistsError): producer._publish_single_writer(second, final)
    assert second.is_dir()
    source = (ROOT / "src/cf4_lg_unconstrained_p1_reference.py").read_text()
    assert "os.rename(source, target)" in source
    assert "renameat2" not in source


def test_program_is_canonical_and_execution_unauthorized():
    path = ROOT / "config/cf4_lg_unconstrained_p1_reference_program_v1.json"
    value = json.loads(path.read_bytes())
    assert path.read_bytes() == producer.canonical_bytes(value)
    assert value["authorization"]["reference_execution"] is False
    assert value["resources"]["requested_host_memory_GiB"] >= 1.2 * value["resources"]["estimated_peak_host_memory_GiB"]


def test_program_and_grant_parsers_reject_extra_and_noncanonical_bytes(tmp_path):
    program_path = ROOT / "config/cf4_lg_unconstrained_p1_reference_program_v1.json"
    program = json.loads(program_path.read_bytes())
    extra = dict(program); extra["unexpected"] = True
    bad_extra = tmp_path / "extra.json"; bad_extra.write_bytes(producer.canonical_bytes(extra))
    with pytest.raises(ValueError): producer.load_canonical_json(bad_extra, producer.PROGRAM_KEYS, "program")
    bad_pretty = tmp_path / "pretty.json"; bad_pretty.write_text(json.dumps(program, indent=2) + "\n")
    with pytest.raises(ValueError): producer.load_canonical_json(bad_pretty, producer.PROGRAM_KEYS, "program")
    grant_keys = set(program["grant_contract"]["exact_top_level_keys"])
    grant = {key: None for key in grant_keys}; grant["extra"] = False
    bad_grant = tmp_path / "grant.json"; bad_grant.write_bytes(producer.canonical_bytes(grant))
    with pytest.raises(ValueError): producer.load_canonical_json(bad_grant, grant_keys, "grant")
    exact_grant = {key: None for key in grant_keys}
    pretty_grant = tmp_path / "pretty-grant.json"
    pretty_grant.write_text(json.dumps(exact_grant, indent=2) + "\n")
    with pytest.raises(ValueError): producer.load_canonical_json(pretty_grant, grant_keys, "grant")


def test_grant_authorization_scope_and_gpu_allocation_tokens():
    program = json.loads((ROOT / "config/cf4_lg_unconstrained_p1_reference_program_v1.json").read_bytes())
    contract = program["grant_contract"]
    assert set(contract["authorization_required_true"]) == {
        "reference_execution", "GPFS_read", "GPFS_write", "output_staging", "IC_generation",
        "field_generation", "PM_forward", "scoring", "publication"}
    assert set(contract["authorization_required_false"]) == {
        "retry", "resubmit", "replacement", "GPFS_overwrite", "automatic_promotion",
        "threshold_change", "downstream_execution", "Slurm_submission", "manual_execution",
        "ranking", "promotion", "HOP", "RAMSES"}
    assert producer.allocated_gpu_tokens({"CUDA_VISIBLE_DEVICES": "0", "SLURM_JOB_GPUS": "3",
                                          "SLURM_GPUS_ON_NODE": "gpu:1"}, 1) == ["3"]
    uuid = "GPU-12345678-abcd-1234-abcd-123456789abc"
    assert producer.allocated_gpu_tokens({"CUDA_VISIBLE_DEVICES": "0","SLURM_JOB_GPUS":uuid,
                                          "SLURM_GPUS_ON_NODE": "1"}, 1) == [uuid]
    for environment in ({"CUDA_VISIBLE_DEVICES": "0,1", "SLURM_GPUS_ON_NODE": "2"},
                        {"CUDA_VISIBLE_DEVICES": "GPU-not-a-uuid", "SLURM_GPUS_ON_NODE": "1"},
                        {"CUDA_VISIBLE_DEVICES": "0", "SLURM_GPUS_ON_NODE": "2"}, {}):
        with pytest.raises(RuntimeError): producer.allocated_gpu_tokens(environment, 1)


def test_repository_identity_is_stable_before_and_after_exact_grant(monkeypatch):
    program = json.loads((ROOT / "config/cf4_lg_unconstrained_p1_reference_program_v1.json").read_bytes())
    parent = program["lineage"]["required_parent_commit"]; paths = program["lineage"]["implementation_exact_added_paths"]
    grant_path = program["lineage"]["future_grant_path"]
    state = {"head": "implementation"}
    def fake_git(*args):
        if args == ("rev-parse", "HEAD"): return state["head"]
        if args[:4] == ("rev-list", "--parents", "-n", "1"):
            commit = args[4]
            return f"grant implementation" if commit == "grant" else f"implementation {parent}"
        if args[:3] == ("diff", "--no-renames", "--name-status"):
            before, after = args[3], args[4]
            if (before, after) == ("implementation", "grant"): return f"A\t{grant_path}"
            if (before, after) == (parent, "implementation"):
                return "\n".join(f"A\t{path}" for path in paths)
        if args[:2] == ("ls-tree", "implementation"):
            path = args[-1]; return f"100644 blob deadbeef\t{path}"
        raise AssertionError(args)
    monkeypatch.setattr(producer, "_git", fake_git)
    before = producer.repository_implementation_identity(program)
    state["head"] = "grant"
    after = producer.repository_implementation_identity(program)
    assert before == after and before["commit"] == "implementation"


def test_committed_seed_manifest_has_exact_canonical_keyset():
    path = ROOT / "config/cf4_lg_unconstrained_p1_reference_seed_manifest_v1.json"
    value = json.loads(path.read_bytes())
    assert set(value) == producer.SEED_MANIFEST_KEYS
    assert path.read_bytes() == producer.canonical_bytes(value)


def test_checker_pure_seed_formula_and_lossless_provenance():
    checker = checker_module(); domain = "tiny-domain"; commit = "1" * 40; design_sha = "2" * 64
    rows = []
    for index in range(2):
        raw = hashlib.sha256(domain.encode() + b"\0" + commit.encode() + b"\0" +
                             design_sha.encode() + b"\0" + index.to_bytes(8, "big")).digest()
        rows.append([index, raw.hex(), int.from_bytes(raw[:8], "big"),
                     int.from_bytes(raw[:4], "big"), int.from_bytes(raw[4:8], "big")])
    checker.verify_reference_rows(rows, domain, commit, design_sha, 2)
    occurrences = [[0, "/seed", 7, "typed_integer_leaf", None]]
    raw = json.dumps(occurrences, sort_keys=True, separators=(",", ":")).encode()
    compressed = zlib.compress(raw, 9)
    seed = {"forbidden_inventory": {"occurrence_count": 1, "lossless_occurrence_record": {
        "base64": base64.b64encode(compressed).decode(), "compressed_size": len(compressed),
        "compressed_sha256": hashlib.sha256(compressed).hexdigest(), "uncompressed_size": len(raw),
        "uncompressed_sha256": hashlib.sha256(raw).hexdigest()}}}
    assert checker.decode_forbidden_occurrences(seed) == occurrences


def test_typed_source_occurrences_provenance_and_discovery_fail_closed(tmp_path):
    checker=checker_module()
    document={"alpha_seed":7,"group":{"count":2,"worker_seed_start":20},
              "master_seeds":[8,9],"proposal_seed_range_python":[30,32],
              "z_seed_range_inclusive":[40,41],"notseed":999}
    rows,raw,jax=checker.typed_json_occurrences(document,4)
    assert raw==8 and not jax
    assert rows==[[4,"/alpha_seed",7,"typed_integer_leaf",None],
                  [4,"/group/worker_seed_start",20,"start_plus_count",0],
                  [4,"/group/worker_seed_start",21,"start_plus_count",1],
                  [4,"/master_seeds/0",8,"typed_integer_array",None],
                  [4,"/master_seeds/1",9,"typed_integer_array",None],
                  [4,"/proposal_seed_range_python",30,"range_python",0],
                  [4,"/proposal_seed_range_python",31,"range_python",1],
                  [4,"/z_seed_range_inclusive",40,"range_inclusive",0],
                  [4,"/z_seed_range_inclusive",41,"range_inclusive",1]]
    forged=copy.deepcopy(rows);forged[0][1]="/wrong"
    with pytest.raises(RuntimeError):checker.require_exact_occurrences(rows,forged)
    blobs={"config/cf4_a.json":b'{"seed":1}',"config/p1_b.json":b'{"value":1}',
           "config/v3_c.json":b'{"master_seed":2}',"other/cf4_x.json":b'{"seed":3}'}
    discovered=checker.discover_seed_json_blobs(blobs)
    assert discovered==["config/cf4_a.json","config/v3_c.json"]
    checker.require_exact_discovery(discovered,list(discovered))
    with pytest.raises(RuntimeError):checker.require_exact_discovery(discovered,discovered[:-1])
    with pytest.raises(RuntimeError):checker.require_exact_discovery(discovered,discovered+["config/p2_extra.json"])
    with pytest.raises(RuntimeError):checker.discover_seed_json_blobs(
        {"config/cf4_duplicate.json":b'{"seed":1,"seed":2}'})


def test_lossless_occurrence_row_types_and_ranges_are_strict():
    checker=checker_module();occurrences=[[True,"/seed",1,"typed_integer_leaf",None]]
    raw=json.dumps(occurrences,separators=(",",":")).encode();compressed=zlib.compress(raw,9)
    seed={"forbidden_inventory":{"occurrence_count":1,"lossless_occurrence_record":{
        "base64":base64.b64encode(compressed).decode(),"compressed_size":len(compressed),
        "compressed_sha256":hashlib.sha256(compressed).hexdigest(),"uncompressed_size":len(raw),
        "uncompressed_sha256":hashlib.sha256(raw).hexdigest()}}}
    with pytest.raises(RuntimeError):checker.decode_forbidden_occurrences(seed)


def test_explicit_json_and_npz_selectors_exclude_geometry_keys(tmp_path):
    checker=checker_module();document={"entries":[{"seed":3},{"seed":4}],"other_seed":99}
    rows,raw=checker.explicit_json_selector_occurrences(document,97,["/entries/*/seed"])
    assert raw==2 and [row[2] for row in rows]==[3,4]
    path=tmp_path/"schedule.npz"
    np.savez(path,parent_seed=np.array([5,6],dtype=np.int16),
             fine_field_seed=np.array([7,8],dtype=np.int64),
             likelihood_noise_seed=np.array([9,10],dtype=np.uint64),
             keys=np.array([[-2,0],[1,2]],dtype=np.int16))
    selectors=["parent_seed[:]","fine_field_seed[:]","likelihood_noise_seed[:]"]
    rows,raw=checker.explicit_npz_selector_occurrences(path,99,selectors,{"keys"})
    assert raw==6 and [row[2] for row in rows]==[5,6,7,8,9,10]
    with pytest.raises(RuntimeError):checker.explicit_npz_selector_occurrences(path,99,["keys[:,:]"] ,{"keys"})
    with pytest.raises(RuntimeError):checker.explicit_npz_selector_occurrences(path,99,["keys[:]"] ,{"keys"})


def test_local_parent_git_discovery_and_97_occurrences_match_committed_manifest():
    checker=checker_module();seed=json.loads(
        (ROOT/"config/cf4_lg_unconstrained_p1_reference_seed_manifest_v1.json").read_bytes())
    parent=seed["sources"]["parent_git_commit"];tracked=seed["sources"]["rows"][:97]
    names=subprocess.check_output(["git","-C",str(ROOT),"ls-tree","-r","--name-only",parent,"--","config"],text=True).splitlines()
    names=[path for path in names if Path(path).parent==Path("config") and Path(path).name.endswith(".json")
           and Path(path).name.startswith(("cf4_","p1_","p2_","v3_"))]
    blobs={path:subprocess.check_output(["git","-C",str(ROOT),"show",f"{parent}:{path}"]) for path in names}
    checker.require_exact_discovery(checker.discover_seed_json_blobs(blobs),[row[2] for row in tracked])
    recomputed=[]
    for row in tracked:
        extracted,raw,jax=checker.typed_json_occurrences(checker.strict_json_bytes(blobs[row[2]]),row[0])
        assert (raw,len(extracted),len(jax))==(row[6],row[7],row[8]);recomputed.extend(extracted)
    recorded=checker.decode_forbidden_occurrences(seed)
    checker.require_exact_occurrences(recomputed,recorded[:len(recomputed)])


def test_summary_inventory_integrity_and_forgery_rejected(design):
    checker=checker_module();contract=design["margin_and_joint_diagnostics_contract"]
    score=synthetic_score(contract);rows=[fake_row(0,producer.derive_member(score,contract),score)]
    intersections={"seed_uint64":[],"jax_key_words":[]}
    summary=checker.independent_summary(rows,contract["component_order"],intersections)
    assert summary["forbidden_inventory_disjointness"]["pass"] is True
    assert summary["stage_hash_integrity"]["cross_stage_confusion_count"]==0
    forged=copy.deepcopy(summary);forged["forbidden_inventory_disjointness"]["pass"]=1
    with pytest.raises(RuntimeError):checker.verify_summary_exact(forged,summary)
    with pytest.raises(RuntimeError):producer.build_summary(rows,contract["component_order"],
                                                            {"seed_uint64":[1],"jax_key_words":[]})


def test_exact_boolean_fields_reject_integer_forgery(design):
    checker=checker_module();contract=design["margin_and_joint_diagnostics_contract"]
    score=synthetic_score(contract);score["gates"]["Virgo"]=1
    with pytest.raises(RuntimeError):producer.derive_member(score,contract)
    score=synthetic_score(contract);derived=producer.derive_member(score,contract)
    row=fake_row(0,derived,score);seed=[0,row["seed_digest_sha256"],row["seed_uint64"],0,row["seed_uint64"]]
    row["all_five_pass"]=1
    with pytest.raises(RuntimeError):checker.verify_member(row,seed,contract,fake_specs())
    row=fake_row(0,derived,score);row["streaming_receipts"][0]["pass"]=1
    with pytest.raises(RuntimeError):checker.verify_member(row,seed,contract,fake_specs())
    score=synthetic_score(contract);score["n_gates_passed"]=5.0
    with pytest.raises(RuntimeError):producer.derive_member(score,contract)
    with pytest.raises(RuntimeError):checker.independent_derive(score,contract)


@pytest.mark.parametrize("mutation",["cosmology","smoothing","output","field","margin"])
def test_critical_contract_mutations_rejected(design,mutation):
    program=json.loads((ROOT/"config/cf4_lg_unconstrained_p1_reference_program_v1.json").read_bytes())
    changed_program=copy.deepcopy(program);changed_design=copy.deepcopy(design)
    if mutation=="cosmology":changed_program["science"]["cosmology"]["Om"]=.32
    elif mutation=="smoothing":changed_program["science"]["density_smoothing_mpc_h"]=5.0
    elif mutation=="output":changed_program["outputs"]["canonical_root"]+="_wrong"
    elif mutation=="field":changed_design["field_hash_contract"]["per_member_fields"][0]["domain_tag"]+="-wrong"
    else:changed_design["margin_and_joint_diagnostics_contract"]["components"][0]["denominator"]=2.0
    with pytest.raises(RuntimeError):producer.validate_critical_contracts(changed_program,changed_design)
    with pytest.raises(RuntimeError):checker_module().independently_validate_critical_contracts(changed_program,changed_design)


def test_seal_schema_status_science_and_items_fail_closed():
    checker=checker_module()
    manifest={"schema":"ouruniv-cf4-lg-unconstrained-p1-reference-seal-manifest-v1","status":"complete",
              "files":[{"name":name,"size":1,"sha256":"a"*64,"mode":"0444"}
                       for name in ("input_manifest.json","member_metrics.jsonl","summary.json")]}
    complete={"schema":"ouruniv-cf4-lg-unconstrained-p1-reference-complete-v1","status":"complete",
              "scientific_result":"descriptive_only","automatic_promotion":False}
    checker.verify_seal_contract_values(manifest,complete)
    for target,key,value in ((manifest,"schema","wrong"),(manifest,"status","wrong"),
                             (complete,"schema","wrong"),(complete,"status","wrong"),
                             (complete,"scientific_result","wrong"),(complete,"automatic_promotion",True)):
        forged=copy.deepcopy(target);forged[key]=value
        with pytest.raises(RuntimeError):checker.verify_seal_contract_values(
            forged if target is manifest else manifest,forged if target is complete else complete)
    forged=copy.deepcopy(manifest);forged["files"][0]["extra"]=1
    with pytest.raises(RuntimeError):checker.verify_seal_contract_values(forged,complete)
    for key,value in (("mode","0644"),("size",True),("sha256","A"*64)):
        forged=copy.deepcopy(manifest);forged["files"][0][key]=value
        with pytest.raises(RuntimeError):checker.verify_seal_contract_values(forged,complete)


def test_strict_slurm_resource_parsers_and_no_reacquire():
    program=json.loads((ROOT/"config/cf4_lg_unconstrained_p1_reference_program_v1.json").read_bytes())
    base={"NumNodes":"1","NumTasks":"1","NumCPUs":"16","CPUs/Task":"16","Partition":"h200",
          "TimeLimit":"1-00:00:00","Requeue":"0","Restarts":"0","ReqTRES":"cpu=16,mem=20G,gres/gpu=1",
          "TresPerNode":"gres/gpu:h200:1","AllocTRES":"cpu=16,mem=20G,gres/gpu=1"}
    producer.validate_slurm_resources(base,program["resources"])
    assert producer.slurm_gpu_counts("TresPerNode=gres/gpu:1 AllocTRES=gres/gpu=1")==[1,1]
    assert producer.slurm_gpu_counts("gres/gpu:h200:1 gres/gpu:h200=1")==[1,1]
    assert producer.slurm_memory_mib("20G")==20480 and producer.slurm_memory_mib("20480M")==20480
    assert producer.slurm_time_seconds("24:00:00")==producer.slurm_time_seconds("1-00:00:00")==86400
    for key,value in (("NumNodes","2"),("NumTasks","2"),("NumCPUs","15"),("CPUs/Task","15"),
                      ("Requeue","1"),("Restarts","1"),("TimeLimit","23:59:59")):
        forged=dict(base);forged[key]=value
        with pytest.raises(RuntimeError):producer.validate_slurm_resources(forged,program["resources"])
