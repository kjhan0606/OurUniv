#!/usr/bin/env python3
"""Independent offline sealed-output checker for P1 reference v1.

This file intentionally imports no producer module and duplicates framing,
margin, aggregation, and seal validation logic from the declarative contracts.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import zlib
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GATE_ORDER = ("Virgo", "Coma", "LocalVoid", "BootesVoid", "ObserverEnvironment")
ENTRY_SET = {"input_manifest.json", "member_metrics.jsonl", "summary.json", "manifest.json", "COMPLETE"}
PROGRAM_KEYS = {"schema", "status", "date", "purpose", "lineage", "inputs", "sampling",
                "science", "field_integrity", "diagnostics", "outputs", "grant_contract",
                "resources", "held_allocation_protocol", "execution", "authorization",
                "forbidden", "audit_sequence"}
GRANT_KEYS = {"schema", "status", "date", "purpose", "lineage", "program", "implementation",
              "seed_manifest", "allocation_receipt", "runtime_pins", "outputs", "authorization"}
GRANT_AUTH_TRUE = {"reference_execution", "GPFS_read", "GPFS_write", "output_staging",
                   "IC_generation", "field_generation", "PM_forward", "scoring", "publication"}
GRANT_AUTH_FALSE = {"retry", "resubmit", "replacement", "GPFS_overwrite", "automatic_promotion",
                    "threshold_change", "downstream_execution", "Slurm_submission", "manual_execution",
                    "ranking", "promotion", "HOP", "RAMSES"}
SEED_MANIFEST_KEYS = {"audit", "authorization", "date", "forbidden", "forbidden_inventory",
                      "frozen", "independent_verification", "integrity", "lineage", "purpose",
                      "reference_inventory", "schema", "seed_derivation", "sources", "status",
                      "structural_in_place_update"}


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode()


def _object_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_regular_nofollow(path: Path,expected_mode: str) -> bytes:
    descriptor=os.open(path,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
    try:
        metadata=os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or format(stat.S_IMODE(metadata.st_mode),"04o")!=expected_mode:
            raise RuntimeError(f"external source type/mode mismatch: {path}")
        chunks=[]
        while True:
            block=os.read(descriptor,1<<20)
            if not block:break
            chunks.append(block)
        return b"".join(chunks)
    finally:os.close(descriptor)


def strict_json_bytes(data: bytes) -> Any:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result={}
        for key,value in rows:
            if key in result: raise RuntimeError(f"duplicate JSON key: {key}")
            result[key]=value
        return result
    try:
        return json.loads(data.decode("utf-8"),object_pairs_hook=pairs,
                          parse_constant=lambda token: (_ for _ in ()).throw(
                              RuntimeError(f"nonfinite JSON token: {token}")))
    except UnicodeDecodeError as error:
        raise RuntimeError("source JSON is not strict UTF-8") from error


def _pointer(parent: str, token: str | int) -> str:
    escaped = str(token).replace("~", "~0").replace("/", "~1")
    return f"{parent}/{escaped}"


def _seed_key(name: str) -> bool:
    return "seed" in name.split("_") or "seeds" in name.split("_")


def _uint64(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**64:
        raise RuntimeError(f"invalid uint64 seed at {label}")
    return int(value)


def typed_json_occurrences(value: Any, source_index: int) -> tuple[list[list[Any]], int, list[list[int]]]:
    """Independently apply erratum-v1's recursive typed seed extraction rule."""
    found: list[list[Any]] = []; raw_count = 0; explicit_jax: list[list[int]] = []

    def walk(item: Any, location: str) -> None:
        nonlocal raw_count
        if isinstance(item, Mapping):
            for key in sorted(item):
                child = item[key]; child_location = _pointer(location, key)
                if key in {"jax_key_words", "key_data"} and isinstance(child, list) \
                        and len(child) == 2 and all(isinstance(x, int) and not isinstance(x, bool)
                                                    and 0 <= x < 2**32 for x in child):
                    explicit_jax.append([int(child[0]), int(child[1])])
                if key.endswith(("_seed_range_inclusive", "_seeds_range_inclusive")):
                    if not isinstance(child, list) or len(child) != 2:
                        raise RuntimeError(f"malformed inclusive seed range at {child_location}")
                    low, high = (_uint64(x, child_location) for x in child); raw_count += 2
                    if high < low: raise RuntimeError(f"descending inclusive seed range at {child_location}")
                    for offset, number in enumerate(range(low, high + 1)):
                        found.append([source_index, child_location, number, "range_inclusive", offset])
                    continue
                if key.endswith(("_seed_range_python", "_seeds_range_python")):
                    if not isinstance(child, list) or len(child) != 2:
                        raise RuntimeError(f"malformed Python seed range at {child_location}")
                    low, high = (_uint64(x, child_location) for x in child); raw_count += 2
                    if high < low: raise RuntimeError(f"descending Python seed range at {child_location}")
                    for offset, number in enumerate(range(low, high)):
                        found.append([source_index, child_location, number, "range_python", offset])
                    continue
                if key.endswith(("_seed_start", "_seeds_start")) and _seed_key(key) \
                        and isinstance(item.get("count"), int) and not isinstance(item.get("count"), bool):
                    start = _uint64(child, child_location); count = int(item["count"]); raw_count += 1
                    if count < 0 or start + count > 2**64:
                        raise RuntimeError(f"invalid seed start/count at {child_location}")
                    for offset in range(count):
                        found.append([source_index, child_location, start + offset,
                                      "start_plus_count", offset])
                    continue
                if _seed_key(key) and isinstance(child, int) and not isinstance(child, bool):
                    found.append([source_index, child_location, _uint64(child, child_location),
                                  "typed_integer_leaf", None]); raw_count += 1; continue
                if _seed_key(key) and isinstance(child, list) and child \
                        and all(isinstance(x, int) and not isinstance(x, bool) for x in child):
                    raw_count += len(child)
                    for offset, number in enumerate(child):
                        where = _pointer(child_location, offset)
                        found.append([source_index, where, _uint64(number, where),
                                      "typed_integer_array", None])
                    continue
                walk(child, child_location)
        elif isinstance(item, list):
            for offset, child in enumerate(item): walk(child, _pointer(location, offset))

    walk(value, "")
    return found, raw_count, explicit_jax


def explicit_json_selector_occurrences(value: Any, source_index: int,
                                       selectors: Sequence[str]) -> tuple[list[list[Any]], int]:
    answer: list[list[Any]] = []
    for selector in selectors:
        if not isinstance(selector, str) or not selector.startswith("/") or selector == "/":
            raise RuntimeError("invalid explicit JSON selector")
        tokens = selector[1:].split("/")
        states: list[tuple[Any, str]] = [(value, "")]
        for encoded in tokens:
            token = encoded.replace("~1", "/").replace("~0", "~")
            next_states: list[tuple[Any, str]] = []
            for current, location in states:
                if token == "*":
                    if isinstance(current, list):
                        next_states.extend((child, _pointer(location, index))
                                           for index, child in enumerate(current))
                    elif isinstance(current, Mapping):
                        next_states.extend((current[key], _pointer(location, key)) for key in sorted(current))
                    else: raise RuntimeError("selector wildcard applied to scalar")
                elif isinstance(current, Mapping) and token in current:
                    next_states.append((current[token], _pointer(location, token)))
                elif isinstance(current, list) and token.isdigit() and int(token) < len(current):
                    next_states.append((current[int(token)], _pointer(location, int(token))))
                else: raise RuntimeError("explicit selector did not resolve")
            states = next_states
        for selected, location in states:
            answer.append([source_index, location, _uint64(selected, location),
                           "explicit_selector", None])
    return answer, len(answer)


def explicit_npz_selector_occurrences(path_or_bytes: Path | bytes, source_index: int,
                                      selectors: Sequence[str], excluded_arrays: set[str]) \
        -> tuple[list[list[Any]], int]:
    answer: list[list[Any]] = []
    source=io.BytesIO(path_or_bytes) if isinstance(path_or_bytes,bytes) else path_or_bytes
    with np.load(source, allow_pickle=False) as archive:
        for selector in selectors:
            if not isinstance(selector, str) or not selector.endswith("[:]") \
                    or selector[:-3] in excluded_arrays:
                raise RuntimeError("NPZ selector is not an approved whole 1-D seed array")
            name = selector[:-3]
            if name not in archive.files: raise RuntimeError("selected NPZ array absent")
            array = archive[name]
            if array.ndim != 1 or array.dtype.kind not in "iu":
                raise RuntimeError("selected NPZ seed array dtype/shape mismatch")
            for offset, value in enumerate(array):
                where=f"{name}[{offset}]"; number=_uint64(int(value),where)
                answer.append([source_index,where,number,"explicit_npz_selector",None])
    return answer, len(answer)


def require_exact_occurrences(recomputed: Sequence[Sequence[Any]],
                              recorded: Sequence[Sequence[Any]]) -> None:
    if list(recomputed) != list(recorded):
        mismatch=next((index for index,(left,right) in enumerate(zip(recomputed,recorded))
                       if left!=right),min(len(recomputed),len(recorded)))
        raise RuntimeError(f"typed forbidden occurrence/provenance mismatch at row {mismatch}")


def require_exact_discovery(discovered: Sequence[str], authoritative: Sequence[str]) -> None:
    if list(discovered) != list(authoritative):
        raise RuntimeError("parent Git seed-source discovery omission/addition/order mismatch")


def discover_seed_json_blobs(blobs: Mapping[str, bytes]) -> list[str]:
    retained=[]
    for path in sorted(blobs):
        base=Path(path).name
        if Path(path).parent!=Path("config") or not base.endswith(".json") \
                or not base.startswith(("cf4_","p1_","p2_","v3_")):
            continue
        value=strict_json_bytes(blobs[path])
        occurrences,_raw,_jax=typed_json_occurrences(value,-1)
        if occurrences: retained.append(path)
    return retained


def independently_validate_critical_contracts(program: Mapping[str, Any], design: Mapping[str, Any]) -> None:
    expected_cosmology={"A_s_1e9":1.63,"Ob":0.05,"Om":0.31,"h":0.746,"ns":0.96}
    science=program["science"]
    if science.get("cosmology")!=expected_cosmology or science.get("mesh_N")!=192 \
            or science.get("spacing_mpc_h")!=2.0 or science.get("box_size_mpc_h")!=384.0 \
            or science.get("density_smoothing_mpc_h")!=4.0 or science.get("target_redshift")!=0.0 \
            or science.get("observer_offset_mpc_h")!=[0.0,0.0,0.0] \
            or science.get("gate_order")!=list(GATE_ORDER) \
            or science.get("initial_generator")!="PCG64 seed_uint64 standard_normal float64 cube then exactly one float32 cast" \
            or science.get("smoothing")!="gaussian_filter sigma=2 wrap; divide float64 mean; subtract 1; float32":
        raise RuntimeError("independent program science contract mismatch")
    frozen=design["frozen_forward_model_contract"]
    ds=design["science_contract"]
    if frozen["cosmology"]["canonical_object"]!=expected_cosmology or ds["mesh_N"]!=192 \
            or ds["box_size_mpc_h"]!=384.0 or ds["density_smoothing_mpc_h"]!=4.0 \
            or ds["target_redshift"]!=0.0 or ds["observer_offset_mpc_h"]!=[0.0,0.0,0.0] \
            or ds["exact_gate_order"]!=list(GATE_ORDER):
        raise RuntimeError("independent design science contract mismatch")
    dtype=frozen["dtypes"];generator=frozen["initial_condition_generator"]
    if dtype["IC_generator_draw"]!="float64 then one cast to float32 before forward" \
            or dtype["forward_field_particle_and_mesh"]!="float32" \
            or dtype["smoothed_density_mean"]!="float64 accumulator" \
            or generator["exact_operation"]!="np.random.Generator(np.random.PCG64(seed_uint64)).standard_normal(size=(192,192,192), dtype=np.float64), then astype(np.float32) once; this is the unconstrained xi draw used by the current posterior without the subsequent CF4 Matheron conditioning term" \
            or frozen["PM_forward"]["forward_call"]!="make_forward(192,2.0,jnp.float32,return_dens=True,cosmology=canonical_object); linear_modes -> 2LPT -> nbody -> scatter; take first returned object as z=0 density" \
            or "sigma 2.0 cells" not in frozen["density_and_smoothing"]["smoothing"] \
            or "mode='wrap'" not in frozen["density_and_smoothing"]["delta_formula"]:
        raise RuntimeError("independent generator/dtype/forward semantics mismatch")
    if program["inputs"]["P1_config"]!=ds["P1_config"] or program["inputs"]["scorer"]!=ds["scorer"] \
            or program["inputs"]["forward_factory"]!=frozen["PM_forward"]["forward_factory"]:
        raise RuntimeError("independent science pin mismatch")
    outputs=program["outputs"];declared=design["outputs_contract"]
    if outputs["canonical_root"]!=declared["prospective_canonical_root"] \
            or outputs["exact_files"]!=declared["exact_files"] or outputs["files_mode"]!="0444" \
            or outputs["final_directory_mode"]!="0555" or outputs["staging_directory_mode"]!="0700" \
            or outputs["atomic_no_overwrite"] is not True:
        raise RuntimeError("independent output contract mismatch")
    expected_domains=("ouruniv:cf4:lg:unconstrained-p1-reference:v1:initial-white-field",
        "ouruniv:cf4:lg:unconstrained-p1-reference:v1:unsmoothed-z0-cic-density",
        "ouruniv:cf4:lg:unconstrained-p1-reference:v1:smoothed-delta-scorer-input")
    fields=design["field_hash_contract"]
    if fields["frame"]["shape"]!=[192,192,192] or fields["frame"]["frame_version"]!="field-frame-v1" \
            or tuple(row["domain_tag"] for row in fields["per_member_fields"])!=expected_domains:
        raise RuntimeError("independent field contract mismatch")
    expected={"Virgo.target_delta_positive":(0.0,1.0),"Virgo.target_shell_percentile":(70.0,100.0),
        "Virgo.peak_shell_percentile":(90.0,100.0),"Virgo.peak_separation_mpc_h":(5.0,5.0),
        "Coma.target_delta_positive":(0.0,1.0),"Coma.target_shell_percentile":(70.0,100.0),
        "Coma.peak_shell_percentile":(90.0,100.0),"Coma.peak_separation_mpc_h":(8.0,8.0),
        "LocalVoid.n_underdense":(3,4.0),"LocalVoid.probe_mean_delta_negative":(0.0,1.0),
        "LocalVoid.median_centre_shell_percentile":(35.0,100.0),"BootesVoid.centre_shell_percentile":(35.0,100.0),
        "BootesVoid.mean_delta_radius_12_mpc_h_negative":(0.0,1.0),
        "BootesVoid.mean_delta_radius_24_mpc_h_negative":(0.0,1.0),
        "ObserverEnvironment.excess_mass_radius_5_mpc_h":(1e13,1e13),
        "ObserverEnvironment.mean_delta_radius_5_mpc_h":(-.5,.5),
        "ObserverEnvironment.excess_mass_radius_8_mpc_h":(5e13,5e13)}
    parts=design["margin_and_joint_diagnostics_contract"]["components"]
    if len(parts)!=17 or {row["id"]:(row["threshold"],row["denominator"]) for row in parts}!=expected:
        raise RuntimeError("independent margin contract mismatch")
    sampling=program["sampling"]
    if sampling["N_ref"]!=768 or sampling["batch_count"]!=48 or sampling["members_per_batch"]!=16 \
            or sampling["member_weight"]!=1/768 \
            or design["sampling_design"]["N_ref"]!=768 or design["sampling_design"]["batch_count"]!=48 \
            or design["sampling_design"]["members_per_batch"]!=16 \
            or any(design["weighting_contract"][key] is not False for key in
                   ("deduplication","importance_weights","normalize_again","proposal_correction","quality_weights")):
        raise RuntimeError("independent sampling/weight contract mismatch")
    if {key for key,value in program["authorization"].items() if value}!={"implementation_creation","unit_static_tests"} \
            or any(type(value) is not bool for value in program["authorization"].values()):
        raise RuntimeError("independent program authorization mismatch")
    forbidden={"execution without exact committed grant/upstream/lineage/runtime pins",
        "manual execution on syntax or syn101","release/reacquire under one grant",
        "retry/resubmit/replacement/seed mutation","importing JAX or PMWD at module import",
        "persisting fields/particles/IC/checkpoints","partial canonical output or overwrite",
        "ranking/promotion/threshold change/downstream work","commit/push/Slurm/GPFS in implementation phase",
        "scripts/tripwire modification"}
    if set(program["forbidden"])!=forbidden or program["diagnostics"].get("component_count")!=17 \
            or program["diagnostics"].get("fixed_denominators_only") is not True \
            or program["diagnostics"].get("ranking_or_identity_extrema") is not False \
            or program["field_integrity"].get("required_each")!=768 \
            or program["field_integrity"].get("required_cross_stage_unique")!=2304:
        raise RuntimeError("independent forbidden/diagnostic/field contract mismatch")


def verify_seal_contract_values(manifest: Mapping[str,Any],complete: Mapping[str,Any]) -> None:
    if manifest.get("schema")!="ouruniv-cf4-lg-unconstrained-p1-reference-seal-manifest-v1" \
            or manifest.get("status")!="complete" \
            or complete.get("schema")!="ouruniv-cf4-lg-unconstrained-p1-reference-complete-v1" \
            or complete.get("status")!="complete" or complete.get("scientific_result")!="descriptive_only" \
            or type(complete.get("automatic_promotion")) is not bool \
            or complete["automatic_promotion"] is not False:
        raise RuntimeError("seal/COMPLETE schema/status/science values mismatch")
    if not isinstance(manifest.get("files"),list) or len(manifest["files"])!=3:
        raise RuntimeError("payload manifest list mismatch")
    for item in manifest["files"]:
        if set(item)!={"name","size","sha256","mode"} or not isinstance(item["name"],str) \
                or type(item["size"]) is not int or item["size"]<0 or item["mode"]!="0444" \
                or not isinstance(item["sha256"],str) or len(item["sha256"])!=64 \
                or any(char not in "0123456789abcdef" for char in item["sha256"]):
            raise RuntimeError("payload manifest item key/type/value mismatch")


def verify_summary_exact(actual: Mapping[str,Any],expected: Mapping[str,Any]) -> None:
    def same(left: Any,right: Any) -> bool:
        if type(left) is not type(right):return False
        if isinstance(left,dict):return set(left)==set(right) and all(same(left[key],right[key]) for key in left)
        if isinstance(left,list):return len(left)==len(right) and all(same(a,b) for a,b in zip(left,right))
        return left==right
    if not same(actual,expected): raise RuntimeError("independent summary mismatch")


def verify_reference_rows(rows: Sequence[Sequence[Any]], domain_tag: str, design_commit: str,
                          design_sha256: str, expected_count: int) -> None:
    if len(rows) != expected_count or [row[0] for row in rows] != list(range(expected_count)):
        raise RuntimeError("reference formula index coverage mismatch")
    digests=[]; seeds=[]; keys=[]
    for index, digest, seed, word0, word1 in rows:
        preimage=(domain_tag.encode()+b"\0"+design_commit.encode()+b"\0"+
                  design_sha256.encode()+b"\0"+int(index).to_bytes(8,"big"))
        calculated=hashlib.sha256(preimage).digest()
        expected=[calculated.hex(),int.from_bytes(calculated[:8],"big"),
                  int.from_bytes(calculated[:4],"big"),int.from_bytes(calculated[4:8],"big")]
        if [digest,seed,word0,word1] != expected: raise RuntimeError("reference formula/decode mismatch")
        digests.append(digest);seeds.append(seed);keys.append((word0,word1))
    if len(set(digests))!=expected_count or len(set(seeds))!=expected_count or len(set(keys))!=expected_count:
        raise RuntimeError("reference internal collision")


def decode_forbidden_occurrences(seed_manifest: Mapping[str, Any]) -> list[list[Any]]:
    record=seed_manifest["forbidden_inventory"]["lossless_occurrence_record"]
    compressed=base64.b64decode(record["base64"],validate=True)
    if len(compressed)!=record["compressed_size"] or hashlib.sha256(compressed).hexdigest()!=record["compressed_sha256"]:
        raise RuntimeError("forbidden compressed provenance mismatch")
    raw=zlib.decompress(compressed)
    if len(raw)!=record["uncompressed_size"] or hashlib.sha256(raw).hexdigest()!=record["uncompressed_sha256"]:
        raise RuntimeError("forbidden provenance payload mismatch")
    occurrences=json.loads(raw)
    if len(occurrences)!=seed_manifest["forbidden_inventory"]["occurrence_count"]:
        raise RuntimeError("forbidden occurrence count mismatch")
    allowed={"range_inclusive","range_python","start_plus_count","typed_integer_leaf",
             "typed_integer_array","explicit_selector","explicit_npz_selector"}
    prior_source=-1
    for row in occurrences:
        if not isinstance(row,list) or len(row)!=5 or isinstance(row[0],bool) \
                or not isinstance(row[0],int) or not 0<=row[0]<106 \
                or not isinstance(row[1],str) or not row[1] \
                or _uint64(row[2],row[1])!=row[2] or row[3] not in allowed \
                or (row[4] is not None and (isinstance(row[4],bool) or not isinstance(row[4],int)
                                             or row[4]<0)) \
                or (row[3] in {"range_inclusive","range_python","start_plus_count"}
                    and type(row[4]) is not int) \
                or (row[3] not in {"range_inclusive","range_python","start_plus_count"}
                    and row[4] is not None) \
                or (row[3]=="explicit_npz_selector" and row[1].startswith("/")) \
                or (row[3]!="explicit_npz_selector" and not row[1].startswith("/")) \
                or row[0]<prior_source:
            raise RuntimeError("forbidden occurrence row structure mismatch")
        prior_source=row[0]
    return occurrences


def verify_seed_manifest_integrity(seed: Mapping[str, Any], program: Mapping[str, Any],
                                   verify_external_sources: bool) -> None:
    seed_path=ROOT/program["inputs"]["seed_manifest"]["path"]
    if set(seed)!=SEED_MANIFEST_KEYS or seed_path.read_bytes()!=_json_bytes(seed):
        raise RuntimeError("seed manifest keyset/canonical bytes mismatch")
    design_spec=program["inputs"]["design"]
    verify_reference_rows(seed["seed_derivation"]["rows"],seed["seed_derivation"]["contract"]["domain_tag_utf8"],
                          design_spec["commit"],design_spec["sha256"],768)
    occurrences=decode_forbidden_occurrences(seed); values=sorted({int(row[2]) for row in occurrences})
    forbidden=seed["forbidden_inventory"]
    if values!=forbidden["unique_seed_uint64"] or len(values)!=forbidden["unique_seed_count"]:
        raise RuntimeError("forbidden unique inventory mismatch")
    seed_digest=hashlib.sha256(b"".join(x.to_bytes(8,"big") for x in values)).hexdigest()
    legacy_jax_values=[x for x in values if x<2**32]
    jax_digest=hashlib.sha256(b"".join((0).to_bytes(4,"big")+x.to_bytes(4,"big")
                                      for x in legacy_jax_values)).hexdigest()
    duplicates=[[value,count] for value,count in sorted(Counter(int(row[2]) for row in occurrences).items())
                if count>1]
    if seed_digest!=forbidden["seed_sha256"] or jax_digest!=forbidden["jax_sha256"] \
            or forbidden["unique_jax_key_count"]!=len(legacy_jax_values) \
            or forbidden["explicit_jax_occurrence_count"]!=0 \
            or forbidden["duplicate_diagnostics"]!=duplicates:
        raise RuntimeError("forbidden inventory encoding digest mismatch")
    rows=seed["seed_derivation"]["rows"]; ref=seed["reference_inventory"]
    sorted_seeds=sorted(row[2] for row in rows);sorted_keys=sorted([row[3],row[4]] for row in rows)
    if sorted_seeds!=ref["sorted_seed_uint64"] or sorted_keys!=ref["sorted_jax_key_words"]:
        raise RuntimeError("reference sorted inventory mismatch")
    rs=hashlib.sha256(b"".join(x.to_bytes(8,"big") for x in sorted_seeds)).hexdigest()
    rk=hashlib.sha256(b"".join(x[0].to_bytes(4,"big")+x[1].to_bytes(4,"big") for x in sorted_keys)).hexdigest()
    full=hashlib.sha256(b"".join(sorted(bytes.fromhex(row[1]) for row in rows))).hexdigest()
    if rs!=ref["seed_sha256"] or rk!=ref["jax_sha256"] \
            or full!=ref["sorted_full_digest_bytes_sha256"] \
            or ref["full_digest_count"]!=768 or ref["seed_count"]!=768 or ref["jax_count"]!=768:
        raise RuntimeError("reference inventory digest mismatch")
    if set(sorted_seeds)&set(values) or set(map(tuple,sorted_keys))&{(0,x) for x in values} \
            or seed["integrity"]["forbidden_seed_intersection"] or seed["integrity"]["forbidden_jax_intersection"]:
        raise RuntimeError("reference/forbidden intersection mismatch")
    source_rows=seed["sources"]["rows"]
    if len(source_rows)!=106 or [row[0] for row in source_rows]!=list(range(106)):
        raise RuntimeError("source provenance row coverage mismatch")
    per_source=Counter(row[0] for row in occurrences)
    for row in source_rows:
        if not isinstance(row,list) or len(row)!=10 or row[1] not in {"git","external"} \
                or not isinstance(row[2],str) or not isinstance(row[3],str) or len(row[3])!=64 \
                or any(char not in "0123456789abcdef" for char in row[3]) \
                or not isinstance(row[4],str) or type(row[6]) is not int or type(row[7]) is not int \
                or type(row[8]) is not int or min(row[6],row[7],row[8])<0 \
                or row[7]!=per_source[row[0]]:
            raise RuntimeError("source provenance row structure/count mismatch")
    if verify_external_sources:
        parent=seed["sources"]["parent_git_commit"]; rows_source=source_rows
        if len(rows_source)!=106 or sum(row[1]=="git" for row in rows_source)!=97 \
                or sum(row[1]=="external" for row in rows_source)!=9 \
                or [row[0] for row in rows_source]!=list(range(106)):
            raise RuntimeError("exact source row count mismatch")
        erratum=json.loads((ROOT/program["inputs"]["erratum_v1"]["path"]).read_bytes())
        corrected=erratum["corrected_forbidden_seed_source_contract"]
        tracked=corrected["tracked_JSON_sources"]
        tracked_digest=_object_digest(tracked)
        if tracked_digest!=seed["sources"]["tracked_list_sha256"] \
                or tracked_digest!=corrected["source_scope"]["tracked_JSON_source_list_canonical_sha256"]:
            raise RuntimeError("tracked source list digest mismatch")
        candidate_names=subprocess.check_output(
            ["git","-C",str(ROOT),"ls-tree","-r","--name-only",parent,"--","config"],
            text=True).splitlines()
        candidate_names=[path for path in candidate_names if Path(path).parent==Path("config")
                         and Path(path).name.endswith(".json")
                         and Path(path).name.startswith(("cf4_","p1_","p2_","v3_"))]
        candidate_blobs={path:subprocess.check_output(
            ["git","-C",str(ROOT),"show",f"{parent}:{path}"]) for path in candidate_names}
        discovered=discover_seed_json_blobs(candidate_blobs)
        require_exact_discovery(discovered,[item["path"] for item in tracked])
        for position,item in enumerate(tracked):
            source_row=rows_source[position]
            if source_row[1]!="git" or source_row[2]!=item["path"] or source_row[3]!=item["sha256"] \
                    or source_row[6]!=item["typed_seed_integer_leaf_count_before_range_expansion"]:
                raise RuntimeError("tracked source authority mismatch")
        externals=corrected["external_sealed_sources"]
        for position,item in enumerate(externals,97):
            source_row=rows_source[position]
            if source_row[1]!="external" or source_row[2]!=item["path"] \
                    or source_row[3]!=item["sha256"] or source_row[9]!=item["selectors"]:
                raise RuntimeError("external source authority mismatch")
        recomputed_occurrences=[];recomputed_explicit_jax=[]
        for index,kind,path,sha,mode,*_rest in rows_source:
            if kind=="git":
                data=candidate_blobs[path]
                tree=subprocess.check_output(["git","-C",str(ROOT),"ls-tree",parent,"--",path],text=True).split()
                if not tree or tree[0]!=mode: raise RuntimeError(f"Git source mode mismatch: {path}")
            elif kind=="external":
                data=_read_regular_nofollow(Path(path),mode)
            else: raise RuntimeError("unknown source class")
            if hashlib.sha256(data).hexdigest()!=sha: raise RuntimeError(f"source SHA mismatch: {path}")
            decoded=None
            if kind=="git" or (kind=="external" and path.endswith(".json")):
                decoded=strict_json_bytes(data) if kind=="git" else json.loads(data)
                if decoded.get("schema")!=rows_source[index][5]:
                    raise RuntimeError(f"source schema mismatch: {path}")
            if kind=="git" or (kind=="external" and rows_source[index][9]==["recursive typed extraction rule"]):
                extracted,raw_count,jax_pairs=typed_json_occurrences(decoded,index)
            elif kind=="external" and path.endswith(".json"):
                extracted,raw_count=explicit_json_selector_occurrences(
                    decoded,index,rows_source[index][9]);jax_pairs=[]
            elif kind=="external" and path.endswith(".npz"):
                if rows_source[index][9]!=seed["sources"]["schedule_selectors"] \
                        or "keys[:,:]" not in seed["sources"]["schedule_keys_exclusion"]:
                    raise RuntimeError("schedule selector/exclusion contract mismatch")
                extracted,raw_count=explicit_npz_selector_occurrences(
                    data,index,rows_source[index][9],{"keys"});jax_pairs=[]
            else: raise RuntimeError("unsupported exact source format")
            if raw_count!=rows_source[index][6] or len(extracted)!=rows_source[index][7] \
                    or len(jax_pairs)!=rows_source[index][8]:
                raise RuntimeError(f"typed extraction source counts mismatch: {path}")
            recomputed_occurrences.extend(extracted);recomputed_explicit_jax.extend(jax_pairs)
        require_exact_occurrences(recomputed_occurrences,occurrences)
        forbidden_jax=sorted({(0,number) for number in values if number<2**32}
                             | {tuple(pair) for pair in recomputed_explicit_jax})
        recomputed_jax_digest=hashlib.sha256(b"".join(
            left.to_bytes(4,"big")+right.to_bytes(4,"big") for left,right in forbidden_jax)).hexdigest()
        if len(recomputed_explicit_jax)!=forbidden["explicit_jax_occurrence_count"] \
                or len(forbidden_jax)!=forbidden["unique_jax_key_count"] \
                or recomputed_jax_digest!=forbidden["jax_sha256"]:
            raise RuntimeError("independently extracted forbidden JAX inventory mismatch")


def independent_program_grant_preflight(program: Mapping[str, Any], grant: Mapping[str, Any],
                                        grant_commit: str) -> None:
    """Validate immutable provenance from Git objects, not the current runtime."""
    contract = program["grant_contract"]
    if program.get("schema") != "ouruniv-cf4-lg-unconstrained-p1-reference-program-v1" \
            or program.get("status") != "implementation_frozen_execution_unauthorized_waiting_exact_one_grant":
        raise RuntimeError("program schema/status mismatch")
    program_path=ROOT/"config/cf4_lg_unconstrained_p1_reference_program_v1.json"
    if set(program)!=PROGRAM_KEYS or program_path.read_bytes()!=_json_bytes(program):
        raise RuntimeError("program canonical bytes/keyset mismatch")
    for spec in program["inputs"].values():
        source=Path(spec["path"]);source=source if source.is_absolute() else ROOT/source
        if _file_digest(source)!=spec["sha256"]: raise RuntimeError("independent program input SHA mismatch")
    design=strict_json_bytes((ROOT/program["inputs"]["design"]["path"]).read_bytes())
    independently_validate_critical_contracts(program,design)
    if contract.get("schema")!="ouruniv-cf4-lg-unconstrained-p1-reference-execution-grant-v1" \
            or contract.get("status")!="authorized_one_live_held_allocation_waiting_worker_activation" \
            or len(contract["exact_top_level_keys"])!=len(GRANT_KEYS) \
            or set(contract["exact_top_level_keys"])!=GRANT_KEYS or set(grant)!=GRANT_KEYS \
            or grant.get("schema") != \
            "ouruniv-cf4-lg-unconstrained-p1-reference-execution-grant-v1" \
            or grant.get("status")!="authorized_one_live_held_allocation_waiting_worker_activation":
        raise RuntimeError("grant schema/status/keyset mismatch")
    grant_path=ROOT/program["lineage"]["future_grant_path"]
    if grant_path.read_bytes()!=_json_bytes(grant): raise RuntimeError("grant noncanonical bytes")
    yes=GRANT_AUTH_TRUE;no=GRANT_AUTH_FALSE
    if len(contract["authorization_required_true"])!=len(yes) \
            or set(contract["authorization_required_true"])!=yes \
            or len(contract["authorization_required_false"])!=len(no) \
            or set(contract["authorization_required_false"])!=no:
        raise RuntimeError("program grant authorization contract changed")
    auth = grant["authorization"]
    if set(auth) != yes | no or any(auth[k] is not True for k in yes) or any(auth[k] is not False for k in no):
        raise RuntimeError("grant authorization mismatch")
    if grant["program"] != {"path":"config/cf4_lg_unconstrained_p1_reference_program_v1.json",
                            "sha256":_file_digest(ROOT/"config/cf4_lg_unconstrained_p1_reference_program_v1.json")}:
        raise RuntimeError("grant program pin mismatch")
    seed_spec=program["inputs"]["seed_manifest"]
    if grant["seed_manifest"] != {"path":seed_spec["path"],"sha256":seed_spec["sha256"],"commit":seed_spec["commit"]}:
        raise RuntimeError("grant seed pin mismatch")
    if grant["runtime_pins"] != grant["allocation_receipt"].get("runtime_pins") \
            or grant["allocation_receipt"].get("one_live_held_allocation") is not True:
        raise RuntimeError("grant allocation/runtime binding mismatch")
    repository_pin=grant["runtime_pins"].get("repository_implementation")
    if not isinstance(repository_pin,Mapping) or repository_pin.get("commit")!=grant["implementation"]["commit"] \
            or repository_pin.get("files")!=grant["implementation"]["files"]:
        raise RuntimeError("grant receipt implementation identity mismatch")
    if grant["outputs"]!=program["outputs"]: raise RuntimeError("grant output contract mismatch")
    git=lambda *a:subprocess.run(["git","-C",str(ROOT),*a],check=True,stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,text=True).stdout.strip()
    impl=grant["implementation"]["commit"]
    if not isinstance(grant_commit,str) or len(grant_commit)!=40 \
            or any(char not in "0123456789abcdef" for char in grant_commit) \
            or git("cat-file","-t",grant_commit)!="commit" \
            or git("rev-list","--parents","-n","1",grant_commit).split()[1:]!=[impl] \
            or git("rev-list","--parents","-n","1",impl).split()[1:]!=[program["lineage"]["required_parent_commit"]]:
        raise RuntimeError("independent immutable Git lineage mismatch")
    paths=program["lineage"]["implementation_exact_added_paths"]
    def rows(a,b):
        raw=git("diff","--no-renames","--name-status",a,b,"--")
        return [tuple(line.split("\t")) for line in raw.splitlines() if line]
    if sorted(rows(program["lineage"]["required_parent_commit"],impl))!=sorted(("A",p) for p in paths) \
            or rows(impl,grant_commit)!=[("A",program["lineage"]["future_grant_path"])]:
        raise RuntimeError("independent exact-six/exact-one diff mismatch")
    committed_grant=subprocess.check_output(
        ["git","-C",str(ROOT),"show",f"{grant_commit}:{program['lineage']['future_grant_path']}"])
    if committed_grant!=grant_path.read_bytes():
        raise RuntimeError("committed grant bytes differ from canonical grant")
    bound={x["path"]:(x["mode"],x["sha256"]) for x in grant["implementation"]["files"]}
    tree_modes={};tree_hashes={}
    for path in paths:
        entry=git("ls-tree",impl,"--",path).split()
        if len(entry)<4 or entry[3]!=path: raise RuntimeError("independent implementation tree entry absent")
        tree_modes[path]=entry[0]
        blob=subprocess.check_output(["git","-C",str(ROOT),"show",f"{impl}:{path}"])
        tree_hashes[path]=hashlib.sha256(blob).hexdigest()
    if set(bound)!=set(paths) or any(tree_modes[p]!="100644" \
            or bound[p] != ("100644",tree_hashes[p]) for p in paths):
        raise RuntimeError("independent implementation file pin mismatch")


def checker_frame_live_array(array: np.ndarray, domain_tag: str,
                             expected_shape: tuple[int, ...] = (192, 192, 192)) -> str:
    """Independent framing implementation; no producer constant/helper is used."""
    if type(array) is not np.ndarray or array.dtype.str != "<f4" \
            or not array.flags["C_CONTIGUOUS"] or tuple(array.shape) != tuple(expected_shape):
        raise RuntimeError("checker received wrong field dtype/order/shape")
    if np.logical_not(np.isfinite(array)).any():
        raise RuntimeError("checker received nonfinite field")
    header = bytearray()
    header.extend(domain_tag.encode("utf-8")); header.append(0)
    header.extend(b"field-frame-v1"); header.append(0)
    header.extend(b"<f4"); header.append(0)
    header.extend(int(len(expected_shape)).to_bytes(4, byteorder="big", signed=False))
    for dimension in expected_shape:
        header.extend(int(dimension).to_bytes(8, byteorder="big", signed=False))
    payload = memoryview(array).cast("B").tobytes()
    header.extend(int(len(payload)).to_bytes(8, byteorder="big", signed=False))
    check = hashlib.sha256(); check.update(header); check.update(payload)
    return check.hexdigest()


def _at(score: Mapping[str, Any], dotted: str) -> float:
    item: Any = score
    pieces = dotted.split("."); position = 0
    while position < len(pieces):
        if not isinstance(item, Mapping): raise KeyError(dotted)
        matches = [(end, ".".join(pieces[position:end]))
                   for end in range(position + 1, len(pieces) + 1)
                   if ".".join(pieces[position:end]) in item]
        if not matches: raise KeyError(dotted)
        end, key = max(matches, key=lambda pair: pair[0])
        item = item[key]; position = end
    if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
        raise RuntimeError(f"invalid primitive score field {dotted}")
    return float(item)


def independent_derive(score: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    margins: dict[str, float] = {}
    passes: dict[str, bool] = {}
    grouped: dict[str, list[float]] = {name: [] for name in GATE_ORDER}
    for item in contract["components"]:
        value = _at(score, item["value_path"]); threshold = float(item["threshold"])
        denominator = float(item["denominator"])
        if denominator <= 0 or not math.isfinite(denominator):
            raise RuntimeError("bad frozen denominator")
        if item["comparison"] in ("GE", "GT"):
            margin = (value - threshold) / denominator
        elif item["comparison"] in ("LE", "LT"):
            margin = (threshold - value) / denominator
        else:
            raise RuntimeError("unknown comparison")
        passed = margin >= 0 if item["comparison"] in ("GE", "LE") else margin > 0
        margins[item["id"]] = float(margin); passes[item["id"]] = bool(passed)
        grouped[item["gate"]].append(float(margin))
    gate_margins = {name: min(grouped[name]) for name in GATE_ORDER}
    gates = {name: all(passes[item["id"]] for item in contract["components"]
                       if item["gate"] == name) for name in GATE_ORDER}
    if not isinstance(score.get("gates"), Mapping) or set(score["gates"]) != set(GATE_ORDER) \
            or any(type(score["gates"].get(g)) is not bool for g in GATE_ORDER) \
            or {g: score["gates"][g] for g in GATE_ORDER} != gates:
        raise RuntimeError("score gate forgery/mismatch")
    if type(score.get("pass")) is not bool or score["pass"] != all(gates.values()):
        raise RuntimeError("score all-five forgery/mismatch")
    if type(score.get("n_gates_passed")) is not int \
            or score["n_gates_passed"] != sum(gates.values()):
        raise RuntimeError("score gate-count forgery/mismatch")
    failed_components = [name for name in contract["component_order"] if not passes[name]]
    failed_gates = [name for name in GATE_ORDER if not gates[name]]
    exact_four = len(failed_gates) == 1
    return {"component_margins": margins, "gate_margins": gate_margins,
            "maximin_margin": min(gate_margins.values()), "gates": gates,
            "all_five_pass": all(gates.values()), "component_failure_set": failed_components,
            "gate_failure_set": failed_gates, "exactly_four_of_five": exact_four,
            "sole_failed_gate": failed_gates[0] if exact_four else None}


def verify_receipts(row: Mapping[str, Any], specs: Sequence[Mapping[str, Any]]) -> None:
    receipts = row.get("streaming_receipts")
    if not isinstance(receipts, list) or len(receipts) != 3:
        raise RuntimeError("missing streaming receipts")
    by_stage = {r.get("stage"): r for r in receipts}
    if set(by_stage) != {x["stage"] for x in specs}:
        raise RuntimeError("swapped/missing stage receipt")
    for spec in specs:
        receipt = by_stage[spec["stage"]]
        receipt_keys = {"stage", "field_name", "domain_tag", "frame_version", "dtype", "shape",
                        "reference_index", "seed_digest_sha256", "producer_sha256",
                        "independent_checker_sha256", "pass"}
        if spec["stage"] == "smoothed_delta_scorer_input": receipt_keys.add("post_score_sha256")
        if set(receipt) != receipt_keys: raise RuntimeError("receipt exact keyset mismatch")
        expected = {"field_name": spec["field_name"], "domain_tag": spec["domain_tag"],
                    "frame_version": "field-frame-v1", "dtype": "<f4",
                    "shape": [192, 192, 192], "reference_index": row["reference_index"],
                    "seed_digest_sha256": row["seed_digest_sha256"], "pass": True}
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise RuntimeError("receipt stage/domain/frame/seed binding mismatch")
        if type(receipt.get("pass")) is not bool or type(receipt.get("reference_index")) is not int:
            raise RuntimeError("receipt pass/index exact type mismatch")
        digest = row[spec["field_name"]]
        if not isinstance(digest, str) or len(digest) != 64 \
                or any(ch not in "0123456789abcdef" for ch in digest):
            raise RuntimeError("malformed stage SHA256")
        if receipt.get("producer_sha256") != digest \
                or receipt.get("independent_checker_sha256") != digest:
            raise RuntimeError("producer/checker/row hash mismatch")
        if spec["stage"] == "smoothed_delta_scorer_input" \
                and receipt.get("post_score_sha256") != digest:
            raise RuntimeError("post-score field hash mismatch")


def verify_member(row: Mapping[str, Any], seed_row: Sequence[Any],
                  margin_contract: Mapping[str, Any], field_specs: Sequence[Mapping[str, Any]],
                  runtime_binding: Mapping[str, Any] | None = None) -> None:
    if row.get("schema") != "ouruniv-cf4-lg-unconstrained-p1-reference-member-metric-row-v1":
        raise RuntimeError("wrong member schema")
    exact_keys = {"schema","reference_index","batch_index","within_batch_index","seed_digest_sha256",
                  "seed_uint64","jax_key_words","initial_white_field_sha256",
                  "unsmoothed_z0_cic_density_sha256","smoothed_delta_scorer_input_sha256",
                  "streaming_receipts","score_member","component_margins","gate_margins",
                  "maximin_margin","gates","all_five_pass","component_failure_set",
                  "gate_failure_set","exactly_four_of_five","sole_failed_gate","pareto"}
    exact_keys |= {"seed_manifest_sha256","grant_sha256","grant_commit",
                   "runtime_pins_sha256","implementation_commit"}
    if set(row) != exact_keys: raise RuntimeError("member exact keyset/forbidden artifact mismatch")
    expected = {"reference_index": seed_row[0], "seed_digest_sha256": seed_row[1],
                "seed_uint64": seed_row[2], "jax_key_words": [seed_row[3], seed_row[4]],
                "batch_index": seed_row[0] // 16, "within_batch_index": seed_row[0] % 16}
    if any(row.get(k) != v for k, v in expected.items()):
        raise RuntimeError("seed/reference/batch correlated forgery")
    if any(type(row.get(k)) is not int for k in ("reference_index","batch_index",
                                                  "within_batch_index","seed_uint64")) \
            or not isinstance(row.get("jax_key_words"),list) \
            or any(type(value) is not int for value in row["jax_key_words"]):
        raise RuntimeError("member integer field type forgery")
    if not isinstance(row.get("gates"),Mapping) or any(type(row["gates"].get(g)) is not bool
                                                        for g in GATE_ORDER) \
            or any(type(row.get(name)) is not bool for name in
                   ("all_five_pass","exactly_four_of_five")):
        raise RuntimeError("member Boolean field type forgery")
    if type(row.get("maximin_margin")) is not float \
            or not isinstance(row.get("component_margins"),Mapping) \
            or any(type(value) is not float for value in row["component_margins"].values()) \
            or not isinstance(row.get("gate_margins"),Mapping) \
            or any(type(value) is not float for value in row["gate_margins"].values()) \
            or not isinstance(row.get("pareto"),Mapping) \
            or type(row["pareto"].get("dominates_count")) is not int \
            or type(row["pareto"].get("dominated_by_count")) is not int \
            or type(row["pareto"].get("nondominated")) is not bool:
        raise RuntimeError("member derived metric exact type forgery")
    if runtime_binding is not None and any(row.get(k) != v for k, v in runtime_binding.items()):
        raise RuntimeError("member runtime/grant/manifest binding mismatch")
    verify_receipts(row, field_specs)
    derived = independent_derive(row["score_member"], margin_contract)
    for key, value in derived.items():
        if row.get(key) != value:
            raise RuntimeError(f"derived member field forgery: {key}")


def _ecdf2(values: Sequence[float]) -> list[dict[str, Any]]:
    data = sorted(float(x) for x in values)
    if not data or any(not math.isfinite(x) for x in data):
        raise RuntimeError("nonfinite ECDF")
    answer = []; cursor = cumulative = 0
    while cursor < len(data):
        stop = cursor + 1
        while stop < len(data) and data[stop] == data[cursor]: stop += 1
        cumulative += stop - cursor
        answer.append({"x": data[cursor], "count": stop - cursor,
                       "cumulative_count": cumulative, "F_n": cumulative / len(data)})
        cursor = stop
    return answer


def _interval2(k: int, n: int) -> list[float]:
    from scipy.stats import beta
    return [0.0 if k == 0 else float(beta.ppf(.025, k, n - k + 1)),
            1.0 if k == n else float(beta.ppf(.975, k + 1, n - k))]


def _checker_bernoulli(k: int, n: int) -> dict[str, Any]:
    return {"count": k, "rate": k / n, "clopper_pearson_95": _interval2(k, n),
            "one_sided_zero_u95": 1 - .05 ** (1 / n) if k == 0 else None}


def _checker_numeric(value: Any, prefix: str = "") -> dict[str, float]:
    answer = {}
    if isinstance(value, bool) or value is None: return answer
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number): raise RuntimeError(f"nonfinite metric {prefix}")
        answer[prefix] = number
    elif isinstance(value, Mapping):
        for name in sorted(value):
            answer.update(_checker_numeric(value[name], f"{prefix}.{name}" if prefix else str(name)))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value): answer.update(_checker_numeric(item, f"{prefix}[{index}]"))
    return answer


def _checker_metric_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tables = [_checker_numeric(row["score_member"]) for row in rows]; names = sorted(tables[0])
    if any(sorted(table) != names for table in tables): raise RuntimeError("metric schema mismatch")
    result = {}
    for name in names:
        ordered = sorted(table[name] for table in tables)
        pick = lambda q: ordered[math.ceil(q * len(ordered)) - 1]
        result[name] = {"count": len(ordered), "min": ordered[0], "max": ordered[-1],
                        "mean": math.fsum(table[name] for table in tables) / len(ordered),
                        "quantile_type1": {"0.05": pick(.05), "0.50": pick(.50), "0.95": pick(.95)}}
    return result


def _independent_pareto(rows: list[dict[str, Any]], order: Sequence[str]) -> None:
    vectors = [tuple(float(row["component_margins"][name]) for name in order) for row in rows]
    for i, row in enumerate(rows):
        dom = by = 0
        for j in range(len(vectors)):
            if i == j: continue
            if all(vectors[i][k] >= vectors[j][k] for k in range(len(order))) \
                    and any(vectors[i][k] > vectors[j][k] for k in range(len(order))): dom += 1
            if all(vectors[j][k] >= vectors[i][k] for k in range(len(order))) \
                    and any(vectors[j][k] > vectors[i][k] for k in range(len(order))): by += 1
        row["pareto"] = {"dominates_count": dom, "dominated_by_count": by, "nondominated": by == 0}


def independent_stage_sets(rows: Sequence[Mapping[str, Any]], n: int) -> dict[str, Any]:
    fields = ("initial_white_field_sha256", "unsmoothed_z0_cic_density_sha256",
              "smoothed_delta_scorer_input_sha256")
    sets = []
    for field in fields:
        values = [row[field] for row in rows]
        if len(values) != n or len(set(values)) != n: raise RuntimeError("stage duplicate/missing")
        sets.append(set(values))
    if len(sets[0] | sets[1] | sets[2]) != 3 * n: raise RuntimeError("cross-stage confusion")
    return {"stage_counts": {f: n for f in fields}, "stage_unique_counts": {f: n for f in fields},
            "stage_missing_counts": {f: 0 for f in fields},
            "stage_duplicate_counts": {f: 0 for f in fields},
            "cross_stage_union_unique_count": 3 * n,
            "cross_stage_confusion_count": 0, "pass": True}


def independent_summary(rows: list[dict[str, Any]], component_order: Sequence[str],
                        forbidden_intersections: Mapping[str, Sequence[Any]]) -> dict[str, Any]:
    n = len(rows); _independent_pareto(rows, component_order)
    seed_intersection=list(forbidden_intersections.get("seed_uint64",[]))
    jax_intersection=list(forbidden_intersections.get("jax_key_words",[]))
    if seed_intersection or jax_intersection: raise RuntimeError("forbidden inventory intersection is nonempty")
    gc = {g: sum(row["gates"][g] is True for row in rows) for g in GATE_ORDER}
    ac = sum(row["all_five_pass"] is True for row in rows)
    pats = {format(i, "05b"): 0 for i in range(32)}
    for row in rows: pats["".join(str(int(row["gates"][g])) for g in GATE_ORDER)] += 1
    four = {g: sum(row["sole_failed_gate"] == g for row in rows) for g in GATE_ORDER}
    gf, cf = Counter(), Counter()
    for row in rows:
        gf[json.dumps(row["gate_failure_set"], separators=(",", ":"))] += 1
        cf[json.dumps(row["component_failure_set"], separators=(",", ":"))] += 1
    co=[]; unions=[]; jacc=[]
    for ga in GATE_ORDER:
        cr=[];ur=[];jr=[]
        for gb in GATE_ORDER:
            aa=[not r["gates"][ga] for r in rows];bb=[not r["gates"][gb] for r in rows]
            inter=sum(a and b for a,b in zip(aa,bb));union=sum(a or b for a,b in zip(aa,bb))
            cr.append(inter);ur.append(union);jr.append(inter/union if union else 1.0)
        co.append(cr);unions.append(ur);jacc.append(jr)
    hist=lambda it:{str(k):v for k,v in sorted(Counter(it).items())}
    return {"schema":"ouruniv-cf4-lg-unconstrained-p1-reference-summary-v1",
            "status":"complete_descriptive_reference_no_downstream_authorization",
            "N_expected":768,"N_complete":n,"equal_member_weight":1/768,
            "gate_pass":{g:_checker_bernoulli(gc[g],n) for g in GATE_ORDER},
            "all_five":{"count":ac,"rate":ac/n,"clopper_pearson_95":_interval2(ac,n),"one_sided_u95":1-.05**(1/n) if ac==0 else None},
            "patterns":{k:_checker_bernoulli(v,n) for k,v in pats.items()},
            "exact_four_of_five":{**_checker_bernoulli(sum(four.values()),n),"sole_failed_gate":{g:_checker_bernoulli(four[g],n) for g in GATE_ORDER}},
            "ECDF":{"components":{k:_ecdf2([r["component_margins"][k] for r in rows]) for k in component_order},
                    "gates":{g:_ecdf2([r["gate_margins"][g] for r in rows]) for g in GATE_ORDER},
                    "maximin":_ecdf2([r["maximin_margin"] for r in rows])},
            "gate_failure_sets":{k:_checker_bernoulli(v,n) for k,v in gf.items()},
            "component_failure_sets":{k:_checker_bernoulli(v,n) for k,v in cf.items()},
            "cofailure":{"gate_order":list(GATE_ORDER),"count":co,"rate":[[x/n for x in z] for z in co],"union_count":unions,"jaccard":jacc},
            "pareto":{"nondominated_count":sum(r["pareto"]["nondominated"] for r in rows),
                      "nondominated_fraction":sum(r["pareto"]["nondominated"] for r in rows)/n,
                      "dominates_count_histogram":hist(r["pareto"]["dominates_count"] for r in rows),
                      "dominated_by_count_histogram":hist(r["pareto"]["dominated_by_count"] for r in rows)},
            "score_member_scalar_metrics":_checker_metric_summary(rows),
            "stage_hash_integrity":independent_stage_sets(rows,n),
            "forbidden_inventory_disjointness":{"seed_uint64_intersection_count":len(seed_intersection),
                "jax_key_words_intersection_count":len(jax_intersection),
                "exact_empty_intersections":True,"pass":True},"automatic_promotion":False,
            "threshold_change":False,"downstream_execution":False}


def check_output_directory(directory: Path, program: Mapping[str, Any],
                           grant: Mapping[str, Any], private: bool = False) -> dict[str, Any]:
    directory = Path(directory); mode = stat.S_IMODE(os.lstat(directory).st_mode)
    if mode != (0o700 if private else 0o555): raise RuntimeError("output directory mode mismatch")
    if {p.name for p in directory.iterdir()} != ENTRY_SET: raise RuntimeError("output entry set mismatch")
    for name in ENTRY_SET:
        p=directory/name
        if p.is_symlink() or not p.is_file() or stat.S_IMODE(os.lstat(p).st_mode)!=0o444:
            raise RuntimeError("output type/mode mismatch")
    manifest=json.loads((directory/"manifest.json").read_bytes());complete=json.loads((directory/"COMPLETE").read_bytes())
    if set(manifest)!={"schema","status","grant_sha256","grant_commit","files","exact_entry_set"} \
            or set(complete)!={"schema","status","manifest_sha256","scientific_result","automatic_promotion"}:
        raise RuntimeError("seal/COMPLETE exact keyset mismatch")
    grant_commit=manifest.get("grant_commit")
    independent_program_grant_preflight(program, grant, grant_commit)
    verify_seal_contract_values(manifest,complete)
    if manifest.get("exact_entry_set")!=sorted(ENTRY_SET) or complete.get("manifest_sha256")!=_file_digest(directory/"manifest.json"):
        raise RuntimeError("seal manifest/COMPLETE mismatch")
    grant_path=ROOT/program["lineage"]["future_grant_path"]
    grant_file_sha=_file_digest(grant_path)
    if manifest.get("grant_sha256")!=grant_file_sha or manifest.get("grant_commit")!=grant_commit:
        raise RuntimeError("seal grant file/commit mismatch")
    if [x.get("name") for x in manifest["files"]] != ["input_manifest.json","member_metrics.jsonl","summary.json"]:
        raise RuntimeError("payload manifest rows mismatch")
    for item in manifest["files"]:
        p=directory/item["name"]
        if p.stat().st_size!=item["size"] or _file_digest(p)!=item["sha256"] or item["mode"]!="0444":
            raise RuntimeError("payload seal mismatch")
    input_manifest=json.loads((directory/"input_manifest.json").read_bytes())
    expected_input_keys={"schema","status","program_sha256","grant_sha256","receipt","seed_manifest_sha256",
                         "grant_commit","pinned_inputs","implementation","science_contract_sha256",
                         "frozen_forward_model_contract_sha256","margin_contract_sha256",
                         "field_hash_contract_sha256","firewall_sha256","source_contract_sha256",
                         "forbidden_inventory_sha256","reference_inventory_sha256","forbidden_intersections",
                         "N_expected","environment_pins","authorization","automatic_promotion","downstream_execution"}
    if set(input_manifest)!=expected_input_keys: raise RuntimeError("input manifest exact keyset mismatch")
    if input_manifest.get("schema")!="ouruniv-cf4-lg-unconstrained-p1-reference-input-manifest-v1" \
            or input_manifest.get("status")!="complete" \
            or input_manifest.get("program_sha256")!=_file_digest(ROOT/"config/cf4_lg_unconstrained_p1_reference_program_v1.json") \
            or input_manifest.get("grant_sha256")!=grant_file_sha \
            or input_manifest.get("grant_commit")!=grant_commit \
            or input_manifest.get("receipt")!=grant["allocation_receipt"] \
            or input_manifest.get("seed_manifest_sha256")!=program["inputs"]["seed_manifest"]["sha256"] \
            or input_manifest.get("N_expected")!=768 \
            or input_manifest.get("automatic_promotion") is not False \
            or input_manifest.get("downstream_execution") is not False:
        raise RuntimeError("input manifest mismatch")
    seed=json.loads((ROOT/program["inputs"]["seed_manifest"]["path"]).read_bytes())
    verify_seed_manifest_integrity(seed,program,verify_external_sources=True)
    design=json.loads((ROOT/program["inputs"]["design"]["path"]).read_bytes())
    expected_bindings={"pinned_inputs":program["inputs"],"implementation":grant["implementation"],
        "environment_pins":grant["runtime_pins"],"science_contract_sha256":_object_digest(design["science_contract"]),
        "authorization":grant["authorization"],
        "frozen_forward_model_contract_sha256":_object_digest(design["frozen_forward_model_contract"]),
        "margin_contract_sha256":_object_digest(design["margin_and_joint_diagnostics_contract"]),
        "field_hash_contract_sha256":_object_digest(design["field_hash_contract"]),
        "firewall_sha256":_object_digest(design["firewall"]),"source_contract_sha256":_object_digest(seed["sources"]),
        "forbidden_inventory_sha256":_object_digest(seed["forbidden_inventory"]),
        "reference_inventory_sha256":_object_digest(seed["reference_inventory"]),
        "forbidden_intersections":{"seed_uint64":[],"jax_key_words":[]}}
    if any(input_manifest.get(k)!=v for k,v in expected_bindings.items()):
        raise RuntimeError("input manifest provenance/science/inventory binding mismatch")
    lines=(directory/"member_metrics.jsonl").read_bytes().splitlines(keepends=True)
    if len(lines)!=768: raise RuntimeError("member row count is not 768")
    rows=[]
    runtime_binding={"seed_manifest_sha256":program["inputs"]["seed_manifest"]["sha256"],
                     "grant_sha256":grant_file_sha,"grant_commit":grant_commit,
                     "runtime_pins_sha256":_object_digest(grant["runtime_pins"]),
                     "implementation_commit":grant["implementation"]["commit"]}
    for index,line in enumerate(lines):
        row=json.loads(line)
        if line!=_json_bytes(row): raise RuntimeError("member row is not canonical")
        verify_member(row,seed["seed_derivation"]["rows"][index],design["margin_and_joint_diagnostics_contract"],design["field_hash_contract"]["per_member_fields"],runtime_binding)
        rows.append(row)
    sealed_pareto=[dict(row["pareto"]) for row in rows]
    expected=independent_summary(rows,design["margin_and_joint_diagnostics_contract"]["component_order"],
                                 input_manifest["forbidden_intersections"])
    if sealed_pareto != [row["pareto"] for row in rows]: raise RuntimeError("per-member Pareto forgery")
    actual=json.loads((directory/"summary.json").read_bytes())
    verify_summary_exact(actual,expected)
    for name in ("input_manifest.json","summary.json","manifest.json","COMPLETE"):
        value=json.loads((directory/name).read_bytes())
        if (directory/name).read_bytes()!=_json_bytes(value): raise RuntimeError(f"noncanonical {name}")
    return {"status":"PASS","rows":768,"entries":sorted(ENTRY_SET)}


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",type=Path,required=True);p.add_argument("--grant",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--private",action="store_true");a=p.parse_args()
    program=json.loads(a.config.read_bytes());grant=json.loads(a.grant.read_bytes())
    if a.config.resolve()!=(ROOT/"config/cf4_lg_unconstrained_p1_reference_program_v1.json").resolve() \
            or a.grant.resolve()!=(ROOT/program["lineage"]["future_grant_path"]).resolve():
        raise SystemExit("canonical program/grant paths are required")
    for spec in program["inputs"].values():
        path=Path(spec["path"]);path=path if path.is_absolute() else ROOT/path
        if _file_digest(path)!=spec["sha256"]: raise SystemExit("pinned input changed")
    print(json.dumps(check_output_directory(a.output,program,grant,a.private),sort_keys=True))


if __name__=="__main__": main()
