#!/usr/bin/env python3
"""Frozen unconstrained P1 reference producer.

Importing this module is CPU-only.  JAX, PMWD, the forward factory, and the P1
scorer are imported only after a committed execution grant and its held Slurm
allocation receipt have passed every fail-closed gate.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import select
import stat
import struct
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROGRAM_SCHEMA = "ouruniv-cf4-lg-unconstrained-p1-reference-program-v1"
GRANT_SCHEMA = "ouruniv-cf4-lg-unconstrained-p1-reference-execution-grant-v1"
GRANT_STATUS = "authorized_one_live_held_allocation_waiting_worker_activation"
ROW_SCHEMA = "ouruniv-cf4-lg-unconstrained-p1-reference-member-metric-row-v1"
EXACT_OUTPUTS = {"input_manifest.json", "member_metrics.jsonl", "summary.json", "manifest.json", "COMPLETE"}
GATES = ("Virgo", "Coma", "LocalVoid", "BootesVoid", "ObserverEnvironment")
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


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode()


def object_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_canonical_json(path: Path, exact_keys: set[str], label: str) -> dict[str, Any]:
    raw = path.read_bytes(); value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != exact_keys:
        raise ValueError(f"{label} exact top-level keyset mismatch")
    if raw != canonical_bytes(value):
        raise ValueError(f"{label} is not canonical compact sorted JSON plus newline")
    return value


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def validate_critical_contracts(program: Mapping[str, Any], design: Mapping[str, Any]) -> None:
    cosmology={"A_s_1e9":1.63,"Ob":0.05,"Om":0.31,"h":0.746,"ns":0.96}
    expected_science={"box_size_mpc_h":384.0,"contract_source":"committed design model/science/margins/field hashes/firewall",
        "cosmology":cosmology,"density_smoothing_mpc_h":4.0,
        "forward":"make_forward(192,2.0,jnp.float32,return_dens=True,cosmology=frozen)",
        "gate_order":list(GATES),"initial_generator":"PCG64 seed_uint64 standard_normal float64 cube then exactly one float32 cast",
        "mesh_N":192,"observer_offset_mpc_h":[0.0,0.0,0.0],
        "smoothing":"gaussian_filter sigma=2 wrap; divide float64 mean; subtract 1; float32",
        "spacing_mpc_h":2.0,"target_redshift":0.0}
    if program["science"]!=expected_science: raise RuntimeError("program science contract mismatch")
    science=design["science_contract"];forward=design["frozen_forward_model_contract"]
    if forward["cosmology"]["canonical_object"]!=cosmology \
            or science["mesh_N"]!=192 or science["box_size_mpc_h"]!=384.0 \
            or science["density_smoothing_mpc_h"]!=4.0 or science["target_redshift"]!=0.0 \
            or science["observer_offset_mpc_h"]!=[0.0,0.0,0.0] \
            or science["exact_gate_order"]!=list(GATES):
        raise RuntimeError("design frozen science contract mismatch")
    dtypes=forward["dtypes"];generator=forward["initial_condition_generator"]
    if dtypes["IC_generator_draw"]!="float64 then one cast to float32 before forward" \
            or dtypes["forward_field_particle_and_mesh"]!="float32" \
            or dtypes["smoothed_density_mean"]!="float64 accumulator" \
            or generator["exact_operation"]!="np.random.Generator(np.random.PCG64(seed_uint64)).standard_normal(size=(192,192,192), dtype=np.float64), then astype(np.float32) once; this is the unconstrained xi draw used by the current posterior without the subsequent CF4 Matheron conditioning term" \
            or forward["PM_forward"]["forward_call"]!="make_forward(192,2.0,jnp.float32,return_dens=True,cosmology=canonical_object); linear_modes -> 2LPT -> nbody -> scatter; take first returned object as z=0 density" \
            or "sigma 2.0 cells" not in forward["density_and_smoothing"]["smoothing"] \
            or "mode='wrap'" not in forward["density_and_smoothing"]["delta_formula"]:
        raise RuntimeError("design generator/dtype/forward/smoothing semantics mismatch")
    if program["inputs"]["P1_config"]!=science["P1_config"] \
            or program["inputs"]["scorer"]!=science["scorer"] \
            or program["inputs"]["forward_factory"]!=forward["PM_forward"]["forward_factory"]:
        raise RuntimeError("program/design science source pin mismatch")
    if design["sampling_design"]["N_ref"]!=768 or design["sampling_design"]["batch_count"]!=48 \
            or design["sampling_design"]["members_per_batch"]!=16 \
            or design["weighting_contract"]["member_weight"]!="exactly 1/768 for every one of the 768 prespecified members" \
            or any(design["weighting_contract"][key] is not False for key in
                   ("deduplication","importance_weights","normalize_again","proposal_correction","quality_weights")):
        raise RuntimeError("design sampling/weighting contract mismatch")
    outputs=program["outputs"];design_outputs=design["outputs_contract"]
    if outputs["canonical_root"]!=design_outputs["prospective_canonical_root"] \
            or outputs["exact_files"]!=design_outputs["exact_files"] \
            or outputs["files_mode"]!="0444" or outputs["final_directory_mode"]!="0555" \
            or outputs["staging_directory_mode"]!="0700" or outputs["atomic_no_overwrite"] is not True:
        raise RuntimeError("output/seal contract mismatch")
    field=design["field_hash_contract"]
    expected_domains=["ouruniv:cf4:lg:unconstrained-p1-reference:v1:initial-white-field",
        "ouruniv:cf4:lg:unconstrained-p1-reference:v1:unsmoothed-z0-cic-density",
        "ouruniv:cf4:lg:unconstrained-p1-reference:v1:smoothed-delta-scorer-input"]
    if field["frame"]["shape"]!=[192,192,192] or field["frame"]["frame_version"]!="field-frame-v1" \
            or [item["domain_tag"] for item in field["per_member_fields"]]!=expected_domains:
        raise RuntimeError("field framing contract mismatch")
    fixed={"Virgo.target_delta_positive":(0.0,1.0),"Virgo.target_shell_percentile":(70.0,100.0),
        "Virgo.peak_shell_percentile":(90.0,100.0),"Virgo.peak_separation_mpc_h":(5.0,5.0),
        "Coma.target_delta_positive":(0.0,1.0),"Coma.target_shell_percentile":(70.0,100.0),
        "Coma.peak_shell_percentile":(90.0,100.0),"Coma.peak_separation_mpc_h":(8.0,8.0),
        "LocalVoid.n_underdense":(3,4.0),"LocalVoid.probe_mean_delta_negative":(0.0,1.0),
        "LocalVoid.median_centre_shell_percentile":(35.0,100.0),
        "BootesVoid.centre_shell_percentile":(35.0,100.0),
        "BootesVoid.mean_delta_radius_12_mpc_h_negative":(0.0,1.0),
        "BootesVoid.mean_delta_radius_24_mpc_h_negative":(0.0,1.0),
        "ObserverEnvironment.excess_mass_radius_5_mpc_h":(1e13,1e13),
        "ObserverEnvironment.mean_delta_radius_5_mpc_h":(-.5,.5),
        "ObserverEnvironment.excess_mass_radius_8_mpc_h":(5e13,5e13)}
    components=design["margin_and_joint_diagnostics_contract"]["components"]
    if len(components)!=17 or {row["id"]:(row["threshold"],row["denominator"]) for row in components}!=fixed:
        raise RuntimeError("frozen margin threshold/denominator contract mismatch")
    if {key for key,value in program["authorization"].items() if value} != {"implementation_creation","unit_static_tests"} \
            or any(type(value) is not bool for value in program["authorization"].values()):
        raise RuntimeError("program authorization contract mismatch")
    expected_forbidden={"execution without exact committed grant/upstream/lineage/runtime pins",
        "manual execution on syntax or syn101","release/reacquire under one grant",
        "retry/resubmit/replacement/seed mutation","importing JAX or PMWD at module import",
        "persisting fields/particles/IC/checkpoints","partial canonical output or overwrite",
        "ranking/promotion/threshold change/downstream work","commit/push/Slurm/GPFS in implementation phase",
        "scripts/tripwire modification"}
    if set(program["forbidden"])!=expected_forbidden \
            or program["diagnostics"].get("component_count")!=17 \
            or program["diagnostics"].get("fixed_denominators_only") is not True \
            or program["diagnostics"].get("ranking_or_identity_extrema") is not False \
            or program["field_integrity"].get("required_each")!=768 \
            or program["field_integrity"].get("required_cross_stage_unique")!=2304:
        raise RuntimeError("program forbidden/diagnostic/field-integrity contract mismatch")


def load_program(path: Path) -> dict[str, Any]:
    canonical_path = ROOT / "config/cf4_lg_unconstrained_p1_reference_program_v1.json"
    if path.resolve() != canonical_path.resolve():
        raise PermissionError("only the canonical program path is accepted")
    value = load_canonical_json(path, PROGRAM_KEYS, "program")
    if value.get("schema") != PROGRAM_SCHEMA:
        raise ValueError("wrong reference program schema")
    if value.get("status") != "implementation_frozen_execution_unauthorized_waiting_exact_one_grant":
        raise PermissionError("program status is not frozen implementation-only")
    if value["execution"].get("currently_authorized") is not False:
        raise PermissionError("program may not authorize its own execution")
    if value["authorization"].get("reference_execution") is not False:
        raise PermissionError("program reference execution must remain false")
    if value["sampling"] != {
        "N_ref": 768, "batch_count": 48, "members_per_batch": 16,
        "maximum_concurrent_member_forwards": 4, "member_weight": 1 / 768,
        "retry": False, "drop": False, "replacement": False,
        "early_stop": False, "best_of_N": False,
    }:
        raise ValueError("sampling contract changed")
    if value["resources"] != {
        "GPU_count": 1, "cluster_name": "syntax", "cpus_per_task": 16,
        "estimated_peak_host_memory_GiB": 16, "host_memory_headroom_fraction": .25,
        "host_memory_estimate_basis": "sequential one-member execution: one N=192 forward plus float64 draw and ephemeral field handoffs",
        "manual_syn101": False, "manual_syntax": False, "nodes": 1,
        "partitions": ["a10", "a40", "h100", "h200", "a100", "a100_pcie"],
        "requested_host_memory_GiB": 20, "requeue": False, "route": "Slurm_only",
        "tasks": 1, "time_limit": "24:00:00",
    }:
        raise ValueError("resource contract changed")
    for name, spec in value["inputs"].items():
        source = _resolve(spec["path"])
        if file_sha256(source) != spec["sha256"]:
            raise RuntimeError(f"pinned input changed: {name}")
    design=json.loads(_resolve(value["inputs"]["design"]["path"]).read_bytes())
    validate_critical_contracts(value,design)
    return value


def load_seed_manifest(program: Mapping[str, Any]) -> dict[str, Any]:
    spec = program["inputs"]["seed_manifest"]
    seed_path = _resolve(spec["path"]); raw = seed_path.read_bytes(); value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != SEED_MANIFEST_KEYS \
            or raw != canonical_bytes(value):
        raise ValueError("seed manifest keyset/canonical bytes mismatch")
    rows = value["seed_derivation"]["rows"]
    if len(rows) != 768 or [row[0] for row in rows] != list(range(768)):
        raise RuntimeError("seed manifest does not cover indices 0..767 exactly")
    if len({row[1] for row in rows}) != 768 or len({row[2] for row in rows}) != 768 \
            or len({(row[3], row[4]) for row in rows}) != 768:
        raise RuntimeError("seed manifest contains an internal collision")
    if value["lineage"]["parent"] != program["inputs"]["erratum_v2"]["commit"]:
        raise RuntimeError("seed manifest parent binding changed")
    if value["integrity"]["forbidden_seed_intersection"] \
            or value["integrity"]["forbidden_jax_intersection"]:
        raise RuntimeError("seed manifest is not disjoint")
    return value


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True).stdout.strip()


def _diff_rows(parent: str, child: str) -> list[tuple[str, str]]:
    text = _git("diff", "--no-renames", "--name-status", parent, child, "--")
    rows = []
    for line in text.splitlines() if text else []:
        fields = line.split("\t")
        if len(fields) != 2 or fields[0] not in {"A", "M", "D"}:
            raise RuntimeError("rename/copy/malformed Git change")
        rows.append((fields[0], fields[1]))
    return rows


def validate_lineage_values(program: Mapping[str, Any], grant: Mapping[str, Any], *,
                            head: str, upstream: str, head_parents: Sequence[str],
                            implementation_parents: Sequence[str],
                            implementation_rows: Sequence[tuple[str, str]],
                            grant_rows: Sequence[tuple[str, str]],
                            implementation_modes: Mapping[str, str],
                            implementation_hashes: Mapping[str, str]) -> None:
    paths = program["lineage"]["implementation_exact_added_paths"]
    parent = program["lineage"]["required_parent_commit"]
    implementation = grant["implementation"]["commit"]
    if head != upstream or list(head_parents) != [implementation] \
            or list(implementation_parents) != [parent]:
        raise RuntimeError("grant/implementation/upstream direct-parent lineage mismatch")
    if sorted(implementation_rows) != sorted(("A", p) for p in paths):
        raise RuntimeError("implementation is not exact-six A-only")
    grant_path = program["lineage"]["future_grant_path"]
    if list(grant_rows) != [("A", grant_path)]:
        raise RuntimeError("grant is not exact-one A-only")
    if set(implementation_modes) != set(paths) or any(v != "100644" for v in implementation_modes.values()):
        raise RuntimeError("implementation mode mismatch")
    declared = {row["path"]: (row["mode"], row["sha256"])
                for row in grant["implementation"]["files"]}
    if set(declared) != set(paths):
        raise RuntimeError("grant does not bind exact-six paths")
    for path in paths:
        if declared[path] != ("100644", implementation_hashes[path]):
            raise RuntimeError(f"grant implementation pin mismatch: {path}")


def load_grant(program: Mapping[str, Any], path: Path, receipt: Mapping[str, Any]) -> dict[str, Any]:
    canonical_grant = _resolve(program["lineage"]["future_grant_path"]).resolve()
    if path.resolve() != canonical_grant:
        raise PermissionError("only the canonical exact-one grant path is accepted")
    contract = program["grant_contract"]
    if len(contract["exact_top_level_keys"]) != len(GRANT_KEYS) \
            or set(contract["exact_top_level_keys"]) != GRANT_KEYS:
        raise RuntimeError("program grant keyset contract changed")
    value = load_canonical_json(path, GRANT_KEYS, "grant")
    if contract.get("schema") != GRANT_SCHEMA or contract.get("status") != GRANT_STATUS \
            or value["schema"] != GRANT_SCHEMA or value["status"] != GRANT_STATUS:
        raise PermissionError("execution grant schema/status mismatch")
    if len(contract["authorization_required_true"])!=len(GRANT_AUTH_TRUE) \
            or set(contract["authorization_required_true"])!=GRANT_AUTH_TRUE \
            or len(contract["authorization_required_false"])!=len(GRANT_AUTH_FALSE) \
            or set(contract["authorization_required_false"])!=GRANT_AUTH_FALSE:
        raise RuntimeError("program grant authorization contract changed")
    auth = value["authorization"]
    expected_auth = GRANT_AUTH_TRUE | GRANT_AUTH_FALSE
    if set(auth) != expected_auth:
        raise PermissionError("execution grant authorization keyset mismatch")
    if any(auth.get(k) is not True for k in contract["authorization_required_true"]) \
            or any(auth.get(k) is not False for k in contract["authorization_required_false"]):
        raise PermissionError("execution grant authorization mismatch")
    if value["allocation_receipt"] != receipt:
        raise RuntimeError("grant is not bound to this exact live allocation receipt")
    if value["seed_manifest"]["sha256"] != program["inputs"]["seed_manifest"]["sha256"]:
        raise RuntimeError("grant seed-manifest pin changed")
    if value["seed_manifest"] != {"path": program["inputs"]["seed_manifest"]["path"],
                                  "sha256": program["inputs"]["seed_manifest"]["sha256"],
                                  "commit": program["inputs"]["seed_manifest"]["commit"]}:
        raise RuntimeError("grant seed-manifest identity mismatch")
    head = _git("rev-parse", "HEAD")
    upstream = _git("rev-parse", "@{upstream}")
    implementation = value["implementation"]["commit"]
    parent = program["lineage"]["required_parent_commit"]
    lineage = value["lineage"]
    required = contract["lineage_required_fields"]
    if lineage.get("seed_manifest_commit") != required["seed_manifest_commit"] \
            or lineage.get("implementation_parent_commit") != required["implementation_parent_commit"] \
            or lineage.get("grant_parent_commit") != implementation \
            or lineage.get("same_live_held_allocation") is not True \
            or lineage.get("prior_grant_reuse") is not False:
        raise RuntimeError("grant internal lineage/held-allocation contract mismatch")
    paths = program["lineage"]["implementation_exact_added_paths"]
    modes, hashes = {}, {}
    for item in paths:
        fields = _git("ls-tree", implementation, "--", item).split()
        if len(fields) < 4 or fields[3] != item:
            raise RuntimeError(f"missing implementation tree entry: {item}")
        modes[item] = fields[0]
        hashes[item] = file_sha256(ROOT / item)
    validate_lineage_values(
        program, value, head=head, upstream=upstream,
        head_parents=_git("rev-list", "--parents", "-n", "1", head).split()[1:],
        implementation_parents=_git("rev-list", "--parents", "-n", "1", implementation).split()[1:],
        implementation_rows=_diff_rows(parent, implementation),
        grant_rows=_diff_rows(implementation, head), implementation_modes=modes,
        implementation_hashes=hashes,
    )
    dirty = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain=v1", "-z", "--untracked-files=all", "--", ".",
         ":(exclude)scripts/tripwire/**"], check=True, stdout=subprocess.PIPE).stdout
    if dirty:
        raise RuntimeError("runtime worktree is not clean outside scripts/tripwire/")
    if value["program"] != {"path": "config/cf4_lg_unconstrained_p1_reference_program_v1.json",
                            "sha256": file_sha256(ROOT / "config/cf4_lg_unconstrained_p1_reference_program_v1.json")}:
        raise RuntimeError("grant program pin mismatch")
    if value["outputs"] != program["outputs"]:
        raise RuntimeError("grant output contract mismatch")
    if value["runtime_pins"] != receipt["runtime_pins"]:
        raise RuntimeError("grant runtime pins mismatch")
    repository_pin = receipt["runtime_pins"].get("repository_implementation")
    if not isinstance(repository_pin, Mapping) \
            or repository_pin.get("commit") != value["implementation"]["commit"] \
            or repository_pin.get("files") != value["implementation"]["files"]:
        raise RuntimeError("allocation receipt implementation identity mismatch")
    return value


def runtime_receipt(program: Mapping[str, Any]) -> dict[str, Any]:
    required = ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_JOB_PARTITION",
                "SLURM_CPUS_PER_TASK", "SLURM_NTASKS")
    if any(not os.environ.get(k) for k in required):
        raise PermissionError("active Slurm allocation is required")
    if os.environ["SLURM_JOB_PARTITION"] not in program["resources"]["partitions"] \
            or int(os.environ["SLURM_CPUS_PER_TASK"]) != program["resources"]["cpus_per_task"] \
            or int(os.environ["SLURM_NTASKS"]) != program["resources"]["tasks"]:
        raise PermissionError("Slurm partition/CPU/task allocation differs from frozen resources")
    host = subprocess.check_output(["hostname"], text=True).strip()
    if host == "syntax":
        raise PermissionError("manual/controller execution is forbidden")
    config = subprocess.check_output(["scontrol", "show", "config"], text=True)
    if "ClusterName             = syntax" not in config:
        raise RuntimeError("wrong Slurm cluster")
    job_text = subprocess.check_output(
        ["scontrol", "show", "job", "--oneliner", os.environ["SLURM_JOB_ID"]], text=True).strip()
    job_fields = {}
    for token in job_text.split():
        if "=" in token:
            key, value = token.split("=", 1); job_fields[key] = value
    allocated_hosts = subprocess.check_output(
        ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]], text=True).split()
    if job_fields.get("JobState") != "RUNNING" or host not in allocated_hosts \
            or job_fields.get("Partition") != os.environ["SLURM_JOB_PARTITION"]:
        raise PermissionError("Slurm variables do not identify this active running allocation")
    validate_slurm_resources(job_fields,program["resources"])
    expected_gpus = int(program["resources"]["GPU_count"])
    gpu_tokens = allocated_gpu_tokens(os.environ, expected_gpus)
    import re
    gpus = []
    for token in gpu_tokens:
        gpu_text = subprocess.check_output(
            ["nvidia-smi", f"--id={token}",
             "--query-gpu=index,name,uuid,memory.total,driver_version",
             "--format=csv,noheader,nounits"], text=True)
        lines = gpu_text.splitlines()
        if len(lines) != 1:
            raise RuntimeError("allocated GPU token did not resolve to exactly one GPU")
        line = lines[0]
        fields = [x.strip() for x in line.split(",")]
        if len(fields) != 5:
            raise RuntimeError("malformed nvidia-smi receipt")
        if token.startswith("GPU-") and fields[2] != token:
            raise RuntimeError("allocated GPU UUID token resolved to a different device")
        gpus.append({"allocation_token": token, "index": int(fields[0]),
                     "model": fields[1], "uuid": fields[2],
                     "memory_MiB": int(fields[3]), "driver": fields[4]})
    if len(gpus) != expected_gpus or len({gpu["uuid"] for gpu in gpus}) != expected_gpus:
        raise RuntimeError("allocated visible GPU count mismatch")
    import importlib.metadata as metadata
    package_pins = {}
    for name in ("numpy", "scipy", "jax", "jaxlib", "pmwd"):
        distribution = metadata.distribution(name)
        metadata_entry = next(item for item in distribution.files or [] if str(item).endswith("METADATA"))
        metadata_path = Path(distribution.locate_file(metadata_entry))
        init_path = Path(distribution.locate_file(Path(name) / "__init__.py"))
        if not metadata_path.is_file() or not init_path.is_file():
            raise RuntimeError(f"package pin files absent: {name}")
        package_pins[name] = {"version": distribution.version,
                              "METADATA_path": str(metadata_path),
                              "METADATA_sha256": file_sha256(metadata_path),
                              "import_init_path": str(init_path),
                              "import_init_sha256": file_sha256(init_path)}
    design=json.loads(_resolve(program["inputs"]["design"]["path"]).read_bytes())
    observed=design["frozen_forward_model_contract"]["parent_pipeline_evidence"]["observed_circle_pmwd"]
    pmwd_distribution=metadata.distribution("pmwd");pmwd_modules={}
    if pmwd_distribution.version!=observed["version"]:
        raise RuntimeError("frozen pmwd version mismatch")
    for module in ("configuration","gravity","lpt","modes","nbody","pm_util","scatter"):
        module_path=Path(pmwd_distribution.locate_file(Path("pmwd")/f"{module}.py"))
        expected_sha=observed[f"{module}_py_sha256"]
        if not module_path.is_file() or file_sha256(module_path)!=expected_sha:
            raise RuntimeError(f"frozen pmwd module pin mismatch: {module}")
        pmwd_modules[module]={"path":str(module_path),"sha256":expected_sha}
    nvidia_banner = subprocess.check_output(
        ["nvidia-smi", f"--id={gpu_tokens[0]}"], text=True)
    match = re.search(r"CUDA Version:\s*([^ |]+)", nvidia_banner)
    if not match:
        raise RuntimeError("CUDA version missing from nvidia-smi")
    pins = {
        "job_id": os.environ["SLURM_JOB_ID"], "nodelist": os.environ["SLURM_JOB_NODELIST"],
        "node": host, "partition": os.environ["SLURM_JOB_PARTITION"],
        "cpus_per_task": int(os.environ["SLURM_CPUS_PER_TASK"]),
        "ntasks": int(os.environ["SLURM_NTASKS"]),
        "allocated_memory": os.environ.get("SLURM_MEM_PER_NODE"),
        "Slurm_GPU_variables": {k: os.environ.get(k) for k in
                                ("SLURM_JOB_GPUS", "SLURM_GPUS", "SLURM_GPUS_ON_NODE")},
        "Slurm_environment_variables":{k:os.environ.get(k) for k in
            ("SLURM_JOB_ID","SLURM_JOB_NODELIST","SLURM_JOB_PARTITION","SLURM_JOB_NUM_NODES",
             "SLURM_NTASKS","SLURM_CPUS_PER_TASK","SLURM_MEM_PER_NODE","SLURM_RESTART_COUNT")},
        "Slurm_raw_resource_fields": {k:job_fields.get(k) for k in
            ("NumNodes","NumTasks","NumCPUs","CPUs/Task","ReqTRES","TresPerNode","AllocTRES",
             "TimeLimit","Requeue","Restarts","Partition","JobState","NodeList")},
        "gpus": gpus, "CUDA_reported": match.group(1),
        "python": {"executable": sys.executable, "version": sys.version.split()[0],
                   "sha256": file_sha256(Path(os.path.realpath(sys.executable)))},
        "packages": package_pins,
        "pmwd_module_files":pmwd_modules,
        "platform":{"uname":list(__import__("platform").uname()),
                    "libc":list(__import__("platform").libc_ver()),"byteorder":sys.byteorder},
        "repository_implementation": repository_implementation_identity(program),
        "environment": {k: os.environ.get(k) for k in (
            "CUDA_VISIBLE_DEVICES", "JAX_ENABLE_X64", "XLA_PYTHON_CLIENT_PREALLOCATE",
            "XLA_PYTHON_CLIENT_MEM_FRACTION", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS","JAX_PLATFORM_NAME","TF_CPP_MIN_LOG_LEVEL")},
    }
    return {"schema": "ouruniv-cf4-lg-unconstrained-p1-reference-allocation-receipt-v1",
            "cluster_name": "syntax", "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "slurm_nodelist": os.environ["SLURM_JOB_NODELIST"], "runtime_pins": pins,
            "one_live_held_allocation": True}


def wait_for_exact_ref_event(ref: Path, expected_old_commit: str, timeout_seconds: int) -> None:
    """Wait for an exact Git loose-ref rename using Linux inotify, not polling."""
    ref = ref.resolve(); parent = ref.parent; name = os.fsencode(ref.name)
    import re
    if not ref.is_file() or not parent.is_dir() or timeout_seconds <= 0 \
            or re.fullmatch(r"[0-9a-f]{40}",expected_old_commit) is None:
        raise RuntimeError("invalid exact ref wait target/timeout")
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.inotify_init1(os.O_CLOEXEC | os.O_NONBLOCK)
    if fd < 0:
        error = ctypes.get_errno(); raise OSError(error, os.strerror(error))
    try:
        watch = libc.inotify_add_watch(fd, os.fsencode(parent), 0x00000080)  # IN_MOVED_TO
        if watch < 0:
            error = ctypes.get_errno(); raise OSError(error, os.strerror(error))
        current=ref.read_text(encoding="ascii").strip()
        if re.fullmatch(r"[0-9a-f]{40}",current) is None:
            raise RuntimeError("exact ref contains a malformed commit")
        if current!=expected_old_commit:
            return
        deadline = time.monotonic() + timeout_seconds
        while True:  # event consumption, never a filesystem/status poll
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
                raise TimeoutError("exact grant ref activation timed out")
            data = os.read(fd, 65536); offset = 0
            while offset < len(data):
                _watch, mask, _cookie, length = struct.unpack_from("iIII", data, offset)
                offset += 16; event_name = data[offset:offset + length].split(b"\0", 1)[0]
                offset += length
                if mask & 0x00000080 and event_name == name:
                    return
    finally:
        os.close(fd)


def slurm_gpu_counts(text: str) -> list[int]:
    import re
    return [int(value) for value in re.findall(
        r"(?:gres/)?gpu(?:[:/][A-Za-z0-9_.-]+)?(?::|=)(\d+)(?=,|\s|\)|$)",text)]


def slurm_memory_mib(value: str) -> int:
    import re
    match=re.fullmatch(r"([0-9]+)([KMGT])",value or "")
    if not match: raise RuntimeError("unsupported Slurm memory syntax")
    amount=int(match.group(1));factor={"K":1/1024,"M":1,"G":1024,"T":1024*1024}[match.group(2)]
    result=amount*factor
    if int(result)!=result: raise RuntimeError("Slurm memory is not an integral MiB value")
    return int(result)


def slurm_time_seconds(value: str) -> int:
    import re
    match=re.fullmatch(r"(?:(\d+)-)?(\d{1,2}):(\d{2}):(\d{2})",value or "")
    if not match: raise RuntimeError("unsupported Slurm time syntax")
    days=int(match.group(1) or 0);hours,minutes,seconds=map(int,match.groups()[1:])
    if minutes>=60 or seconds>=60 or hours>24 or (hours==24 and (days or minutes or seconds)):
        raise RuntimeError("invalid Slurm time")
    return days*86400+hours*3600+minutes*60+seconds


def validate_slurm_resources(fields: Mapping[str,str],resources: Mapping[str,Any]) -> None:
    def exact_int(name: str,expected: int) -> None:
        try:value=int(fields.get(name,""))
        except ValueError as error: raise RuntimeError(f"missing Slurm {name}") from error
        if value!=expected: raise RuntimeError(f"Slurm {name} mismatch")
    exact_int("NumNodes",1);exact_int("NumTasks",1);exact_int("NumCPUs",16);exact_int("CPUs/Task",16)
    if fields.get("Partition") not in resources["partitions"] \
            or slurm_time_seconds(fields.get("TimeLimit",""))!=86400 \
            or fields.get("Requeue")!="0" or fields.get("Restarts")!="0":
        raise RuntimeError("Slurm partition/time/requeue/restart mismatch")
    req=fields.get("ReqTRES","")
    import re
    memory=re.search(r"(?:^|,)mem=([^,]+)",req)
    if memory is None or slurm_memory_mib(memory.group(1))!=resources["requested_host_memory_GiB"]*1024:
        raise RuntimeError("Slurm requested memory mismatch")
    gres=" ".join(fields.get(key,"") for key in ("TresPerNode","Gres","AllocTRES","ReqTRES"))
    if resources["GPU_count"] not in slurm_gpu_counts(gres):
        raise RuntimeError("Slurm requested GPU count mismatch")


def allocated_gpu_tokens(environment: Mapping[str, str | None], expected_count: int) -> list[str]:
    raw = environment.get("SLURM_JOB_GPUS") or environment.get("CUDA_VISIBLE_DEVICES")
    if not raw:
        raise RuntimeError("allocated GPU identity variables are absent")
    tokens = [item.strip() for item in raw.split(",") if item.strip()]
    import re
    gpu_uuid = r"GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}"
    if len(tokens) != expected_count or len(set(tokens)) != expected_count \
            or any(not (token.isdigit() or re.fullmatch(gpu_uuid, token)) for token in tokens):
        raise RuntimeError("allocated GPU tokens/count are invalid")
    for name in ("CUDA_VISIBLE_DEVICES", "SLURM_JOB_GPUS"):
        alternate = environment.get(name)
        if alternate:
            alternate_tokens=[item.strip() for item in alternate.split(",") if item.strip()]
            if len(alternate_tokens)!=expected_count or len(set(alternate_tokens))!=expected_count \
                    or any(not (item.isdigit() or re.fullmatch(gpu_uuid,item)) for item in alternate_tokens):
                raise RuntimeError(f"{name} identity/count disagrees with allocation")
    on_node = environment.get("SLURM_GPUS_ON_NODE")
    match = re.search(r"(\d+)$", on_node or "")
    if match is None or int(match.group(1)) != expected_count:
        raise RuntimeError("Slurm allocated GRES/count disagrees")
    return tokens


def repository_implementation_identity(program: Mapping[str, Any]) -> dict[str, Any]:
    head = _git("rev-parse", "HEAD"); parent = program["lineage"]["required_parent_commit"]
    grant_path = program["lineage"]["future_grant_path"]
    head_parents = _git("rev-list", "--parents", "-n", "1", head).split()[1:]
    if len(head_parents) == 1 and _diff_rows(head_parents[0], head) == [("A", grant_path)]:
        implementation = head_parents[0]
    else:
        implementation = head
    if _git("rev-list", "--parents", "-n", "1", implementation).split()[1:] != [parent]:
        raise RuntimeError("cannot resolve stable implementation commit identity")
    paths = program["lineage"]["implementation_exact_added_paths"]
    if sorted(_diff_rows(parent, implementation)) != sorted(("A", path) for path in paths):
        raise RuntimeError("implementation identity is not the exact-six direct child")
    files = []
    for path in paths:
        tree = _git("ls-tree", implementation, "--", path).split()
        if len(tree) < 4 or tree[0] != "100644" or tree[3] != path:
            raise RuntimeError("implementation tree identity mismatch")
        files.append({"path": path, "mode": "100644", "sha256": file_sha256(ROOT / path)})
    return {"commit": implementation, "parent_commit": parent, "files": files}


def _canonical_field(array: np.ndarray, expected_shape: tuple[int, ...]) -> np.ndarray:
    if not isinstance(array, np.ndarray) or array.dtype != np.dtype("<f4") \
            or not array.flags.c_contiguous or array.shape != expected_shape:
        raise RuntimeError("field must already be exact <f4 C-order expected shape")
    if not np.isfinite(array).all():
        raise RuntimeError("field contains nonfinite values")
    return array


def producer_frame_hash(array: np.ndarray, domain_tag: str,
                        expected_shape: tuple[int, ...] = (192, 192, 192)) -> str:
    """Producer framing implementation; checker has a separate implementation."""
    value = _canonical_field(array, expected_shape)
    prefix = domain_tag.encode() + b"\0" + b"field-frame-v1" + b"\0" + b"<f4" + b"\0"
    prefix += len(expected_shape).to_bytes(4, "big")
    prefix += b"".join(int(n).to_bytes(8, "big") for n in expected_shape)
    payload = value.tobytes(order="C")
    prefix += len(payload).to_bytes(8, "big")
    return hashlib.sha256(prefix + payload).hexdigest()


def _load_checker_module():
    path = ROOT / "scripts/check_cf4_lg_unconstrained_p1_reference_v1.py"
    spec = importlib.util.spec_from_file_location("cf4_reference_independent_checker", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load independent checker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def streaming_receipt(array: np.ndarray, field_spec: Mapping[str, Any],
                      reference_index: int, seed_digest: str,
                      expected_shape: tuple[int, ...] = (192, 192, 192)) -> dict[str, Any]:
    value = _canonical_field(array, expected_shape)
    value.setflags(write=False)
    producer = producer_frame_hash(value, field_spec["domain_tag"], expected_shape)
    independent = _load_checker_module().checker_frame_live_array(
        value, field_spec["domain_tag"], expected_shape)
    if producer != independent:
        raise RuntimeError("producer/checker streaming field hashes differ")
    return {"stage": field_spec["stage"], "field_name": field_spec["field_name"],
            "domain_tag": field_spec["domain_tag"], "frame_version": "field-frame-v1",
            "dtype": "<f4", "shape": list(expected_shape), "reference_index": reference_index,
            "seed_digest_sha256": seed_digest, "producer_sha256": producer,
            "independent_checker_sha256": independent, "pass": True}


def score_with_integrity(delta: np.ndarray, scorer: Callable[..., dict[str, Any]],
                         scorer_args: Sequence[Any], field_spec: Mapping[str, Any],
                         reference_index: int, seed_digest: str,
                         expected_shape: tuple[int, ...] = (192, 192, 192)) -> tuple[dict[str, Any], dict[str, Any]]:
    delta.setflags(write=False)
    before = streaming_receipt(delta, field_spec, reference_index, seed_digest, expected_shape)
    score = scorer(delta, *scorer_args)
    after = streaming_receipt(delta, field_spec, reference_index, seed_digest, expected_shape)
    if before["producer_sha256"] != after["producer_sha256"]:
        raise RuntimeError("scorer input changed during scoring")
    before["post_score_sha256"] = after["producer_sha256"]
    return score, before


def _value_at(score: Mapping[str, Any], path: str) -> float:
    value: Any = score
    tokens = path.split("."); cursor = 0
    while cursor < len(tokens):
        if not isinstance(value, Mapping):
            raise KeyError(path)
        chosen = None
        for stop in range(len(tokens), cursor, -1):
            candidate = ".".join(tokens[cursor:stop])
            if candidate in value:
                chosen = (candidate, stop); break
        if chosen is None:
            raise KeyError(path)
        value = value[chosen[0]]; cursor = chosen[1]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise RuntimeError(f"nonfinite/non-numeric score value: {path}")
    return float(value)


def derive_member(score: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    margins, component_pass, by_gate = {}, {}, {g: [] for g in GATES}
    for spec in contract["components"]:
        x, t, den = _value_at(score, spec["value_path"]), float(spec["threshold"]), float(spec["denominator"])
        if not den > 0:
            raise RuntimeError("nonpositive fixed denominator")
        margin = (x - t) / den if spec["comparison"] in {"GE", "GT"} else (t - x) / den
        passed = margin >= 0.0 if spec["comparison"] in {"GE", "LE"} else margin > 0.0
        margins[spec["id"]] = float(margin); component_pass[spec["id"]] = bool(passed)
        by_gate[spec["gate"]].append(float(margin))
    gate_margins = {g: min(by_gate[g]) for g in GATES}
    gates = {g: all(component_pass[x["id"]] for x in contract["components"] if x["gate"] == g) for g in GATES}
    stored = score.get("gates")
    if not isinstance(stored, Mapping) or set(stored) != set(GATES) \
            or any(type(stored.get(g)) is not bool for g in GATES) \
            or {g: stored[g] for g in GATES} != gates:
        raise RuntimeError("stored P1 gates disagree with independent fixed components")
    all_five = all(gates.values())
    if type(score.get("pass")) is not bool or score["pass"] != all_five:
        raise RuntimeError("stored all-five pass disagrees")
    if type(score.get("n_gates_passed")) is not int \
            or score["n_gates_passed"] != sum(gates.values()):
        raise RuntimeError("stored gate count disagrees")
    cfail = [x for x in contract["component_order"] if not component_pass[x]]
    gfail = [g for g in GATES if not gates[g]]
    four = len(gfail) == 1
    return {"component_margins": margins, "gate_margins": gate_margins,
            "maximin_margin": min(gate_margins.values()), "gates": gates,
            "all_five_pass": all_five, "component_failure_set": cfail,
            "gate_failure_set": gfail, "exactly_four_of_five": four,
            "sole_failed_gate": gfail[0] if four else None}


def _ecdf(values: Sequence[float]) -> list[dict[str, Any]]:
    ordered = sorted(float(x) for x in values)
    if not ordered or not all(math.isfinite(x) for x in ordered):
        raise RuntimeError("ECDF requires finite values")
    result, cumulative, i = [], 0, 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j] == ordered[i]:
            j += 1
        cumulative += j - i
        result.append({"x": ordered[i], "count": j - i, "cumulative_count": cumulative,
                       "F_n": cumulative / len(ordered)})
        i = j
    return result


def _cp(count: int, total: int, alpha: float = .05) -> list[float]:
    from scipy.stats import beta
    low = 0.0 if count == 0 else float(beta.ppf(alpha / 2, count, total - count + 1))
    high = 1.0 if count == total else float(beta.ppf(1 - alpha / 2, count + 1, total - count))
    return [low, high]


def _bernoulli(count: int, total: int) -> dict[str, Any]:
    return {"count": count, "rate": count / total, "clopper_pearson_95": _cp(count, total),
            "one_sided_zero_u95": 1 - .05 ** (1 / total) if count == 0 else None}


def _numeric_leaves(value: Any, prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    if isinstance(value, bool) or value is None:
        return result
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise RuntimeError(f"nonfinite score_member scalar: {prefix}")
        result[prefix] = number
    elif isinstance(value, Mapping):
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(_numeric_leaves(value[key], child))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            result.update(_numeric_leaves(item, f"{prefix}[{index}]"))
    return result


def _metric_summaries(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    flattened = [_numeric_leaves(row["score_member"]) for row in rows]
    paths = sorted(flattened[0])
    if any(sorted(item) != paths for item in flattened):
        raise RuntimeError("score_member scalar schema differs across members")
    result = {}
    for path in paths:
        values = sorted(item[path] for item in flattened)
        quantile = lambda q: values[math.ceil(q * len(values)) - 1]
        result[path] = {"count": len(values), "min": values[0], "max": values[-1],
                        "mean": math.fsum(item[path] for item in flattened) / len(values),
                        "quantile_type1": {"0.05": quantile(.05), "0.50": quantile(.50),
                                           "0.95": quantile(.95)}}
    return result


def add_pareto(rows: list[dict[str, Any]], component_order: Sequence[str]) -> None:
    vectors = [[float(r["component_margins"][k]) for k in component_order] for r in rows]
    for i, row in enumerate(rows):
        dominates = dominated = 0
        for j, other in enumerate(vectors):
            if i == j:
                continue
            if all(a >= b for a, b in zip(vectors[i], other)) and any(a > b for a, b in zip(vectors[i], other)):
                dominates += 1
            if all(b >= a for a, b in zip(vectors[i], other)) and any(b > a for a, b in zip(vectors[i], other)):
                dominated += 1
        row["pareto"] = {"dominates_count": dominates, "dominated_by_count": dominated,
                         "nondominated": dominated == 0}


def validate_stage_hash_sets(rows: Sequence[Mapping[str, Any]], expected: int = 768) -> dict[str, Any]:
    names = ("initial_white_field_sha256", "unsmoothed_z0_cic_density_sha256",
             "smoothed_delta_scorer_input_sha256")
    sets = []
    for name in names:
        values = [str(row[name]) for row in rows]
        if len(values) != expected or len(set(values)) != expected:
            raise RuntimeError(f"missing/duplicate stage hashes: {name}")
        sets.append(set(values))
    if len(set().union(*sets)) != 3 * expected:
        raise RuntimeError("cross-stage digest reuse/confusion")
    return {"stage_counts": {n: expected for n in names},
            "stage_unique_counts": {n: expected for n in names},
            "stage_missing_counts": {n: 0 for n in names},
            "stage_duplicate_counts": {n: 0 for n in names},
            "cross_stage_union_unique_count": 3 * expected,
            "cross_stage_confusion_count": 0, "pass": True}


def build_summary(rows: list[dict[str, Any]], component_order: Sequence[str],
                  forbidden_intersections: Mapping[str, Sequence[Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        raise RuntimeError("empty reference rows")
    seed_intersection=list(forbidden_intersections.get("seed_uint64", []))
    jax_intersection=list(forbidden_intersections.get("jax_key_words", []))
    if seed_intersection or jax_intersection:
        raise RuntimeError("forbidden/reference inventory is not disjoint")
    add_pareto(rows, component_order)
    pass_counts = {g: sum(bool(r["gates"][g]) for r in rows) for g in GATES}
    all_count = sum(bool(r["all_five_pass"]) for r in rows)
    patterns = {format(i, "05b"): 0 for i in range(32)}
    for row in rows:
        patterns["".join("1" if row["gates"][g] else "0" for g in GATES)] += 1
    four = {g: sum(r["sole_failed_gate"] == g for r in rows) for g in GATES}
    component_ecdf = {k: _ecdf([r["component_margins"][k] for r in rows]) for k in component_order}
    gate_ecdf = {g: _ecdf([r["gate_margins"][g] for r in rows]) for g in GATES}
    failure_sets: dict[str, int] = {}
    component_failure_sets: dict[str, int] = {}
    for row in rows:
        key = json.dumps(row["gate_failure_set"], separators=(",", ":"))
        failure_sets[key] = failure_sets.get(key, 0) + 1
        key = json.dumps(row["component_failure_set"], separators=(",", ":"))
        component_failure_sets[key] = component_failure_sets.get(key, 0) + 1
    co, union, jac = [], [], []
    for a in GATES:
        cr, ur, jr = [], [], []
        for b in GATES:
            av = [not r["gates"][a] for r in rows]; bv = [not r["gates"][b] for r in rows]
            inter = sum(x and y for x, y in zip(av, bv)); uni = sum(x or y for x, y in zip(av, bv))
            cr.append(inter); ur.append(uni); jr.append(inter / uni if uni else 1.0)
        co.append(cr); union.append(ur); jac.append(jr)
    histogram = lambda values: {str(k): v for k, v in sorted(__import__("collections").Counter(values).items())}
    return {"schema": "ouruniv-cf4-lg-unconstrained-p1-reference-summary-v1",
            "status": "complete_descriptive_reference_no_downstream_authorization",
            "N_expected": 768, "N_complete": n, "equal_member_weight": 1 / 768,
            "gate_pass": {g: _bernoulli(pass_counts[g], n) for g in GATES},
            "all_five": {"count": all_count, "rate": all_count / n,
                          "clopper_pearson_95": _cp(all_count, n),
                          "one_sided_u95": (1 - .05 ** (1 / n)) if all_count == 0 else None},
            "patterns": {k: _bernoulli(v, n) for k, v in patterns.items()},
            "exact_four_of_five": {**_bernoulli(sum(four.values()), n),
                                     "sole_failed_gate": {g: _bernoulli(four[g], n) for g in GATES}},
            "ECDF": {"components": component_ecdf, "gates": gate_ecdf,
                     "maximin": _ecdf([r["maximin_margin"] for r in rows])},
            "gate_failure_sets": {k: _bernoulli(v, n) for k, v in failure_sets.items()},
            "component_failure_sets": {k: _bernoulli(v, n) for k, v in component_failure_sets.items()},
            "cofailure": {"gate_order": list(GATES), "count": co,
                           "rate": [[x / n for x in row] for row in co],
                           "union_count": union, "jaccard": jac},
            "pareto": {"nondominated_count": sum(r["pareto"]["nondominated"] for r in rows),
                       "nondominated_fraction": sum(r["pareto"]["nondominated"] for r in rows) / n,
                       "dominates_count_histogram": histogram(r["pareto"]["dominates_count"] for r in rows),
                       "dominated_by_count_histogram": histogram(r["pareto"]["dominated_by_count"] for r in rows)},
            "score_member_scalar_metrics": _metric_summaries(rows),
            "stage_hash_integrity": validate_stage_hash_sets(rows, n),
            "forbidden_inventory_disjointness": {
                "seed_uint64_intersection_count": len(seed_intersection),
                "jax_key_words_intersection_count": len(jax_intersection),
                "exact_empty_intersections": True, "pass": True},
            "automatic_promotion": False, "threshold_change": False,
            "downstream_execution": False}


def validate_row_binding(row: Mapping[str, Any], seed_row: Sequence[Any]) -> None:
    expected = {"reference_index": seed_row[0], "seed_digest_sha256": seed_row[1],
                "seed_uint64": seed_row[2], "jax_key_words": [seed_row[3], seed_row[4]]}
    if any(row.get(k) != v for k, v in expected.items()) \
            or any(type(row.get(k)) is not int for k in ("reference_index","batch_index",
                                                          "within_batch_index","seed_uint64")) \
            or not isinstance(row.get("jax_key_words"),list) \
            or any(type(value) is not int for value in row["jax_key_words"]):
        raise RuntimeError("member row seed/reference binding mismatch")
    for receipt in row.get("streaming_receipts", []):
        if receipt.get("reference_index") != seed_row[0] \
                or receipt.get("seed_digest_sha256") != seed_row[1] \
                or type(receipt.get("pass")) is not bool or receipt["pass"] is not True:
            raise RuntimeError("streaming receipt seed/reference binding mismatch")


def _write_file(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data); stream.flush(); os.fchmod(stream.fileno(), 0o444)
        os.fsync(stream.fileno())


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def _publish_noreplace(source: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError(str(target))
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1)
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(target))


def seal_outputs(program: Mapping[str, Any], grant: Mapping[str, Any], receipt: Mapping[str, Any],
                 input_manifest: Mapping[str, Any], rows: list[dict[str, Any]],
                 summary: Mapping[str, Any]) -> Path:
    final = Path(program["outputs"]["canonical_root"])
    staging = final.parent / f".{final.name}.{receipt['slurm_job_id']}.staging"
    if final.exists() or staging.exists():
        raise FileExistsError("canonical or staging target already exists")
    staging.mkdir(mode=0o700)
    payload = {"input_manifest.json": canonical_bytes(input_manifest),
               "member_metrics.jsonl": b"".join(canonical_bytes(r) for r in rows),
               "summary.json": canonical_bytes(summary)}
    for name in ("input_manifest.json", "member_metrics.jsonl", "summary.json"):
        _write_file(staging / name, payload[name])
    grant_file_sha = file_sha256(_resolve(program["lineage"]["future_grant_path"]))
    manifest = {"schema": "ouruniv-cf4-lg-unconstrained-p1-reference-seal-manifest-v1",
                "status": "complete", "grant_sha256": grant_file_sha,
                "grant_commit": _git("rev-parse", "HEAD"),
                "files": [{"name": n, "size": len(payload[n]),
                           "sha256": hashlib.sha256(payload[n]).hexdigest(), "mode": "0444"}
                          for n in ("input_manifest.json", "member_metrics.jsonl", "summary.json")],
                "exact_entry_set": sorted(EXACT_OUTPUTS)}
    _write_file(staging / "manifest.json", canonical_bytes(manifest))
    complete = {"schema": "ouruniv-cf4-lg-unconstrained-p1-reference-complete-v1",
                "status": "complete", "manifest_sha256": file_sha256(staging / "manifest.json"),
                "scientific_result": "descriptive_only", "automatic_promotion": False}
    _write_file(staging / "COMPLETE", canonical_bytes(complete)); _fsync_dir(staging)
    checker = _load_checker_module()
    checker.check_output_directory(staging, program, grant, private=True)
    os.chmod(staging, 0o555); _fsync_dir(staging)
    _publish_noreplace(staging, final); _fsync_dir(final.parent)
    return final


def run_reference(program: Mapping[str, Any], grant: Mapping[str, Any], receipt: Mapping[str, Any]) -> Path:
    seed_manifest = load_seed_manifest(program)
    _load_checker_module().verify_seed_manifest_integrity(
        seed_manifest,program,verify_external_sources=False)
    design = json.loads(_resolve(program["inputs"]["design"]["path"]).read_bytes())
    p1 = json.loads(_resolve(program["inputs"]["P1_config"]["path"]).read_bytes())
    # Science imports are deliberately below the grant gate.
    import jax
    import jax.numpy as jnp
    if jax.config.jax_enable_x64 is not True or jax.default_backend()!="gpu" \
            or len(jax.devices("gpu"))!=program["resources"]["GPU_count"]:
        raise RuntimeError("post-grant JAX x64/GPU visibility preflight failed before field generation")
    from scipy.ndimage import gaussian_filter
    sys.path.insert(0, str(ROOT / "src"))
    from mock_pipeline import make_forward
    from cf4_parent_p1 import score_member
    science = program["science"]
    _, _, forward = make_forward(192, 2.0, jnp.float32, return_dens=True,
                                 cosmology=science["cosmology"])
    field_specs = {x["stage"]: x for x in design["field_hash_contract"]["per_member_fields"]}
    rows = []
    grant_path = _resolve(program["lineage"]["future_grant_path"])
    row_binding = {"seed_manifest_sha256": program["inputs"]["seed_manifest"]["sha256"],
                   "grant_sha256": file_sha256(grant_path),
                   "grant_commit": _git("rev-parse", "HEAD"),
                   "runtime_pins_sha256": object_sha256(grant["runtime_pins"]),
                   "implementation_commit": grant["implementation"]["commit"]}
    for batch in range(48):
        for seed_row in seed_manifest["seed_derivation"]["rows"][batch * 16:(batch + 1) * 16]:
            index, digest, seed, word0, word1 = seed_row
            white = np.random.Generator(np.random.PCG64(seed)).standard_normal(
                size=(192, 192, 192), dtype=np.float64).astype(np.float32)
            r0 = streaming_receipt(white, field_specs["initial_white_field"], index, digest)
            density, _particles = forward(jnp.asarray(white)); density.block_until_ready()
            del white
            density_np = np.ascontiguousarray(np.asarray(density), dtype=np.dtype("<f4"))
            del density, _particles
            r1 = streaming_receipt(density_np, field_specs["unsmoothed_z0_cic_density"], index, digest)
            smoothed = gaussian_filter(density_np, 2.0, mode="wrap")
            del density_np
            delta = np.ascontiguousarray(smoothed / np.mean(smoothed, dtype=np.float64) - 1.0,
                                         dtype=np.dtype("<f4"))
            del smoothed
            score, r2 = score_with_integrity(delta, score_member,
                                             (2.0, p1, science["cosmology"]["Om"]),
                                             field_specs["smoothed_delta_scorer_input"], index, digest)
            derived = derive_member(score, design["margin_and_joint_diagnostics_contract"])
            row = {"schema": ROW_SCHEMA, "reference_index": index, "batch_index": batch,
                   "within_batch_index": index % 16, "seed_digest_sha256": digest,
                   "seed_uint64": seed, "jax_key_words": [word0, word1],
                   "initial_white_field_sha256": r0["producer_sha256"],
                   "unsmoothed_z0_cic_density_sha256": r1["producer_sha256"],
                   "smoothed_delta_scorer_input_sha256": r2["producer_sha256"],
                   "streaming_receipts": [r0, r1, r2], "score_member": score,
                   **row_binding, **derived}
            validate_row_binding(row, seed_row); rows.append(row)
            del delta
    if [r["reference_index"] for r in rows] != list(range(768)):
        raise RuntimeError("member completion is not exact 0..767")
    forbidden_intersections={"seed_uint64":seed_manifest["integrity"]["forbidden_seed_intersection"],
                             "jax_key_words":seed_manifest["integrity"]["forbidden_jax_intersection"]}
    summary = build_summary(rows, design["margin_and_joint_diagnostics_contract"]["component_order"],
                            forbidden_intersections)
    input_manifest = {"schema": "ouruniv-cf4-lg-unconstrained-p1-reference-input-manifest-v1",
                      "status": "complete", "program_sha256": file_sha256(ROOT / "config/cf4_lg_unconstrained_p1_reference_program_v1.json"),
                      "grant_sha256": row_binding["grant_sha256"],
                      "grant_commit": row_binding["grant_commit"], "receipt": receipt,
                      "seed_manifest_sha256": program["inputs"]["seed_manifest"]["sha256"],
                      "pinned_inputs": program["inputs"], "implementation": grant["implementation"],
                      "science_contract_sha256": object_sha256(design["science_contract"]),
                      "frozen_forward_model_contract_sha256": object_sha256(design["frozen_forward_model_contract"]),
                      "margin_contract_sha256": object_sha256(design["margin_and_joint_diagnostics_contract"]),
                      "field_hash_contract_sha256": object_sha256(design["field_hash_contract"]),
                      "firewall_sha256": object_sha256(design["firewall"]),
                      "source_contract_sha256": object_sha256(seed_manifest["sources"]),
                      "forbidden_inventory_sha256": object_sha256(seed_manifest["forbidden_inventory"]),
                      "reference_inventory_sha256": object_sha256(seed_manifest["reference_inventory"]),
                      "forbidden_intersections": forbidden_intersections,
                      "N_expected": 768, "environment_pins": grant["runtime_pins"],
                      "authorization": grant["authorization"],
                      "automatic_promotion": False, "downstream_execution": False}
    return seal_outputs(program, grant, receipt, input_manifest, rows, summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--grant", type=Path)
    parser.add_argument("--emit-runtime-receipt", action="store_true")
    parser.add_argument("--wait-ref", type=Path)
    parser.add_argument("--expected-old-commit")
    parser.add_argument("--wait-timeout", type=int, default=0)
    parser.add_argument("--lineage-preflight", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    args = parser.parse_args()
    program = load_program(args.config)
    load_seed_manifest(program)
    if args.test_only:
        print("TEST_ONLY_PASS_NO_SCIENCE_IMPORTS")
        return
    if args.wait_ref is not None:
        if args.expected_old_commit is None: raise PermissionError("expected old ref commit is required")
        wait_for_exact_ref_event(args.wait_ref,args.expected_old_commit,args.wait_timeout)
        print("EXACT_GRANT_REF_EVENT_OBSERVED")
        return
    receipt = runtime_receipt(program)
    if args.emit_runtime_receipt:
        print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
        return
    if args.grant is None:
        raise PermissionError("exact execution grant is required")
    grant = load_grant(program, args.grant, receipt)
    if args.lineage_preflight:
        print("GRANT_LINEAGE_RUNTIME_PASS")
        return
    output = run_reference(program, grant, receipt)
    print(output)


if __name__ == "__main__":
    main()
