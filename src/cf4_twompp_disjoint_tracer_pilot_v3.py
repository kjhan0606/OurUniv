#!/usr/bin/env python3
"""V3 isolated-import correction for the disjoint 2M++ input pilot.

V2 fixed the VizieR header contract but used a normal sibling import, which is
deliberately unavailable under the runner's ``python -I -P`` security flags.
V3 loads the frozen V1 implementation by exact file path.  The V2 catalogue
header correction and every frozen scientific gate remain unchanged.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _load_frozen_base() -> Any:
    path = Path(__file__).with_name("cf4_twompp_disjoint_tracer_pilot.py")
    name = "_cf4_twompp_disjoint_tracer_pilot_frozen_v1"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot construct the frozen V1 module specification")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = _load_frozen_base()
PROGRAM_SCHEMA = "ouruniv-cf4-twompp-disjoint-tracer-pilot-program-v3"
RESULT_SCHEMA = "ouruniv-cf4-twompp-disjoint-tracer-pilot-v3"


def load_program(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load the V3 correction record over the frozen V1 science program."""

    correction_path = Path(path)
    raw = correction_path.read_bytes()
    correction = json.loads(raw)
    if correction.get("schema") != PROGRAM_SCHEMA:
        raise base.PilotError("unexpected V3 pilot program schema")
    base_binding: Mapping[str, Any] = correction["frozen_v1_program"]
    base_path = Path(str(base_binding["path"]))
    base_raw = base_path.read_bytes()
    if hashlib.sha256(base_raw).hexdigest() != base_binding["sha256"]:
        raise base.PilotError("frozen V1 pilot program changed")
    program = copy.deepcopy(json.loads(base_raw))
    if program.get("schema") != "ouruniv-cf4-twompp-disjoint-tracer-pilot-program-v1":
        raise base.PilotError("frozen V1 pilot program schema changed")
    program["schema"] = PROGRAM_SCHEMA
    program["status"] = correction["status"]
    program["authorization"]["authorization_basis"] = correction[
        "authorization_basis"
    ]
    program["implementation"] = correction["implementation"]
    program["execution"]["output"] = correction["execution"]["output"]
    program["execution"]["maximum_submissions"] = 1
    program["implementation_correction_v3"] = correction[
        "implementation_correction_v3"
    ]
    return program, hashlib.sha256(raw).hexdigest()


def load_catalog(path: str | Path) -> dict[str, np.ndarray]:
    """Load the exact VizieR header while retaining V1 internal names."""

    required_input = {
        "recno",
        "Ksmag",
        "Vcmb",
        "c11_5",
        "c12_5",
        "Cln",
        "Ref",
        "_RA",
        "_DE",
    }
    columns: dict[str, list[Any]] = {
        "recno": [],
        "Ksmag": [],
        "Vcmb": [],
        "c11_5": [],
        "c12_5": [],
        "Cln": [],
        "Ref": [],
        "RA": [],
        "DEC": [],
    }
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required_input.issubset(reader.fieldnames):
            raise base.PilotError("2M++ catalogue header changed")
        for row in reader:
            columns["recno"].append(int(row["recno"]))
            columns["Ksmag"].append(float(row["Ksmag"]))
            columns["Vcmb"].append(float(row["Vcmb"]))
            columns["c11_5"].append(float(row["c11_5"]))
            columns["c12_5"].append(
                float(row["c12_5"]) if row["c12_5"].strip() else np.nan
            )
            columns["Cln"].append(int(row["Cln"]))
            columns["Ref"].append(row["Ref"].strip())
            columns["RA"].append(float(row["_RA"]))
            columns["DEC"].append(float(row["_DE"]))
    if len(set(columns["recno"])) != len(columns["recno"]):
        raise base.PilotError("2M++ recno is not unique")
    return {
        "recno": np.asarray(columns["recno"], dtype=np.int64),
        "Ksmag": np.asarray(columns["Ksmag"], dtype=np.float64),
        "Vcmb": np.asarray(columns["Vcmb"], dtype=np.float64),
        "c11_5": np.asarray(columns["c11_5"], dtype=np.float64),
        "c12_5": np.asarray(columns["c12_5"], dtype=np.float64),
        "Cln": np.asarray(columns["Cln"], dtype=np.int8),
        "Ref": np.asarray(columns["Ref"], dtype=str),
        "RA": np.asarray(columns["RA"], dtype=np.float64),
        "DEC": np.asarray(columns["DEC"], dtype=np.float64),
    }


_run_audit_v1 = base.run_audit


def run_audit(*args: Any, **kwargs: Any) -> dict[str, Any]:
    result = _run_audit_v1(*args, **kwargs)
    result["implementation_correction_v3"] = {
        "failed_v1_job_id": 329498,
        "failed_v2_job_id": 329499,
        "v1_failure_layer": "input_header_assertion_before_catalog_deserialization",
        "v2_failure_layer": "isolated_python_sibling_import_before_program_load",
        "v3_base_import": "exact sibling file path through importlib",
        "v3_exact_input_columns": ["_RA", "_DE"],
        "science_or_gate_changed": False,
    }
    return result


base.PROGRAM_SCHEMA = PROGRAM_SCHEMA
base.SCHEMA = RESULT_SCHEMA
base.load_program = load_program
base.load_catalog = load_catalog
base.run_audit = run_audit


if __name__ == "__main__":
    base.main()
