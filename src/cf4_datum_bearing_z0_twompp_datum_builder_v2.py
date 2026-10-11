#!/usr/bin/env python3
"""V2 loader-lineage correction for the Phase-A 2M++ datum builder.

V1 accidentally bound the frozen tracer V1 loader, which compares its internal
RA/DEC names to the actual VizieR _RA/_DE input header.  This wrapper retains
the complete V1 datum implementation but binds the already validated tracer V3
module and returns its patched base facade.  No datum, selection, split, gate,
or scientific policy changes.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


def _load_v1() -> Any:
    path = Path(__file__).with_name("cf4_datum_bearing_z0_twompp_datum_builder_v1.py")
    name = "_cf4_datum_bearing_z0_twompp_datum_builder_frozen_v1"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen datum-builder V1")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v1 = _load_v1()
_original_load_module = v1._load_module


def _load_corrected_module(path: Path, name: str) -> Any:
    module = _original_load_module(path, name)
    if path.name == "cf4_twompp_disjoint_tracer_pilot_v3.py":
        return module.base
    return module


v1._load_module = _load_corrected_module
v1.PROGRAM_SCHEMA = "ouruniv-cf4-datum-bearing-z0-twompp-datum-builder-program-v2"
v1.RESULT_SCHEMA = "ouruniv-cf4-datum-bearing-z0-twompp-datum-builder-result-v2"
v1.MANIFEST_SCHEMA = "ouruniv-cf4-datum-bearing-z0-twompp-datum-builder-manifest-v2"
v1.COMPLETE_SCHEMA = "ouruniv-cf4-datum-bearing-z0-twompp-datum-builder-complete-v2"


def main() -> None:
    v1.main()


if __name__ == "__main__":
    main()
