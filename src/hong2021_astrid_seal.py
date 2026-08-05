"""Dispatch Astrid seal verification without reading independent data."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def verify_astrid_seal(
    seal: str | Path,
    *,
    repo: str | Path,
    require_committed: bool = True,
    require_unopened: bool = False,
) -> dict[str, Any]:
    path = Path(seal)
    if not path.is_absolute():
        path = Path(repo) / path
    schema = json.loads(path.read_text()).get("schema")
    if schema == "hong2021-v14-astrid-one-shot-artifact-seal-v1":
        from hong2021_v14_freeze import verify_seal
    elif schema == "hong2021-v15-astrid-one-shot-artifact-seal-v1":
        from hong2021_v15_freeze import verify_seal
    elif schema == "hong2021-v16-astrid-one-shot-artifact-seal-v1":
        from hong2021_v16_freeze import verify_seal
    elif schema == "hong2021-v17-astrid-one-shot-artifact-seal-v1":
        from hong2021_v17_freeze import verify_seal
    else:
        raise ValueError("unsupported Astrid seal schema")
    return verify_seal(
        path,
        repo=repo,
        require_committed=require_committed,
        require_unopened=require_unopened,
    )
