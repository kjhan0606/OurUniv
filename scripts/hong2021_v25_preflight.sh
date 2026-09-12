#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
training=$tng/training/tng100_simba_swift_v25_e13_unweighted_edm
evaluation=$tng/evaluation/tng100_simba_swift_v25_e13_unweighted
sequence=$tng/evaluation/tng100_simba_swift_v25_sequence
out=${1:-$sequence/preflight.json}
cd "$repo"
export PYTHONPATH=$repo/src

[[ ${HOSTNAME,,} == lageunha ]] || { echo "V25 preflight must run on Lageunha" >&2; exit 1; }
[[ -z $(git status --porcelain) ]] || { echo "V25 preflight requires a clean committed worktree" >&2; exit 1; }
for forbidden in "$training" "$evaluation"; do
  [[ ! -e $forbidden ]] || { echo "V25 preflight refuses pre-existing output: $forbidden" >&2; exit 1; }
done
pytest -q
mkdir -p "$(dirname "$out")"
python - "$repo" "$out" <<'PY'
import argparse, json, os, socket, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from hong2021_residual_v8_context import ObservableContextUNet
from hong2021_v18_init import sha256_file
from hong2021_v24_edm import PARAMETERS
from hong2021_v25_edm import (
    REGISTRY_SHA256, frozen_training_namespace, load_frozen_program,
)

repo, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
registry_path = repo / "config/hong2021_v25_development_program.json"
registry, artifacts, _, decision = load_frozen_program(registry_path, repo)
if not torch.cuda.is_available():
    raise RuntimeError("V25 preflight requires the Lageunha Ada GPU")
gpu = torch.cuda.get_device_name(0)
if "ada" not in gpu.lower():
    raise RuntimeError(f"V25 preflight expected Ada GPU, found {gpu}")
training = Path(
    "/gpfs/kjhan/IllustrisTNG/TNG100-1/training/"
    "tng100_simba_swift_v25_e13_unweighted_edm"
)
namespace = frozen_training_namespace(
    argparse.Namespace(
        repo=repo, registry=registry_path, out=training, device="cuda"
    ),
    require_preflight=False,
)
parameters = sum(
    value.numel() for value in ObservableContextUNet(base_channels=48).parameters()
)
if parameters != PARAMETERS:
    raise RuntimeError("V25 preflight parameter count mismatch")
commit = subprocess.run(
    ["git", "-C", str(repo), "rev-parse", "HEAD"],
    check=True, capture_output=True, text=True,
).stdout.strip()
report = {
    "schema": "hong2021-v25-hard-preflight-v1",
    "status": "pass",
    "host": socket.gethostname(),
    "gpu": gpu,
    "code_commit": commit,
    "registry": str(registry_path),
    "registry_sha256": sha256_file(registry_path),
    "registry_sha256_expected": REGISTRY_SHA256,
    "parent_v24_development_pass": decision["development_pass"],
    "mechanism_audit_attestation_sha256": registry["mechanism_audit"][
        "attestation_sha256"
    ],
    "profile_sha256": artifacts["profile"]["sha256"],
    "base_channels": namespace.base_channels,
    "parameters": parameters,
    "steps": namespace.steps,
    "candidate_steps": namespace.candidate_steps,
    "optimized_loss_coefficients": {
        "unweighted": 1.0, "tail_weighted": 0.0,
    },
    "tail_weighted_role": "detached_diagnostic_only",
    "Astrid_accessed": False,
    "historical_EAGLE_accessed": False,
    "created_utc": datetime.now(timezone.utc).isoformat(),
}
if report["registry_sha256"] != report["registry_sha256_expected"]:
    raise RuntimeError("V25 preflight registry hash mismatch")
partial = out.with_suffix(out.suffix + ".partial")
partial.write_text(json.dumps(report, indent=2) + "\n")
os.replace(partial, out)
print(json.dumps(report, indent=2))
PY
