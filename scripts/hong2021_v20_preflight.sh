#!/usr/bin/env bash
set -euo pipefail

repo=/home/kjhan/BACKUP/CF4
tng=/gpfs/kjhan/IllustrisTNG/TNG100-1
registry=$repo/config/hong2021_v20_development_program.json
training=$tng/training/tng100_simba_swift_v20_e8_gaussianized_marginal_edm
evaluation=$tng/evaluation/tng100_simba_swift_v20_e8_gaussianized_marginal
sequence=$tng/evaluation/tng100_simba_swift_v20_sequence
out=${1:-$sequence/preflight.json}
cd "$repo"
export PYTHONPATH=$repo/src

host=$(hostname)
if [[ ${host,,} != lageunha ]]; then
  echo "V20 preflight must run on Lageunha, not $host" >&2
  exit 1
fi
if [[ -n $(git status --porcelain) ]]; then
  echo "V20 preflight requires a clean committed worktree" >&2
  git status --short >&2
  exit 1
fi
for forbidden in "$training" "$evaluation" "$repo/config/hong2021_v20_astrid_one_shot_seal.json"; do
  if [[ -e $forbidden ]]; then
    echo "V20 preflight refuses forbidden output: $forbidden" >&2
    exit 1
  fi
done

gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
if [[ ${gpu_name,,} != *ada* ]]; then
  echo "V20 requires the Lageunha Ada GPU, found: $gpu_name" >&2
  exit 1
fi

audit_tmp=$(mktemp "$tng/derived/hong2021_v20/preflight_audit.XXXXXX.json")
trap 'rm -f -- "$audit_tmp"' EXIT
bash scripts/hong2021_v20_prepare_gaussianized.sh >"$audit_tmp"
PYTHONPATH=src pytest -q

python - "$repo" "$registry" "$audit_tmp" "$out" "$gpu_name" <<'PY'
import json, math, os, sys, torch
from pathlib import Path
from hong2021_residual_v12_gaussianized import inverse_gaussianize_torch
from hong2021_v15_edm import git_state
from hong2021_v20_edm import P_MEAN, P_STD, load_frozen_registry
from hong2021_v20_gaussianize import FROZEN_REGISTRY_SHA256
repo, registry_path, audit_path, out, gpu_name = (
    Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), sys.argv[5]
)
commit, clean = git_state(repo)
if not clean:
    raise RuntimeError("V20 preflight worktree became dirty")
registry = load_frozen_registry(registry_path, repo)
experiment = registry["e8_gaussianized_marginal_retrain"]
audit = json.loads(audit_path.read_text())
if audit.get("complete") is not True:
    raise RuntimeError("V20 full derived-artifact audit is incomplete")
sigma_data = float(experiment["initialization_and_normalization"]["sigma_data"])
if abs(math.log(0.6 * sigma_data) - P_MEAN) > 1e-15 or P_STD != 1.2:
    raise RuntimeError("V20 E3 relative-noise derivation failed")
device = torch.device("cuda")
transform = json.loads(Path(experiment["gaussianization"]["path"]).read_text())
z = torch.as_tensor(transform["z_knots"], dtype=torch.float32, device=device)
r = torch.as_tensor(transform["residual_value_knots"], dtype=torch.float32, device=device)
generator = torch.Generator(device=device).manual_seed(202008)
state = generator.get_state().clone()
value = torch.randn((1, 1, 4, 4, 4), generator=generator, device=device)
before_inverse = generator.get_state().clone()
result = inverse_gaussianize_torch(value, z, r)
if not torch.equal(before_inverse, generator.get_state()) or not torch.isfinite(result).all():
    raise RuntimeError("V20 inverse transform consumed RNG or produced non-finite values")
report = {
    "schema": "hong2021-v20-hard-preflight-v1",
    "registry_sha256": FROZEN_REGISTRY_SHA256,
    "code_commit": commit,
    "worktree_clean": clean,
    "host": os.uname().nodename,
    "gpu": gpu_name,
    "cuda_device": torch.cuda.get_device_name(0),
    "derived_artifact_audit": audit,
    "e3_relative_noise": {"sigma_data": sigma_data, "p_mean": P_MEAN, "p_std": P_STD},
    "latent_inverse_additional_rng_draws": 0,
    "independent_data_paths_accessed": False,
    "tests": "full pytest passed immediately before this report",
    "complete": True,
}
out.parent.mkdir(parents=True, exist_ok=True)
partial = out.with_suffix(out.suffix + ".partial")
partial.write_text(json.dumps(report, indent=2) + "\n")
os.replace(partial, out)
print(json.dumps(report, indent=2))
PY
