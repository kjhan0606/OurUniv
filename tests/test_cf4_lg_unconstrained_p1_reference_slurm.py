from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SBATCH = ROOT / "scripts/run_cf4_lg_unconstrained_p1_reference_v1.sbatch"
PROGRAM = ROOT / "config/cf4_lg_unconstrained_p1_reference_program_v1.json"


def text():
    return SBATCH.read_text()


def test_sbatch_syntax_and_exact_resources():
    subprocess.run(["bash", "-n", str(SBATCH)], check=True)
    source = text()
    for directive in ["#SBATCH --partition=a10,a40,h100,h200,a100,a100_pcie",
                      "#SBATCH --nodes=1", "#SBATCH --ntasks=1",
                      "#SBATCH --cpus-per-task=16", "#SBATCH --mem=20G",
                      "#SBATCH --gres=gpu:1", "#SBATCH --time=24:00:00",
                      "#SBATCH --no-requeue", "#SBATCH --export=NONE"]:
        assert directive in source
    assert "#SBATCH --array" not in source and "#SBATCH --requeue" not in source
    config = json.loads(PROGRAM.read_bytes())
    resources = config["resources"]
    assert resources["requested_host_memory_GiB"] == 20
    assert resources["estimated_peak_host_memory_GiB"] == 16
    assert resources["GPU_count"] == 1
    assert "h200" in resources["partitions"]
    assert resources["host_memory_headroom_fraction"] >= .20
    assert "sequential" in resources["host_memory_estimate_basis"]


def test_held_allocation_is_event_driven_and_fail_closed():
    source = text()
    assert "--wait-ref" in source and "--wait-timeout" in source
    assert "--expected-old-commit" in source
    assert "command -v inotifywait" not in source
    assert "HELD_ALLOCATION_BLOCKED_NO_SCIENCE" in source
    assert "GRANT_ACTIVE_SCIENCE_GATE_OPEN" in source
    assert source.index("--emit-runtime-receipt") < source.index("--wait-ref")
    assert source.index("--lineage-preflight") < source.index('"$python" -P "$runner" --config "$program" --grant "$grant"\n')
    commands = [line.strip() for line in source.splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
    assert not any(line.startswith("sleep ") or line.startswith("while ")
                   or line.startswith("for ") for line in commands)
    assert "sbatch " not in source and "srun " not in source
    assert "abort_preexecution" in source
    assert "TECHNICAL_PRE_EXECUTION_ABORT_NO_SCIENCE" in source
    assert "TECHNICAL_POST_GATE_INCOMPLETE_NO_RETRY_OR_CLEANUP" in source
    assert source.index("implementation_commit=") < source.index("--emit-runtime-receipt")
    assert source.index("phase=postgate") < source.index("GRANT_ACTIVE_SCIENCE_GATE_OPEN")
    runner = (ROOT / "src/cf4_lg_unconstrained_p1_reference.py").read_text()
    checker = (ROOT / "scripts/check_cf4_lg_unconstrained_p1_reference_v1.py").read_text()
    assert 'f"--id={token}"' in runner
    assert '["nvidia-smi", "--query-gpu=' not in runner
    assert 'check_output(["nvidia-smi"],text=True)' not in runner.replace(" ", "")
    assert 'check_output(["nvidia-smi"],text=True)' not in checker.replace(" ", "")
    assert 'f"--id={gpu_tokens[0]}"' in runner
    assert "f\"--id={current_gpus[0]['allocation_token']}\"" in checker
    assert "JAX_PLATFORM_NAME=gpu" in source and "TF_CPP_MIN_LOG_LEVEL=2" in source


def test_slurm_only_and_manual_controller_guards():
    source = text()
    assert 'SLURM_CLUSTER_NAME:-}" == syntax' in source
    assert '"$(hostname)" != syntax' in source
    assert 'SLURM_JOB_ID:-' in source and 'SLURM_JOB_NODELIST:-' in source
    config = json.loads(PROGRAM.read_bytes())
    assert config["resources"]["route"] == "Slurm_only"
    assert config["resources"]["manual_syntax"] is False
    assert config["resources"]["manual_syn101"] is False


def test_modules_have_no_jax_or_pmwd_top_level_import():
    for path in [ROOT / "src/cf4_lg_unconstrained_p1_reference.py",
                 ROOT / "scripts/check_cf4_lg_unconstrained_p1_reference_v1.py"]:
        tree = ast.parse(path.read_text())
        top = []
        for node in tree.body:
            if isinstance(node, ast.Import): top.extend(alias.name.split('.')[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module: top.append(node.module.split('.')[0])
        assert "jax" not in top and "pmwd" not in top
    runner=(ROOT/"src/cf4_lg_unconstrained_p1_reference.py").read_text()
    for module in ("configuration","gravity","lpt","modes","nbody","pm_util","scatter"):
        assert f'"{module}"' in runner
    assert runner.index('jax.config.jax_enable_x64') < runner.rindex('.standard_normal(')
    checker=(ROOT/"scripts/check_cf4_lg_unconstrained_p1_reference_v1.py").read_text()
    assert "independent_live_allocation_recheck(grant)" in checker


def _git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args]).decode().strip()


def _diff_rows(parent, child):
    output = _git("diff", "--no-renames", "--name-status", parent, child, "--")
    rows = []
    for line in output.splitlines() if output else []:
        fields = line.split("\t")
        assert len(fields) == 2 and fields[0] in {"A", "M", "D"}
        rows.append((fields[0], fields[1]))
    return rows


def _non_tripwire_status():
    raw = subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain=v1", "-z", "--untracked-files=all"])
    rows = []
    for row in raw.split(b"\0"):
        if not row:
            continue
        assert len(row) >= 4 and row[2:3] == b" "
        status, path = row[:2].decode(), row[3:].decode()
        if not path.startswith("scripts/tripwire/"):
            rows.append((status, path))
    return rows


def _assert_exact_implementation(parent, implementation, expected):
    assert _git("rev-list", "--parents", "-n", "1", implementation).split()[1:] == [parent]
    assert sorted(_diff_rows(parent, implementation)) == sorted(("A", path) for path in expected)
    for path in expected:
        tree = _git("ls-tree", implementation, "--", path).split()
        assert len(tree) >= 4 and tree[0] == "100644" and tree[3] == path


def test_exact_six_paths_are_only_new_non_tripwire_files():
    config = json.loads(PROGRAM.read_bytes())
    expected = sorted(config["lineage"]["implementation_exact_added_paths"])
    parent = config["lineage"]["required_parent_commit"]
    grant_path = config["lineage"]["future_grant_path"]
    head = _git("rev-parse", "HEAD")
    status = _non_tripwire_status()
    if head == parent:
        assert sorted(status) == sorted(("??", path) for path in expected)
    else:
        head_parents = _git("rev-list", "--parents", "-n", "1", head).split()[1:]
        assert len(head_parents) == 1
        if head_parents == [parent]:
            implementation = head
            _assert_exact_implementation(parent, implementation, expected)
            assert status in ([], [("??", grant_path)])
        else:
            implementation = head_parents[0]
            _assert_exact_implementation(parent, implementation, expected)
            assert _diff_rows(implementation, head) == [("A", grant_path)]
            grant_tree = _git("ls-tree", head, "--", grant_path).split()
            assert len(grant_tree) >= 4 and grant_tree[0] == "100644" and grant_tree[3] == grant_path
            assert status == []
    assert all((ROOT / path).is_file() and not (ROOT / path).is_symlink() for path in expected)


def test_no_self_hash_prediction_or_current_execution_authority():
    config = json.loads(PROGRAM.read_bytes())
    assert config["lineage"]["self_hash_prediction"] is False
    assert config["lineage"]["implementation_commit"] == "resolved_only_after_exact_six_commit"
    assert config["authorization"]["Slurm_submission"] is False
    assert config["authorization"]["GPFS_write"] is False
    assert config["authorization"]["reference_execution"] is False
