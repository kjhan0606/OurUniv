from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SBATCH = ROOT / "scripts/run_cf4_lg_highk_terminal_publication_repair_v2.sbatch"
CONFIG = ROOT / "config/cf4_lg_highk_terminal_publication_repair_program_v2.json"
RUNNER = ROOT / "src/cf4_lg_highk_terminal_publication_repair_v2.py"
CHECKER = ROOT / "scripts/check_cf4_lg_highk_terminal_publication_repair_v2.py"
THIS_TEST = Path(__file__).resolve()


def _text() -> str:
    return SBATCH.read_text()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_slurm_is_exact_debug_cpu_only_grammar() -> None:
    text = _text()
    required = {
        "#SBATCH --partition=a10",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=1",
        "#SBATCH --mem=1G",
        "#SBATCH --time=00:10:00",
        "#SBATCH --no-requeue",
        "#SBATCH --export=NONE",
        "#SBATCH --chdir=/home/kjhan/BACKUP/CF4",
    }
    directives = {line for line in text.splitlines() if line.startswith("#SBATCH ")}
    assert required <= directives
    assert not any("--gres" in line or "--gpus" in line or "--array" in line or "--nodelist" in line for line in directives)
    assert "nvidia-smi" not in text
    config = json.loads(CONFIG.read_text())
    assert config["resources"] == {
        "partition": "a10", "nodes": 1, "tasks": 1,
        "cpus_per_task": 1, "memory": "1G", "walltime": "00:10:00",
        "gpus": 0,
    }


def test_slurm_uses_only_fixed_circle_python_and_three_fail_closed_phases() -> None:
    text = _text()
    assert "readonly python=/home/kjhan/miniconda3/envs/circle/bin/python3.11" in text
    assert "3.11.15" in text
    assert text.count('"$python" -P "$runner" --config "$config"') == 3
    assert '--lineage-preflight' in text and '--test-only' in text
    assert 'exec "$python" -P "$runner" --config "$config"' in text
    assert "jax" not in text.lower() and "pmwd" not in text.lower()
    assert 'os.environ["SLURM_JOB_PARTITION"] == "a10"' in text
    assert "SLURMD_NODENAME" not in text
    assert "grammar-debug" not in text
    assert 'os.environ["SLURM_NNODES"] == "1"' in text
    assert 'os.environ["SLURM_NTASKS"] == "1"' in text
    assert 'os.environ["SLURM_CPUS_PER_TASK"] == "1"' in text
    assert 'os.environ.get("SLURM_JOB_GPUS", "") == ""' in text
    assert "required_design_audit=DESIGN_GO" in text
    assert "required_code_audit=CODE_GO" in text


def test_sbatch_hash_pins_match_current_non_sbatch_bytes() -> None:
    text = _text()
    expected = {
        "config": _sha(CONFIG),
        "runner": _sha(RUNNER),
        "checker": _sha(CHECKER),
        "slurm_test": _sha(THIS_TEST),
    }
    for label, digest in expected.items():
        match = re.search(rf"readonly expected_{label}_sha=([0-9a-f]{{64}})", text)
        assert match and match.group(1) == digest


def test_repair_modules_have_no_science_or_accelerator_imports() -> None:
    forbidden = {
        "jax", "jaxlib", "pmwd", "numpy", "scipy", "cf4_parent_p1",
        "cf4_p2_screen", "cf4_lg_z0_likelihood",
        "cf4_lg_highk_terminal_aggregation",
    }
    for path in (RUNNER, CHECKER):
        tree = ast.parse(path.read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert imported.isdisjoint(forbidden)


def test_protocol_primitives_and_independent_checker_invocation_are_present() -> None:
    runner = RUNNER.read_text()
    for token in (
        "os.O_NOFOLLOW", "os.O_EXCL", "os.mkdir(", "dir_fd=parent_fd",
        "os.fsync(", "os.fchmod(", "0o700", "0o555", "0o444",
        "_verify_source", "_run_private_checker", "_assert_target_binding",
    ):
        assert token in runner
    checker = CHECKER.read_text()
    assert "git\", \"-C" in checker
    assert "source_data[name] != target_data[name]" in checker
    assert "_bind_private_target" in checker and "0o700" in checker
    assert "st_nlink != 1" in checker
    assert "import cf4_lg_highk_terminal_publication_repair" not in checker


def test_exact_six_additive_paths_exist_mode_644() -> None:
    config = json.loads(CONFIG.read_text())
    paths = config["lineage"]["required_exact_added_paths"]
    assert len(paths) == 6 and len(set(paths)) == 6
    for relative in paths:
        path = ROOT / relative
        assert path.is_file() and path.stat().st_mode & 0o777 == 0o644


def test_sbatch_has_valid_bash_syntax() -> None:
    subprocess.run(["bash", "-n", str(SBATCH)], check=True)
