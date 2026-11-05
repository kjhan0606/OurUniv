from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
SBATCH = ROOT / "scripts/run_cf4_lg_highk_terminal_aggregation_v1.sbatch"


def _text() -> str:
    return SBATCH.read_text()


def test_exact_terminal_slurm_resources() -> None:
    text = _text()
    for directive in (
        "#SBATCH --partition=a40,a100,h100,h200",
        "#SBATCH --nodes=1", "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=16", "#SBATCH --mem=20G",
        "#SBATCH --gres=gpu:1", "#SBATCH --time=02:00:00",
        "#SBATCH --no-requeue", "#SBATCH --export=NONE",
    ):
        assert directive in text
    assert "#SBATCH --array" not in text


def test_gpu_guard_and_no_JAX_preallocation() -> None:
    text = _text()
    assert "nvidia-smi --query-gpu=memory.total" in text
    assert '${#visible_memory[@]}" -eq 1' in text
    assert "visible_memory[0] >= 40960" in text
    assert "XLA_PYTHON_CLIENT_PREALLOCATE=false" in text


def test_preflight_is_test_only_then_single_execution_and_check() -> None:
    text = _text()
    test_position = text.index('--config "$config" --test-only')
    run_position = text.index('--config "$config"\n', test_position)
    check_position = text.index('"$checker" --config "$config"')
    assert test_position < run_position < check_position
    assert text.count('"$runner" --config "$config"') == 2
    assert text.count('"$python" -P') >= 3
    assert "--array" not in text
    assert "sbatch " not in "\n".join(
        line for line in text.splitlines() if not line.startswith("#SBATCH")
    )


def test_pinned_code_and_exact_commit_preflight() -> None:
    text = _text()
    assert "required_code_audit=CODE_GO" in text
    assert "status --porcelain=v1 -z --untracked-files=all" in text
    assert "':(exclude)scripts/tripwire/**'" in text
    assert "rev-parse '@{upstream}'" in text
    assert "rev-list --parents -n 1 HEAD" in text
    assert "diff-tree --root --no-commit-id --name-status -r HEAD" in text
    assert "ls-tree HEAD" in text and "100644" in text
    for name in (
        "config/cf4_lg_highk_terminal_aggregation_v1.json",
        "src/cf4_lg_highk_terminal_aggregation.py",
        "scripts/run_cf4_lg_highk_terminal_aggregation_v1.sbatch",
        "scripts/check_cf4_lg_highk_terminal_aggregation_v1.py",
        "tests/test_cf4_lg_highk_terminal_aggregation.py",
        "tests/test_cf4_lg_highk_terminal_aggregation_slurm.py",
    ):
        assert name in text
    for variable in ("expected_config_sha", "expected_runner_sha", "expected_checker_sha"):
        match = re.search(rf"readonly {variable}=([0-9a-f]{{64}})", text)
        assert match is not None


def test_no_downstream_or_retry_submission() -> None:
    text = _text()
    executable = "\n".join(line for line in text.splitlines() if not line.startswith("#SBATCH"))
    assert "--requeue" not in executable
    assert "ramses" not in executable.lower()
    assert "promot" not in executable.lower()


def test_circle_python_and_import_preflight_are_fixed() -> None:
    text = _text()
    assert "readonly python=/home/kjhan/miniconda3/envs/circle/bin/python3.11" in text
    assert 'sys.version.split()[0] == "3.11.15"' in text
    for package in ("jax", "jaxlib", "pmwd"):
        assert f'metadata.version("{package}")' in text
    assert "import jax" in text and "import pmwd" in text
