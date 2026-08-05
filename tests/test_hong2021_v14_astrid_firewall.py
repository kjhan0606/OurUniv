import os
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_development_downloader_rejects_astrid(tmp_path):
    result = subprocess.run(
        [
            str(REPO / "scripts/download_hong2021_camels_raw_development.sh"),
            "Astrid",
            str(tmp_path / "Astrid"),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "Refusing non-development suite" in result.stderr
    assert not (tmp_path / "Astrid").exists()


def test_one_shot_downloader_requires_runner_guard(tmp_path):
    environment = os.environ.copy()
    environment.pop("HONG2021_V14_ASTRID_ONE_SHOT", None)
    result = subprocess.run(
        [
            str(REPO / "scripts/download_hong2021_v14_astrid_one_shot.sh"),
            "config/nonexistent-seal.json",
            str(tmp_path / "Astrid"),
        ],
        cwd=REPO,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "outside the sealed one-shot runner" in result.stderr
    assert not (tmp_path / "Astrid").exists()
