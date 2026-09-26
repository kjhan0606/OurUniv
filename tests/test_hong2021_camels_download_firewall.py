from pathlib import Path
import subprocess


SCRIPT = Path(__file__).parents[1] / "scripts" / "download_hong2021_camels_raw_development.sh"


def test_download_script_rejects_sealed_astrid_before_network_access(tmp_path):
    result = subprocess.run(
        ["bash", str(SCRIPT), "Astrid", str(tmp_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Refusing non-development suite" in result.stderr
    assert list(tmp_path.iterdir()) == []
