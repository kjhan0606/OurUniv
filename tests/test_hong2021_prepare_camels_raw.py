import json

import numpy as np
import pytest

from hong2021_prepare_camels_raw import validated_cic_grid


def test_validated_cic_grid_accepts_complete_cache(tmp_path):
    path = tmp_path / "grid.npy"
    field = np.full((80, 80, 80), 256**3 / 80**3, dtype=np.float32)
    np.save(path, field)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema": "hong2021-periodic-dm-particle-grid-v1",
                "complete": True,
                "assignment": "cic",
                "grid": 80,
                "box_mpc_h": 25.0,
                "dm_particles": 256**3,
            }
        )
    )
    actual, metadata = validated_cic_grid(path)
    assert actual.shape == (80, 80, 80)
    assert metadata["assignment"] == "cic"


def test_validated_cic_grid_rejects_ngp_or_zero(tmp_path):
    path = tmp_path / "grid.npy"
    field = np.full((80, 80, 80), 256**3 / 80**3, dtype=np.float32)
    field[0, 0, 0] = 0
    np.save(path, field)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema": "hong2021-periodic-dm-particle-grid-v1",
                "complete": True,
                "assignment": "ngp",
                "grid": 80,
                "box_mpc_h": 25.0,
                "dm_particles": 256**3,
            }
        )
    )
    with pytest.raises(ValueError, match="invalid"):
        validated_cic_grid(path)
