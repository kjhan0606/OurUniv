from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_zoom_ic_runner_binds_new_parent_mask_and_lcdm_tail() -> None:
    text = (ROOT / "scripts/run_cf4_lg_b3292_s40349_zoom_ic_l12_v2.sbatch").read_text()
    assert "i248_s40349/grafic_l9" in text
    assert "transfer_l9_eh98_v1.npz" in text
    assert "cf4_lg_b3292_s40349_spatial_mask_l9_v1.npz" in text
    assert "--parent-level 9 --parent-grid-size 512" in text
    assert "--tier pilot --levelmin 9 --levelmax-ic 12 --runtime-levelmax 19" in text
    assert "--seed 403495108" in text


def test_zoom_ic_runner_preserves_trace_only_limitations() -> None:
    text = (ROOT / "scripts/run_cf4_lg_b3292_s40349_zoom_ic_l12_v2.sbatch").read_text()
    assert "parent_promoted=false" in text
    assert "m33_resolved=false" in text
    assert 'parent_promoted\"] is False' in text
    assert 'm33_resolved\"] is False' in text
    assert "TRACE_ONLY_ZOOM_IC_L12_GENERATED_PREFLIGHT_PENDING" in text


def test_extended_grafic_header_roundtrip(tmp_path) -> None:
    import numpy as np

    from src.grafic_io import read_grafic_field, write_grafic_field

    field = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
    path = tmp_path / "ic_velcx"
    write_grafic_field(
        path, field, 1.0, (2.0, 3.0, 4.0), 0.02, 0.31, 0.69, 74.6,
        omega_b=0.0,
    )
    restored, metadata = read_grafic_field(path)
    np.testing.assert_array_equal(restored, field)
    assert metadata["header_bytes"] == 48
    assert metadata["omega_b"] == 0.0
