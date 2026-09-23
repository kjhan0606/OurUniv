from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_zoom_ic_runner_binds_new_parent_mask_and_lcdm_tail() -> None:
    text = (ROOT / "scripts/run_cf4_lg_b3292_s40349_zoom_ic_l12_v1.sbatch").read_text()
    assert "i248_s40349/grafic_l9" in text
    assert "transfer_l9_eh98_v1.npz" in text
    assert "cf4_lg_b3292_s40349_spatial_mask_l9_v1.npz" in text
    assert "--parent-level 9 --parent-grid-size 512" in text
    assert "--tier pilot --levelmin 9 --levelmax-ic 12 --runtime-levelmax 19" in text
    assert "--seed 403495108" in text


def test_zoom_ic_runner_preserves_trace_only_limitations() -> None:
    text = (ROOT / "scripts/run_cf4_lg_b3292_s40349_zoom_ic_l12_v1.sbatch").read_text()
    assert "parent_promoted=false" in text
    assert "m33_resolved=false" in text
    assert 'parent_promoted\"] is False' in text
    assert 'm33_resolved\"] is False' in text
    assert "TRACE_ONLY_ZOOM_IC_L12_GENERATED_PREFLIGHT_PENDING" in text
