import hashlib

import numpy as np

import hong2021_v80_sample as sample


def test_member_seeds_are_stable_and_distinct() -> None:
    first = sample.member_seed(180080, "TNG100", 0, 0)
    assert first == sample.member_seed(180080, "TNG100", 0, 0)
    assert first != sample.member_seed(180080, "TNG100", 0, 1)
    assert first != sample.member_seed(180080, "SIMBA", 0, 0)


def test_innovation_is_reproducible_float32(monkeypatch) -> None:
    monkeypatch.setattr(sample, "GRID", 4)
    first = sample.innovation_numpy(180080, "TNG100", 1, 2)
    second = sample.innovation_numpy(180080, "TNG100", 1, 2)
    assert first.shape == (1, 4, 4, 4)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)


def test_pairing_digest_hashes_per_member_digest_bytes(monkeypatch) -> None:
    monkeypatch.setattr(sample, "GRID", 2)
    monkeypatch.setattr(sample, "QUERIES", 2)
    monkeypatch.setattr(sample, "MEMBERS", 3)
    table = sample.innovation_digest_table(180080, "TNG100")
    assert table.shape == (2, 3, 32)
    assert table.dtype == np.uint8
    assert sample.innovation_pairing_digest(180080, "TNG100") == hashlib.sha256(
        table.tobytes()
    ).hexdigest()


def test_calibration_and_projection_has_small_stored_dc() -> None:
    rng = np.random.default_rng(80010)
    mean = rng.normal(scale=0.05, size=(1, 4, 4, 4))
    total = mean + rng.normal(scale=0.1, size=(3, 1, 4, 4, 4))
    source = np.asarray([-1.0, 0.0, 1.0])
    mapped = np.asarray([-0.9, 0.0, 0.8])
    output, dc = sample._calibrate_and_project(total, mean, source, mapped)
    assert output.dtype == np.float32
    assert dc.shape == (3, 1)
    assert dc.max() <= 1e-8
