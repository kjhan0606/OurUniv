import hashlib
from pathlib import Path

import numpy as np

from hong2021_v35_spectrum_phase import (
    PROGRAM_SHA256,
    _phase_accumulator,
    _phase_result,
    band_cross,
    decomposition_summary,
    transforms_and_spectra,
)


REPO = Path(__file__).resolve().parents[1]


def test_v35_program_hash_revision_and_firewall():
    path = REPO / "config/hong2021_v35_residual_spectrum_phase_program.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PROGRAM_SHA256
    text = path.read_text()
    assert '"status": "frozen_revision_before_implementation_or_execution"' in text
    assert '"execution_before_revision": false' in text
    assert '"posthoc_Ak": false' in text
    assert '"Astrid_access": "forbidden"' in text


def test_fourier_total_power_identity_in_every_band():
    rng = np.random.default_rng(3)
    backbone = rng.normal(size=(64, 64, 64))
    residual = rng.normal(size=(64, 64, 64))
    transforms, power = transforms_and_spectra(backbone, residual, backbone + residual)
    cross = band_cross(transforms[0], transforms[1])
    np.testing.assert_allclose(power[2], power[0] + power[1] + 2 * cross, rtol=1e-12)
    summary = decomposition_summary(power[0], power[1], cross)
    np.testing.assert_allclose(summary["total_power"], power[2], rtol=1e-12)


def test_phase_result_reports_candidate_power_error():
    accumulator = _phase_accumulator()
    for key in accumulator:
        accumulator[key][:] = 4.0
    accumulator["candidate_residual_power"][:] = 5.0
    result = _phase_result(accumulator)
    assert len(result["absolute_log10_total_power_error"]) == 8
    assert np.all(np.asarray(result["candidate_over_reference_total_power"]) > 1)
