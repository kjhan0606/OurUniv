import hashlib
from pathlib import Path

import numpy as np

from hong2021_residual_evaluate import CENTERED_SCHEMAS
from hong2021_v28_empirical import DOMAIN_ORDER
from hong2021_v37_development_gate import classify
from hong2021_v37_query_alignment import (
    CHANNELS,
    ENSEMBLE_SCHEMA,
    GRID,
    PROGRAM_SHA256,
    SHIFT_CANDIDATES,
    best_periodic_shift,
    source_balanced_moments,
)


REPO = Path(__file__).resolve().parents[1]


def test_v37_program_hash_and_firewall():
    path = REPO / "config/hong2021_v37_query_aligned_copula_program.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PROGRAM_SHA256
    text = path.read_text()
    assert '"validation_used_for_model_or_hyperparameter_choice": false' in text
    assert '"donor_reselection": false' in text
    assert '"field_clipping": false' in text
    assert '"posthoc_Ak": false' in text
    assert '"Astrid_access": "forbidden"' in text


def test_source_balanced_moments_ignore_object_count():
    payload = {
        DOMAIN_ORDER[0]: (np.full(4, 2.0), np.full(4, 4.0), 2),
        DOMAIN_ORDER[1]: (np.full(4, 1000.0), np.full(4, 10000.0), 1000),
        DOMAIN_ORDER[2]: (np.full(4, 6.0), np.full(4, 12.0), 2),
    }
    mean, std = source_balanced_moments(payload)
    np.testing.assert_allclose(mean, 5.0 / 3.0)
    assert np.all(std > 0)


def test_fft_shift_matches_exact_brute_force():
    rng = np.random.default_rng(3701)
    donor = rng.normal(size=(len(CHANNELS), GRID, GRID, GRID))
    expected = (2, -1, 3)
    query = np.roll(donor, shift=expected, axis=(1, 2, 3))
    shift, before, after = best_periodic_shift(query, donor)
    brute = min(
        SHIFT_CANDIDATES,
        key=lambda candidate: (
            np.mean(
                np.square(
                    query
                    - np.roll(donor, shift=candidate, axis=(1, 2, 3))
                )
            ),
            sum(value * value for value in candidate),
            candidate,
        ),
    )
    assert shift == brute == expected
    assert after < before
    assert after < 1.0e-20


def test_classification_order_is_frozen():
    assert classify(primary_pass=True, q3=True, q4=True, causal_material=True)[0] == (
        "bounded_query_alignment_sufficient"
    )
    assert classify(primary_pass=False, q3=True, q4=True, causal_material=False)[0] == (
        "global_query_alignment_repairs_tails_but_not_morphology"
    )
    assert classify(primary_pass=False, q3=False, q4=False, causal_material=True)[0] == (
        "global_query_alignment_is_causal_but_insufficient"
    )


def test_v37_schema_is_evaluable_as_centered_residual():
    assert ENSEMBLE_SCHEMA in CENTERED_SCHEMAS
