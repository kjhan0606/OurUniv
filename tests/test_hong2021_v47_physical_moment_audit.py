import hashlib
import math
from pathlib import Path

from hong2021_v47_physical_moment_audit import PROGRAM_SHA256, classify


REPO = Path(__file__).resolve().parents[1]


def test_program_hash_and_analytic_limits() -> None:
    path = REPO / "config/hong2021_v47_physical_moment_existence_audit_program.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PROGRAM_SHA256
    target_std = 0.09877202271987233
    assert abs(1.0 / (4.5 * math.log(10.0) * target_std) - 0.9770973825361591) < 1.0e-12
    assert abs(1.0 / (9.0 * math.log(10.0) * target_std) - 0.48854869126807954) < 1.0e-12
    text = path.read_text()
    assert '"fit_or_optimizer": false' in text
    assert '"scale_cap_or_clipping": false' in text
    assert '"independent_gate_locked": true' in text


def test_classification_precedence() -> None:
    assert classify(True, True, True)[0] == (
        "logistic_components_make_the_physical_delta_squared_moment_divergent"
    )
    assert classify(False, True, True)[0] == (
        "logistic_components_make_even_the_mean_physical_density_divergent"
    )
    assert classify(False, False, True)[0] == (
        "rare_logistic_scale_excursions_are_mathematically_incompatible_with_Q4"
    )
    assert classify(False, False, False)[0] == "logistic_physical_moments_exist_on_train_probes"
