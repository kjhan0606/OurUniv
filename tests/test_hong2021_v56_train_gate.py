import json
from pathlib import Path

from hong2021_v18_init import sha256_file
from hong2021_v28_empirical import DOMAIN_ORDER
from hong2021_v56_train import PROGRAM_SHA256
from hong2021_v56_train_gate import mechanism_pass


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "config/hong2021_v56_survival_grid_program.json"


def test_gate_is_bound_to_frozen_program_and_predevelopment_firewall():
    program = json.loads(PROGRAM.read_text())
    assert sha256_file(PROGRAM) == PROGRAM_SHA256
    assert program["train_mechanism_gate"]["failure_action"].startswith("seal failure")
    assert program["firewall"]["development_access_before_train_mechanism_pass"] == "forbidden"
    assert program["firewall"]["historical_EAGLE_access"] == "forbidden"


def test_gate_requires_absolute_pass_in_every_domain():
    ratios = {domain: 1.0 for domain in DOMAIN_ORDER}
    convergence = {domain: 0.001 for domain in DOMAIN_ORDER}
    assert mechanism_pass(ratios, convergence) is True
    ratios["SIMBA"] = 1.5000001
    assert mechanism_pass(ratios, convergence) is False


def test_gate_rejects_quadrature_failure_or_missing_domain():
    ratios = {domain: 1.0 for domain in DOMAIN_ORDER}
    convergence = {domain: 0.001 for domain in DOMAIN_ORDER}
    convergence["Swift"] = 0.0050001
    assert mechanism_pass(ratios, convergence) is False
    del ratios["TNG100"]
    assert mechanism_pass(ratios, convergence) is False
