import json
from pathlib import Path

import numpy as np
import torch

from hong2021_v18_init import sha256_file
from hong2021_v50_network import INITIAL_BIASES
from hong2021_v61_preflight import (
    EXISTING_CELLS,
    GRID_CELLS,
    PROGRAM_SHA256,
    reachable_survival_grid_score,
)


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "config/hong2021_v61_reachable_support_model_program.json"


def test_program_is_frozen_hash_bound_and_single_change_only():
    row = json.loads(PROGRAM.read_text())
    parent = row["parent_evidence"]
    record = json.loads((ROOT / parent["v60_record"]).read_text())
    assert sha256_file(PROGRAM) == PROGRAM_SHA256
    assert sha256_file(ROOT / parent["v60_record"]) == parent["v60_record_sha256"]
    assert row["status"] == "frozen_before_preflight_model_implementation_training_or_evaluation"
    assert record["grid"]["total_cells"] == 134
    assert row["single_model_change"]["V56_grid_cells"] == 16
    assert row["single_model_change"]["V61_grid_cells"] == 134
    assert row["single_model_change"]["upper_survival_score_coefficient"] == 0.1
    assert row["firewall"]["training_before_preflight_pass"] == "forbidden"
    assert row["firewall"]["development_access_before_train_gate_pass"] == "forbidden"
    assert row["firewall"]["independent_gate_locked"] is True


def test_reachable_score_decomposition_identity_and_appended_gradient():
    parameters = (
        torch.tensor(INITIAL_BIASES, dtype=torch.float32)
        .reshape(1, 15, 1, 1, 1)
        .expand(1, 15, 2, 2, 2)
        .clone()
        .requires_grad_(True)
    )
    target = torch.zeros((1, 1, 2, 2, 2), dtype=torch.float32)
    backbone = torch.zeros_like(target)
    thresholds = torch.linspace(-2.0, 2.0, GRID_CELLS)
    weights = torch.arange(1, GRID_CELLS + 1, dtype=torch.float64)
    weights = weights / weights.sum()
    total, components, existing, appended = reachable_survival_grid_score(
        parameters,
        target,
        backbone,
        0.0,
        1.0,
        thresholds,
        weights,
    )
    assert components.shape == (GRID_CELLS,)
    assert torch.allclose(total, existing + appended, rtol=0.0, atol=1e-14)
    assert existing > 0.0
    assert appended > 0.0
    gradient = torch.autograd.grad(appended, parameters)[0]
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient)
    assert EXISTING_CELLS == 16


def test_reachable_score_rejects_non_134_cell_grid():
    parameters = torch.zeros((1, 15, 1, 1, 1))
    target = torch.zeros((1, 1, 1, 1, 1))
    with np.testing.assert_raises(ValueError):
        reachable_survival_grid_score(
            parameters,
            target,
            target,
            0.0,
            1.0,
            torch.zeros(GRID_CELLS - 1),
            torch.ones(GRID_CELLS - 1),
        )
