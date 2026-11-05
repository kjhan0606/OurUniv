from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).parents[1] / "scripts/hong2021_v26_gradient_scale_audit.py"
SPEC = importlib.util.spec_from_file_location("hong2021_v26_gradient_scale_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_registered_gradient_geometry_uses_dimension_fractions() -> None:
    report = MODULE.summarize_gradient_scales(
        [1.0] * 6,
        [10.0] * 6,
        weight_decay=1.0e-4,
        clip_threshold=1.0,
    )
    rows = report["scales"]
    assert rows[0]["registered_objective_gradient_norm"] == pytest.approx(
        7 / 262143
    )
    assert rows[-1]["registered_objective_gradient_norm"] == pytest.approx(
        229376 / 262143
    )
    assert report["global_clip_factor"] == pytest.approx(1.0)
    assert rows[0]["weight_decay_over_registered_gradient_norm_proxy"] > 1.0
    assert rows[-1]["weight_decay_over_registered_gradient_norm_proxy"] < 0.01


def test_gradient_geometry_reports_global_clipping() -> None:
    report = MODULE.summarize_gradient_scales(
        [100.0] * 6,
        [1.0] * 6,
        weight_decay=0.0,
        clip_threshold=1.0,
    )
    assert report["global_registered_gradient_norm"] > 1.0
    assert 0.0 < report["global_clip_factor"] < 1.0


class _TwoScaleModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flows = torch.nn.ModuleList(
            [torch.nn.Linear(1, 1, bias=False) for _ in range(6)]
        )


def test_adamw_update_geometry_uses_stored_moments_not_raw_gradient() -> None:
    model = _TwoScaleModel()
    named = list(model.named_parameters())
    state = {
        index: {
            "step": torch.tensor(100.0),
            "exp_avg": torch.ones_like(parameter) * (index + 1),
            "exp_avg_sq": torch.ones_like(parameter) * (index + 1) ** 2,
        }
        for index, (_, parameter) in enumerate(named)
    }
    optimizer = {
        "state": state,
        "param_groups": [
            {
                "params": list(range(len(named))),
                "lr": 1.0e-3,
                "weight_decay": 1.0e-4,
                "eps": 1.0e-8,
                "betas": (0.9, 0.999),
            }
        ],
    }
    report = MODULE.adamw_update_geometry(model, optimizer)
    updates = [row["adam_data_update_norm"] for row in report["scales"]]
    assert max(updates) / min(updates) < 1.001
    assert max(
        row["weight_decay_over_data_update_norm"] for row in report["scales"]
    ) < 1.0e-3
