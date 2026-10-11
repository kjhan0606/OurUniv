from pathlib import Path

import numpy as np

from cf4_aggregate_evidence_oracle import geometry_key, logmeanexp_parent
from cf4_lowk_terminal_rejuvenation_pilot import run_pilot


class FakeParentEvaluator:
    def __call__(self, keys):
        rows = []
        for key in keys:
            value = np.asarray(key, dtype=np.float64)
            rows.append([-.01 * np.sum(value**2), -.01 * np.sum((value - 1.0)**2)])
        return keys, np.asarray(rows, dtype=np.float64)


def write_fixture(root: Path, count=16):
    root.mkdir()
    parent = []
    log_i = []
    for replicate in range(4):
        midpoint = np.zeros((count, 3), dtype=np.float64)
        midpoint[:, 0] = replicate * 0.1
        axis = np.zeros((count, 3), dtype=np.float64)
        axis[:, 0] = 1.0
        keys = np.asarray([geometry_key(q, a) for q, a in zip(midpoint, axis)], dtype=np.int16)
        rows = FakeParentEvaluator()(list(map(tuple, keys)))[1]
        log_z_bar = logmeanexp_parent(rows)
        weights = np.full(count, 1.0 / count)
        probability = np.exp(rows - rows.max(axis=1, keepdims=True))
        probability /= probability.sum(axis=1, keepdims=True)
        parent.append(weights @ probability)
        log_i.append(float(np.mean(log_z_bar)))
        np.savez(
            root / f"replicate_{replicate}.npz",
            master_seed=np.asarray(100 + replicate),
            midpoint_mpc_h=midpoint,
            axis=axis,
            keys=keys,
            log_Z_bar=log_z_bar,
            weights=weights,
            ancestor_labels=np.arange(count),
        )
    np.savez(
        root / "terminal_parent_frozen.npz",
        P_rep=np.asarray(parent),
        log_I_bar=np.asarray(log_i),
    )


def test_pilot_writes_checkpoints_and_summary(tmp_path):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    cache = tmp_path / "cache.npz"
    write_fixture(input_root)
    np.savez(cache, keys=np.empty((0, 6), np.int16), log_Z=np.empty((0, 2)))
    result = run_pilot(
        input_root=input_root,
        cache_shard=cache,
        output_root=output_root,
        evaluator=FakeParentEvaluator(),
        checkpoints=(1, 2),
    )
    assert result["status"] == "complete_diagnostic"
    assert [row["sweep"] for row in result["checkpoints"]] == [1, 2]
    assert len(result["checkpoints"][0]["pairs"]) == 6
    assert (output_root / "result.json").is_file()
    for replicate in range(4):
        assert (output_root / f"replicate_{replicate}_sweep_2.npz").is_file()
        assert (output_root / f"new_evidence_cache_replicate_{replicate}.npz").is_file()
