from pathlib import Path

import numpy as np

from cf4_aggregate_evidence_oracle import geometry_key, logmeanexp_parent
from cf4_lowk_cross_mode_bridge_pilot import run_pilot


class FakeParentEvaluator:
    def __call__(self, keys):
        rows = []
        for key in keys:
            value = np.asarray(key, dtype=np.float64)
            rows.append([-.001 * np.sum(value**2), -.001 * np.sum((value - 1.0)**2)])
        return keys, np.asarray(rows)


def fixture(root: Path, count=12):
    root.mkdir()
    evaluator = FakeParentEvaluator()
    for group in range(4):
        q = np.zeros((count, 3)); q[:, 0] = group * 0.2
        axis = np.zeros((count, 3)); axis[:, 0] = 1.0
        keys = np.asarray([geometry_key(x, a) for x, a in zip(q, axis)], np.int16)
        _, rows = evaluator(list(map(tuple, keys)))
        np.savez(
            root / f"replicate_{group}_sweep_32.npz",
            master_seed=np.asarray(100 + group), midpoint_mpc_h=q, axis=axis,
            keys=keys, log_Z_bar=logmeanexp_parent(rows),
        )


def test_bridge_pilot_writes_group_checkpoints_and_comparison(tmp_path):
    source = tmp_path / "source"; output = tmp_path / "output"
    cache = tmp_path / "cache.npz"
    fixture(source)
    np.savez(cache, keys=np.empty((0, 6), np.int16), log_Z=np.empty((0, 2)))
    result = run_pilot(
        source_root=source,
        cache_shards=(cache,),
        output_root=output,
        evaluator=FakeParentEvaluator(),
        particle_count=8,
        betas=np.asarray([0.0, 1.0]),
        bridge_cycles=(1, 2),
        control_sweeps=(2, 4),
        lower_burnin_sweeps=1,
    )
    assert result["status"] == "complete_diagnostic"
    assert len(result["checkpoints"]) == 2
    assert len(result["groups"]) == 4
    assert (output / "result.json").is_file()
    for group in range(4):
        assert (output / f"group_{group}_bridge_cycle_2.npz").is_file()
        assert (output / f"group_{group}_control_sweep_4.npz").is_file()
    assert (output / "new_evidence_cache.npz").is_file()
