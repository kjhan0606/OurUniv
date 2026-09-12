from pathlib import Path

import numpy as np

from cf4_aggregate_evidence_oracle import logmeanexp_parent
from cf4_lowk_parent_posterior_freeze import freeze_parent_posterior


def test_freeze_reproduces_parent_marginal_without_seed_draw(tmp_path: Path):
    artifact = tmp_path / "artifact"; artifact.mkdir()
    cache = tmp_path / "cache.npz"
    source = tmp_path / "source.npz"
    output = tmp_path / "output"
    all_keys, all_evidence = [], []
    for group in range(4):
        q = np.zeros((2048, 3)); q[:, 0] = group
        axis = np.zeros((2048, 3)); axis[:, 0] = 1.0
        keys = np.zeros((2048, 6), dtype=np.int16); keys[:, 0] = group
        evidence = np.full((2048, 256), -5.0)
        evidence[:, group] = 0.0
        log_z = logmeanexp_parent(evidence)
        conditional = np.exp(evidence - evidence.max(axis=1, keepdims=True))
        conditional /= conditional.sum(axis=1, keepdims=True)
        np.savez(
            artifact / f"group_{group}_bridge_cycle_16.npz",
            cycle=np.asarray(16), midpoint_mpc_h=q, axis=axis, keys=keys,
            log_Z_bar=log_z, P_parent=conditional.mean(axis=0),
        )
        all_keys.append(keys[0]); all_evidence.append(evidence[0])
    np.savez(cache, keys=np.asarray(all_keys), log_Z=np.asarray(all_evidence))
    np.savez(source, parent_seed=np.arange(3193, 3449))
    result = freeze_parent_posterior(
        artifact_root=artifact, cache_shards=(cache,), source_terminal=source,
        output_root=output,
    )
    assert result["particle_count"] == 8192
    assert result["unique_geometry_key_count"] == 4
    assert result["decision"]["single_parent_seed_selected"] is False
    with np.load(output / "posterior_bank.npz", allow_pickle=False) as item:
        assert item["parent_conditional_probability"].shape == (8192, 256)
        assert np.isclose(item["weight"].sum(), 1.0)
