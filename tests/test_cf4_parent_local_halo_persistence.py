import numpy as np

from cf4_parent_local_halo_persistence import connected_clusters, summarize_clusters


def test_connected_clusters_and_unique_realization_fraction():
    rows = [
        {"realization_id": 1, "position_mpc_h": [0.0, 0.0, 0.0], "mass_msun_h": 5.0},
        {"realization_id": 2, "position_mpc_h": [0.5, 0.0, 0.0], "mass_msun_h": 6.0},
        {"realization_id": 2, "position_mpc_h": [0.6, 0.0, 0.0], "mass_msun_h": 7.0},
        {"realization_id": 3, "position_mpc_h": [4.0, 0.0, 0.0], "mass_msun_h": 8.0},
    ]
    summary = summarize_clusters(rows, n_realizations=4, linking_length=1.0)
    assert len(summary) == 2
    assert summary[0]["unique_realization_count"] == 2
    assert summary[0]["detection_count"] == 3
    assert summary[0]["realization_fraction"] == 0.5
    components = connected_clusters(
        np.asarray([row["position_mpc_h"] for row in rows]), 1.0
    )
    assert [component.tolist() for component in components] == [[0, 1, 2], [3]]
