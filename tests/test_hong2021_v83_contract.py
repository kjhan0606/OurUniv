from __future__ import annotations

from hong2021_v83_contract import DOMAIN_ORDER, partition_digest, partition_indices


def test_partition_is_deterministic_disjoint_and_exhaustive() -> None:
    counts = {"TNG100": 432, "SIMBA": 202, "Swift": 409}
    first = partition_indices(counts)
    second = partition_indices(counts)
    assert first == second
    assert partition_digest(first) == partition_digest(second)
    assert [len(first[d]["holdout"]) for d in DOMAIN_ORDER] == [44, 21, 41]
    for domain in DOMAIN_ORDER:
        fit = first[domain]["fit"]
        holdout = first[domain]["holdout"]
        assert not set(fit) & set(holdout)
        assert sorted(fit + holdout) == list(range(counts[domain]))
