"""Identity-graph closure for source-bound R2 train/holdout roles."""

from __future__ import annotations

from collections import Counter
from typing import Hashable, Iterable, Mapping


def close_graph_roles(
    node_roles: Mapping[Hashable, int],
    edges: Iterable[tuple[Hashable, Hashable]],
) -> tuple[dict[Hashable, int], dict[str, object]]:
    """Buffer every connected component crossing train/holdout or a buffer.

    Role codes are 0=train, 1=heldout, 2=buffer. Nodes absent from
    ``node_roles`` are unlabelled and inherit their component's final role.
    Unconnected unlabelled nodes receive role 3 (unassigned).
    """
    keys: dict[Hashable, int] = {}
    labels: list[Hashable] = []
    parents: list[int] = []
    sizes: list[int] = []

    def intern(key: Hashable) -> int:
        index = keys.get(key)
        if index is not None:
            return index
        index = len(labels)
        keys[key] = index
        labels.append(key)
        parents.append(index)
        sizes.append(1)
        return index

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        root_l, root_r = find(left), find(right)
        if root_l == root_r:
            return
        if sizes[root_l] < sizes[root_r]:
            root_l, root_r = root_r, root_l
        parents[root_r] = root_l
        sizes[root_l] += sizes[root_r]

    for key, role in node_roles.items():
        if role not in (0, 1, 2):
            raise ValueError(f"invalid source-graph role {role!r} for {key!r}")
        intern(key)

    edge_count = 0
    for edge in edges:
        if len(edge) != 2:
            raise ValueError("every source-graph edge must have two endpoints")
        union(intern(edge[0]), intern(edge[1]))
        edge_count += 1

    component_roles: dict[int, set[int]] = {}
    for key, role in node_roles.items():
        component_roles.setdefault(find(keys[key]), set()).add(int(role))

    final_by_root: dict[int, int] = {}
    mixed = buffered = 0
    for index in range(len(labels)):
        root = find(index)
        if root in final_by_root:
            continue
        roles = component_roles.get(root, set())
        if 2 in roles or (0 in roles and 1 in roles):
            final_by_root[root] = 2
            buffered += 1
            if 0 in roles and 1 in roles:
                mixed += 1
        elif 0 in roles:
            final_by_root[root] = 0
        elif 1 in roles:
            final_by_root[root] = 1
        else:
            final_by_root[root] = 3

    final = {key: final_by_root[find(index)] for key, index in keys.items()}
    before = Counter(node_roles.values())
    after = Counter(final[key] for key in node_roles)
    summary = dict(
        graph_nodes=len(labels),
        graph_edges=edge_count,
        connected_components=len(final_by_root),
        mixed_train_heldout_components=mixed,
        components_buffered_by_mixing_or_existing_buffer=buffered,
        labelled_nodes=len(node_roles),
        labelled_roles_before={str(role): int(before.get(role, 0))
                               for role in (0, 1, 2)},
        labelled_roles_after={str(role): int(after.get(role, 0))
                              for role in (0, 1, 2)},
    )
    return final, summary
