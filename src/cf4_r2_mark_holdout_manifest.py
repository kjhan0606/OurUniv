"""Score-blind structural manifest for the v6 selected-mark holdout.

This module consumes identifiers, split roles and graph/ownership metadata
only. It must not be passed photometry, velocities, counts, field values or
held-out radii/positions. Measurement extraction belongs to post-fit scoring.
"""

from collections import Counter, defaultdict

import numpy as np


ALLOWED_MEMBERSHIP = frozenset({
    'source_ungrouped_catalogue_present',
    'source_ungrouped_catalogue_absent',
    'source_grouped_catalogue_present',
})


def build_role_aware_mark_holdout_manifest(
        *, fp_pgc, fp_source_group, membership_state, source_obj_id, cf4_group,
        group_labels, fp_role, row_group, anchor_group,
        point_recno, point_role, crossmatch_rows,
        parent_index_by_objid, parent_role_by_index,
        m2pp_gid_by_recno, m2pp_gid_role,
        cf4_role_by_id, target_role=1):
    """Return eligible identity rows and explicit structural rejections.

    All supplied arrays are identity, role, or factor-ownership metadata. A
    candidate needs one source mark row, one secure linked point in
    ``target_role``, no anchor, allowed source-membership state, and no
    unresolved/colliding crossmatch edge. Recorded parent, 2M++ GID and CF4
    group roles must agree with the target fold when those links exist.

    This establishes split/ownership bookkeeping only. It does not establish
    physical membership, a calibrated observation law, or posterior validity.
    """
    if target_role not in (0, 1, 2):
        raise ValueError('target role must be train=0, heldout=1, or buffer=2')

    pgc = np.asarray(fp_pgc, dtype=np.int64)
    source_group = np.asarray(fp_source_group).astype(str)
    membership = np.asarray(membership_state).astype(str)
    source_obj = np.asarray(source_obj_id, dtype=np.int64)
    cf4 = np.asarray(cf4_group, dtype=np.int64)
    labels = np.asarray(group_labels).astype(str)
    group_roles = np.asarray(fp_role, dtype=np.int8)
    rows = np.asarray(row_group, dtype=np.int64)
    anchors = np.asarray(anchor_group, dtype=np.int64)
    recnos = np.asarray(point_recno, dtype=np.int64)
    point_roles = np.asarray(point_role, dtype=np.int8)

    n_fp = len(pgc)
    if any(array.shape != (n_fp,) for array in
           (source_group, membership, source_obj, cf4)):
        raise ValueError('FP identity/role columns must align')
    if len(np.unique(labels)) != len(labels) or group_roles.shape != labels.shape:
        raise ValueError('source-group labels must be unique and role-aligned')
    if rows.shape != (n_fp,) or np.any(rows < 0) or np.any(rows >= len(labels)):
        raise ValueError('every FP row must map to one declared source group')
    if anchors.ndim != 1 or np.any(anchors < 0) or np.any(anchors >= len(labels)):
        raise ValueError('anchor ownership indices are invalid')
    if len(np.unique(recnos)) != len(recnos) or point_roles.shape != recnos.shape:
        raise ValueError('point recnos must be unique and role-aligned')
    if np.any(~np.isin(group_roles, (0, 1, 2))) or np.any(~np.isin(point_roles, (0, 1, 2))):
        raise ValueError('unknown source-group or point split role')
    label_index = {label: i for i, label in enumerate(labels)}
    if any(label not in label_index for label in source_group):
        raise ValueError('an FP source label is absent from the frozen v6 role map')

    point_role_by_recno = dict(zip(map(int, recnos), map(int, point_roles)))
    row_count = np.bincount(rows, minlength=len(labels))
    anchor_count = np.bincount(anchors, minlength=len(labels))
    pgc_groups = defaultdict(set)
    group_fp_rows = defaultdict(list)
    for row_index, (object_id, label) in enumerate(zip(pgc, source_group)):
        pgc_groups[int(object_id)].add(str(label))
        group_fp_rows[str(label)].append(row_index)

    links = defaultdict(set)
    unresolved = defaultdict(Counter)
    edge_keys = set()
    for edge in crossmatch_rows:
        object_id = int(edge['PGC'])
        matched_labels = sorted(pgc_groups.get(object_id, ()))
        if not matched_labels:
            continue
        match_class = str(edge.get('match_class', '')).strip()
        raw_recno = str(edge.get('twompp_recno', '')).strip()
        edge_key = (object_id, match_class, raw_recno)
        if edge_key in edge_keys:
            for label in matched_labels:
                unresolved[label]['duplicate_crossmatch_edge'] += 1
            continue
        edge_keys.add(edge_key)
        if len(matched_labels) > 1:
            for label in matched_labels:
                unresolved[label]['PGC_group_collision'] += 1
        if match_class == 'unmatched':
            continue
        if match_class != 'secure_joint_mark':
            for label in matched_labels:
                unresolved[label]['nonsecure_edge'] += 1
            continue
        if not raw_recno:
            for label in matched_labels:
                unresolved[label]['secure_edge_missing_recno'] += 1
            continue
        recno = int(raw_recno)
        if recno not in point_role_by_recno:
            for label in matched_labels:
                unresolved[label]['secure_edge_recno_not_in_v6_split'] += 1
            continue
        if point_role_by_recno[recno] != target_role:
            for label in matched_labels:
                unresolved[label]['secure_edge_outside_target_role'] += 1
            continue
        for label in matched_labels:
            links[label].add(recno)

    manifest = []
    rejected = []
    for group_index, label in enumerate(labels):
        if int(group_roles[group_index]) != target_role:
            continue
        member_rows = group_fp_rows[str(label)]
        if len(member_rows) != 1 or row_count[group_index] != 1:
            continue
        row_index = member_rows[0]
        state = str(membership[row_index])
        if state not in ALLOWED_MEMBERSHIP:
            continue
        if anchor_count[group_index] != 0:
            continue

        reasons = []
        group_links = sorted(links[str(label)])
        if len(group_links) != 1:
            reasons.append('secure_target_role_link_count_not_one')
            rejected.append(dict(source_group=str(label), PGC=int(pgc[row_index]),
                preledger_candidate=False, reasons=reasons,
                unresolved_edge_counts=dict(sorted(unresolved[str(label)].items()))))
            continue
        preledger_candidate = True
        edge_problems = unresolved[str(label)]
        if edge_problems:
            reasons.append('unresolved_association_edge')
        if reasons:
            rejected.append(dict(source_group=str(label), PGC=int(pgc[row_index]),
                preledger_candidate=preledger_candidate, reasons=reasons,
                unresolved_edge_counts=dict(sorted(edge_problems.items()))))
            continue

        recno = group_links[0]
        object_id = int(source_obj[row_index])
        parent_index = parent_index_by_objid.get(object_id)
        if parent_index is None:
            reasons.append('source_parent_unmapped')
        else:
            parent_index = int(parent_index)
            parent_role = parent_role_by_index.get(parent_index)
            if parent_role is None:
                reasons.append('source_parent_role_unmapped')
            elif int(parent_role) != target_role:
                reasons.append('source_parent_role_mismatch')

        gid = m2pp_gid_by_recno.get(recno)
        if gid is not None and int(gid) >= 0:
            gid = int(gid)
            gid_role = m2pp_gid_role.get(gid)
            if gid_role is None:
                reasons.append('m2pp_gid_role_unmapped')
            elif int(gid_role) != target_role:
                reasons.append('m2pp_gid_role_mismatch')
        else:
            gid = None
            gid_role = None

        cf4_id = int(cf4[row_index])
        if cf4_id >= 0:
            cf4_role = cf4_role_by_id.get(cf4_id)
            if cf4_role is None:
                reasons.append('cf4_group_role_unmapped')
            elif int(cf4_role) != target_role:
                reasons.append('cf4_group_role_mismatch')
        else:
            cf4_id = None
            cf4_role = None
        if reasons:
            rejected.append(dict(source_group=str(label), PGC=int(pgc[row_index]),
                preledger_candidate=preledger_candidate,
                reasons=sorted(set(reasons))))
            continue

        manifest.append(dict(
            PGC=int(pgc[row_index]), source_group=str(label), point_recno=recno,
            membership_state=state, source_obj_id=object_id,
            tempel_parent_index=parent_index,
            m2pp_gid=gid, CF4_group=cf4_id,
            roles=dict(fp_source_group=target_role, linked_point=target_role,
                tempel_parent=target_role,
                m2pp_gid=None if gid is None else int(gid_role),
                CF4_group=None if cf4_id is None else int(cf4_role)),
        ))

    manifest.sort(key=lambda row: (row['source_group'], row['PGC'], row['point_recno']))
    for key in ('PGC', 'source_group', 'point_recno'):
        values = [row[key] for row in manifest]
        if len(values) != len(set(values)):
            raise ValueError(f'certified mark manifest repeats {key}')
    return manifest, rejected
