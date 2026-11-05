"""Score-blind v6 mark-holdout ownership and role checks."""
import unittest

from cf4_r2_mark_holdout_manifest import build_role_aware_mark_holdout_manifest


def fixture():
    return dict(
        fp_pgc=[101, 202],
        fp_source_group=['holdout-A', 'train-B'],
        membership_state=['source_ungrouped_catalogue_present'] * 2,
        source_obj_id=[1001, 1002],
        cf4_group=[11, 22],
        group_labels=['holdout-A', 'train-B'],
        fp_role=[1, 0],
        row_group=[0, 1],
        anchor_group=[],
        point_recno=[501, 502],
        point_role=[1, 0],
        crossmatch_rows=[
            {'PGC': 101, 'twompp_recno': '501', 'match_class': 'secure_joint_mark'},
            {'PGC': 202, 'twompp_recno': '502', 'match_class': 'secure_joint_mark'},
        ],
        parent_index_by_objid={1001: 31, 1002: 32},
        parent_role_by_index={31: 1, 32: 0},
        m2pp_gid_by_recno={501: 71, 502: 72},
        m2pp_gid_role={71: 1, 72: 0},
        cf4_role_by_id={11: 1, 22: 0},
    )


class RoleAwareMarkHoldoutManifestTests(unittest.TestCase):
    def build(self, **overrides):
        values = fixture()
        values.update(overrides)
        return build_role_aware_mark_holdout_manifest(**values)

    def test_certifies_one_role_consistent_identity_only_row(self):
        manifest, rejected = self.build()
        self.assertEqual(rejected, [])
        self.assertEqual(len(manifest), 1)
        row = manifest[0]
        self.assertEqual((row['PGC'], row['source_group'], row['point_recno']),
                         (101, 'holdout-A', 501))
        self.assertEqual(row['roles'], dict(fp_source_group=1, linked_point=1,
            tempel_parent=1, m2pp_gid=1, CF4_group=1))
        self.assertEqual(set(row), {
            'PGC', 'source_group', 'point_recno', 'membership_state',
            'source_obj_id', 'tempel_parent_index', 'm2pp_gid', 'CF4_group', 'roles'})

    def test_unmatched_edges_do_not_count_as_unresolved_association(self):
        edges = fixture()['crossmatch_rows'] + [
            {'PGC': 101, 'twompp_recno': '', 'match_class': 'unmatched'}]
        manifest, rejected = self.build(crossmatch_rows=edges)
        self.assertEqual([row['PGC'] for row in manifest], [101])
        self.assertEqual(rejected, [])

    def test_secure_link_to_another_fold_is_unresolved(self):
        edges = fixture()['crossmatch_rows'] + [
            {'PGC': 101, 'twompp_recno': '502', 'match_class': 'secure_joint_mark'}]
        manifest, rejected = self.build(crossmatch_rows=edges)
        self.assertEqual(manifest, [])
        self.assertIn('unresolved_association_edge', rejected[0]['reasons'])

    def test_collision_and_nonsecure_edges_fail_closed(self):
        data = fixture()
        data['fp_pgc'] = [101, 101]
        manifest, rejected = build_role_aware_mark_holdout_manifest(**data)
        self.assertEqual(manifest, [])
        self.assertIn('unresolved_association_edge', rejected[0]['reasons'])
        self.assertIn('PGC_group_collision', rejected[0]['unresolved_edge_counts'])

    def test_parent_gid_and_cf4_role_mismatches_are_reported(self):
        for field, value, reason in (
                ('parent_role_by_index', {31: 0, 32: 0}, 'source_parent_role_mismatch'),
                ('m2pp_gid_role', {71: 0, 72: 0}, 'm2pp_gid_role_mismatch'),
                ('cf4_role_by_id', {11: 0, 22: 0}, 'cf4_group_role_mismatch')):
            with self.subTest(reason=reason):
                manifest, rejected = self.build(**{field: value})
                self.assertEqual(manifest, [])
                self.assertIn(reason, rejected[0]['reasons'])

    def test_two_target_links_and_anchor_are_not_silently_selected(self):
        data = fixture()
        data['point_recno'] = [501, 503, 502]
        data['point_role'] = [1, 1, 0]
        data['crossmatch_rows'] = data['crossmatch_rows'] + [
            {'PGC': 101, 'twompp_recno': '503', 'match_class': 'secure_joint_mark'}]
        manifest, rejected = build_role_aware_mark_holdout_manifest(**data)
        self.assertEqual(manifest, [])
        self.assertIn('secure_target_role_link_count_not_one', rejected[0]['reasons'])

        data = fixture()
        data['anchor_group'] = [0]
        manifest, rejected = build_role_aware_mark_holdout_manifest(**data)
        self.assertEqual(manifest, [])
        self.assertEqual(rejected, [])

    def test_missing_parent_is_not_claimed_as_certified(self):
        manifest, rejected = self.build(parent_index_by_objid={1002: 32})
        self.assertEqual(manifest, [])
        self.assertIn('source_parent_unmapped', rejected[0]['reasons'])

    def test_duplicate_point_identity_is_rejected(self):
        data = fixture()
        data['point_recno'] = [501, 501]
        with self.assertRaisesRegex(ValueError, 'point recnos must be unique'):
            build_role_aware_mark_holdout_manifest(**data)


if __name__ == '__main__':
    unittest.main()
