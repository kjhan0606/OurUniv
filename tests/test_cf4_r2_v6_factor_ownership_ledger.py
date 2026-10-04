import unittest

from cf4_r2_v6_factor_ownership_ledger import make_ledger


class V6FactorOwnershipLedgerTests(unittest.TestCase):
    def setUp(self):
        self.key_a = 3
        self.key_b = 4
        self.points = [
            dict(recno=101, role=0, population=0, flat_cell=3),
            dict(recno=102, role=0, population=0, flat_cell=3),
            dict(recno=103, role=0, population=0, flat_cell=4),
        ]
        self.groups = {
            'G1': dict(label='G1', fp_rows=[(0, 501), (1, 502)], anchor_count=0),
            'G2': dict(label='G2', fp_rows=[(2, 504)], anchor_count=0),
            'G3': dict(label='G3', fp_rows=[(3, 503)], anchor_count=0),
        }
        self.edges = [
            dict(PGC='501', match_class='secure_joint_mark', twompp_recno='101'),
            dict(PGC='502', match_class='secure_joint_mark', twompp_recno='102'),
            dict(PGC='503', match_class='extended_review_candidate', twompp_recno=''),
        ]

    def test_one_owner_per_count_key_and_group_mark_with_multimember_class(self):
        count, groups, marks, edges, conflicts = make_ledger(
            [self.key_a, self.key_b], [2, 1], self.points, self.groups, self.edges)
        self.assertEqual([row['count_key'] for row in count], [3, 4])
        self.assertEqual([row['count'] for row in count], [2, 1])
        self.assertEqual(count[0]['linkage_class'], 'count_with_one_CF4_group')
        self.assertEqual(count[1]['linkage_class'], 'count_only')
        by_group = {row['source_group_label']: row for row in groups}
        self.assertEqual(by_group['G1']['category'], 'multi_member_shared_latent_candidate')
        self.assertEqual(by_group['G1']['secure_training_recno_count'], 2)
        self.assertEqual(by_group['G2']['category'], 'unanchored_selected_group_conditional')
        self.assertEqual(by_group['G3']['category'], 'ambiguous_association_unresolved')
        self.assertEqual(len({row['source_group_label'] for row in groups}), 3)
        self.assertEqual(len({row['fp_row_index'] for row in marks}), 4)
        self.assertEqual(len(edges), 3)
        self.assertEqual(conflicts, set())

    def test_shared_recno_group_collision_is_not_silently_owned_twice(self):
        groups = {
            'G1': dict(label='G1', fp_rows=[(0, 501)], anchor_count=0),
            'G2': dict(label='G2', fp_rows=[(1, 505)], anchor_count=0),
        }
        edges = [
            dict(PGC='501', match_class='secure_joint_mark', twompp_recno='101'),
            dict(PGC='505', match_class='secure_joint_mark', twompp_recno='101'),
        ]
        _, rows, _, edge_rows, conflicts = make_ledger(
            [self.key_a, self.key_b], [2, 1], self.points, groups, edges)
        self.assertEqual(conflicts, {101})
        self.assertTrue(all(row['association_collision'] for row in rows))
        self.assertEqual([row['category'] for row in rows],
                         ['association_conflict_unresolved']*2)
        self.assertEqual(sum(row['status'] == 'secure_training_count_link'
                             for row in edge_rows), 2)

    def test_training_points_must_reconstruct_each_count_cell_once(self):
        with self.assertRaisesRegex(ValueError, 'does not reconstruct'):
            make_ledger([self.key_a, self.key_b], [1, 2], self.points,
                        self.groups, self.edges)


if __name__ == '__main__':
    unittest.main()
