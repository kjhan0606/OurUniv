import unittest

from cf4_r2_v6_active1414_association_reconciliation import reconcile_selected_rows


class V6Active1414AssociationReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.chosen = [
            ('G1', 0, 12, 3, 0),
            ('G2', 1, 13, 4, 1),
        ]
        self.fp_pgcs = [501, 502, 503]
        self.active_pgcs = [501, 502]
        self.groups = [
            dict(source_group_label='G1', FP_row_count='1', anchor_row_count='0',
                 secure_training_recno_count='1', ambiguous_edge_count='0',
                 category='one_linked_count_point', association_collision='False'),
            dict(source_group_label='G2', FP_row_count='1', anchor_row_count='0',
                 secure_training_recno_count='1', ambiguous_edge_count='1',
                 category='association_conflict_unresolved', association_collision='False'),
        ]
        self.edges = [
            dict(PGC='501', candidate_group_labels='G1', status='secure_training_count_link'),
            dict(PGC='502', candidate_group_labels='G2', status='secure_training_count_link'),
            dict(PGC='502', candidate_group_labels='G2', status='extended_review_candidate'),
        ]

    def test_identifies_ledger_unresolved_edges(self):
        rows = reconcile_selected_rows(self.chosen, self.fp_pgcs,
                                       self.active_pgcs, self.groups, self.edges)
        self.assertEqual([row['fp_pgc'] for row in rows], [501, 502])
        self.assertEqual([row['unresolved_for_conditional_use'] for row in rows],
                         [False, True])
        self.assertEqual(rows[1]['associated_edge_statuses'],
                         'extended_review_candidate;secure_training_count_link')

    def test_active_order_mismatch_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'order/identity mismatch'):
            reconcile_selected_rows(self.chosen, self.fp_pgcs, [502, 501],
                                    self.groups, self.edges)

    def test_missing_group_in_ledger_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'absent from ownership ledger'):
            reconcile_selected_rows([('G3', 2, 14, 5, 2)], self.fp_pgcs,
                                    [501], self.groups, self.edges)


if __name__ == '__main__':
    unittest.main()
