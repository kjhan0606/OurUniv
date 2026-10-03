import unittest

from cf4_r2_v6_multimember_graph_census import eligible_multilink_training_groups


class V6MultiMemberGraphCensusTests(unittest.TestCase):
    def test_eligibility_requires_training_members_and_an_FP_row(self):
        labels = ['A_simple', 'B_more_FP', 'C_anchor', 'D_heldout',
                  'E_buffer_member', 'F_no_FP']
        roles = [0, 0, 0, 1, 0, 0]
        row_count = [1, 2, 1, 1, 1, 0]
        anchors = [0, 0, 1, 0, 0, 0]
        members = {'A_simple': {10, 11}, 'B_more_FP': {12, 13},
                   'C_anchor': {14, 15}, 'D_heldout': {20, 21},
                   'E_buffer_member': {40, 41}, 'F_no_FP': {50, 51}}
        point_roles = {10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0,
                       20: 0, 21: 0, 40: 0, 41: 2, 50: 0, 51: 0}

        self.assertEqual(eligible_multilink_training_groups(
            labels, roles, row_count, anchors, members, point_roles),
            ['A_simple', 'B_more_FP', 'C_anchor'])

    def test_candidates_are_ranked_by_complexity_then_lexical_label(self):
        labels = ['Z_train', 'B_train', 'A_train']
        members = {'Z_train': {1, 2}, 'B_train': {3, 4}, 'A_train': {5, 6}}
        point_roles = {i: 0 for i in range(1, 7)}

        self.assertEqual(eligible_multilink_training_groups(
            labels, [0, 0, 0], [1, 1, 1], [0, 1, 0], members, point_roles),
            ['A_train', 'Z_train', 'B_train'])


if __name__ == '__main__':
    unittest.main()
