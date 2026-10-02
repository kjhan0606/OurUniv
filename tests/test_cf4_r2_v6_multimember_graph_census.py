import unittest

from cf4_r2_v6_multimember_graph_census import eligible_multilink_training_groups


class V6MultiMemberGraphCensusTests(unittest.TestCase):
    def test_eligibility_requires_training_single_mark_unanchored_group(self):
        labels = ['A_train', 'B_heldout', 'C_multiple_marks', 'D_buffer_member']
        roles = [0, 1, 0, 0]
        row_count = [1, 1, 2, 1]
        anchors = [0, 0, 0, 0]
        members = {'A_train': {10, 11}, 'B_heldout': {20, 21},
                   'C_multiple_marks': {30, 31}, 'D_buffer_member': {40, 41}}
        point_roles = {10: 0, 11: 0, 20: 0, 21: 0, 30: 0, 31: 0,
                       40: 0, 41: 2}

        self.assertEqual(eligible_multilink_training_groups(
            labels, roles, row_count, anchors, members, point_roles), ['A_train'])

    def test_candidates_are_returned_in_lexical_order(self):
        labels = ['Z_train', 'A_train']
        members = {'Z_train': {1, 2}, 'A_train': {3, 4}}
        point_roles = {1: 0, 2: 0, 3: 0, 4: 0}

        self.assertEqual(eligible_multilink_training_groups(
            labels, [0, 0], [1, 1], [0, 0], members, point_roles),
            ['A_train', 'Z_train'])


if __name__ == '__main__':
    unittest.main()
