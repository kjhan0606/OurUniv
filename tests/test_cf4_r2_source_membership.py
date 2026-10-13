import unittest

from cf4_r2_source_membership import membership_state


class SourceMembershipTests(unittest.TestCase):
    def test_missing_is_not_a_bad_distance_or_physical_isolation(self):
        self.assertEqual(membership_state(0, 1), 'source_ungrouped_catalogue_absent')
        self.assertEqual(membership_state(0, 1, 0, 1), 'source_ungrouped_catalogue_present')
        self.assertEqual(membership_state(9, 4), 'unresolved_missing_group_member')

    def test_conflicts_never_fall_back_to_singleton(self):
        self.assertEqual(membership_state(9, 4, 9, 4), 'source_grouped_catalogue_present')
        self.assertEqual(membership_state(9, 4, 8, 4), 'unresolved_membership_conflict')
        self.assertEqual(membership_state(9, 4, 9, 3), 'unresolved_membership_conflict')
        with self.assertRaises(ValueError):
            membership_state(0, 0)


if __name__ == '__main__':
    unittest.main()
