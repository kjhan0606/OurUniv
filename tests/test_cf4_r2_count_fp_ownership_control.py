import unittest
from unittest.mock import call, patch

from cf4_r2_count_fp_ownership_control import (
    BASE, ROOT, SPLIT, support_components_by_scale, verify_source_commit,
    select_median_nonzero_offset_index,
)


class SourceCommitGuardTests(unittest.TestCase):
    def test_control_uses_the_graph_closed_v6_split(self):
        self.assertEqual(SPLIT, BASE/'r2_sky_closed_split_v6/split.npz')

    @patch('cf4_r2_count_fp_ownership_control.subprocess.check_output')
    def test_abbreviated_expected_revision_is_resolved_before_comparison(self, check_output):
        revision = '116018a2f1971a756a1bfb385ad22899814fba63'
        check_output.side_effect = [revision + '\n', revision + '\n']

        self.assertEqual(verify_source_commit('116018a'), revision)
        self.assertEqual(check_output.call_args_list, [
            call(['git', 'rev-parse', '--verify', '116018a^{commit}'],
                 cwd=ROOT, text=True),
            call(['git', 'rev-parse', '--verify', 'HEAD^{commit}'],
                 cwd=ROOT, text=True),
        ])

    @patch('cf4_r2_count_fp_ownership_control.subprocess.check_output')
    def test_mismatched_resolved_revision_fails_closed(self, check_output):
        check_output.side_effect = [
            '116018a2f1971a756a1bfb385ad22899814fba63\n',
            '0000000000000000000000000000000000000000\n',
        ]

        with self.assertRaisesRegex(RuntimeError, 'source commit mismatch'):
            verify_source_commit('116018a')


class SupportComponentTests(unittest.TestCase):
    def test_support_scale_values_are_population_pack_tuples(self):
        empty = dict(mask=[False], ids=[0], node=[0], bin=[0])
        active = dict(mask=[True], ids=[42], node=[3], bin=[2])
        packs = (empty, active, empty, empty, empty, empty)

        found = support_components_by_scale({-1.: packs, 0.: packs, 1.: packs}, 1)

        self.assertEqual(found, {-1.: {(42, 3, 2)},
                                  0.: {(42, 3, 2)},
                                  1.: {(42, 3, 2)}})

    def test_geometry_only_control_is_nearest_median_nonzero_radius_offset(self):
        index, median = select_median_nonzero_offset_index(
            [30, 10, 20, 40], [10., 10., 10., 10.], [10., 11., 13., 15.])
        self.assertEqual(index, 2)
        self.assertEqual(median, 3.)


if __name__ == '__main__':
    unittest.main()
