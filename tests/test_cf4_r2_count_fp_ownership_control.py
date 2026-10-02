import unittest
from unittest.mock import call, patch

from cf4_r2_count_fp_ownership_control import ROOT, verify_source_commit


class SourceCommitGuardTests(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
