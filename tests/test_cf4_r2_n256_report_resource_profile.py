import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cf4_r2_n256_report_resource_profile import (
    build_fixtures,
    finalize_profile,
    validate_synthetic_report,
)


class N256ReportResourceProfileTests(unittest.TestCase):
    def test_synthetic_fixture_runs_real_report_and_cleans_only_its_fixture(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training_split = root / "split.npz"
            train_keys = np.asarray([0, 15], dtype=np.int64)
            np.savez(training_split, train_keys=train_keys,
                heldout_payload=np.asarray(["must not be read"]))
            job_root = root / "450000"
            job_root.mkdir()
            with patch.dict(os.environ, {
                    "CF4_EXPECTED_COMMIT": "a" * 40,
                    "SLURM_JOB_ID": "450000"}):
                paths = build_fixtures(job_root, training_split,
                    grid_size=4, draws=16, batch_sizes=(4, 8),
                    minimum_batches=2, count_grid_size=2,
                    population_count=6)
                raw_report = job_root / "raw_report.json"
                subprocess.run([sys.executable, "-I", "-P",
                    str(ROOT / "scripts/cf4_r2_four_chain_field_mcse_report.py"),
                    *paths, "--training-split", str(training_split),
                    "--output", str(raw_report)], check=True,
                    capture_output=True, text=True)
                report = json.loads(raw_report.read_text(encoding="utf-8"))
                expected_bytes = validate_synthetic_report(report,
                    grid_size=4, draws=16, batch_sizes=(4, 8),
                    training_keys=train_keys, count_grid_size=2,
                    population_count=6)
                self.assertEqual(expected_bytes, 37 * 4**3 * 8 * 2)
                json.dumps(report, allow_nan=False)
                profile = finalize_profile(job_root, raw_report,
                    training_split, grid_size=4, draws=16,
                    batch_sizes=(4, 8), generation_seconds=1.2,
                    report_seconds=2.3, requested_mem_gib=28,
                    expected_peak_gib=21, count_grid_size=2,
                    population_count=6)

            self.assertEqual(profile["status"],
                "SYNTHETIC_N256_REPORT_RESOURCE_PROFILE_ONLY")
            self.assertTrue(profile["synthetic_fixture"])
            self.assertFalse(profile["posterior_promoted"])
            self.assertEqual(profile["fixture_cleanup_status"],
                "REMOVED_AFTER_VALIDATED_PROFILE")
            self.assertFalse((job_root / "fixtures").exists())
            self.assertFalse(raw_report.exists())
            saved = json.loads((job_root / "resource_profile.json")
                .read_text(encoding="utf-8"))
            self.assertTrue(saved["synthetic_fixture"])
            self.assertEqual(saved["integrity"]["pooled_retained_states"], 64)
            self.assertEqual(saved["measured_peak_headroom_status"],
                "PENDING_POST_JOB_SLURM_ACCOUNTING")


if __name__ == "__main__":
    unittest.main()
