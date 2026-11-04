import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cf4_r2_four_chain_field_mcse import evaluate_four_chain_field_mcse
from cf4_r2_retained_field_mcse import RetainedFieldMCSEDiagnostics


def _field(value, shape=(2, 2, 2)):
    rho = np.full(shape, value, dtype=np.float64)
    velocity = np.full((3,) + shape, value * 10., dtype=np.float64)
    return dict(rho=rho, mean_velocity_km_s=velocity,
        velocity_valid=rho > 0)


def _write_chain(directory, index, train_keys):
    diag = RetainedFieldMCSEDiagnostics((2, 2, 2), train_keys=train_keys,
        batch_sizes=(2, 4), minimum_batches=2,
        count_grid_size=2, population_count=2)
    for transition in range(1, 9):
        value = float(index + transition // 2)
        diag.update(_field(value), transition=transition,
            accepted=(transition % 2 == 1))

    statistics_name = "field_mcse_sufficient_statistics_unassessed.npz"
    np.savez(directory / statistics_name, **diag.checkpoint_arrays())
    scale_records = {}
    for batch_size in diag.batch_sizes:
        result = diag.finalize_scale(batch_size)
        map_name = f"field_mcse_unassessed_batch_{batch_size}.npz"
        np.savez(directory / map_name,
            **{key: value for key, value in result.items()
                if isinstance(value, np.ndarray)})
        scale_records[str(batch_size)] = dict(
            status=result["status"], map_file=map_name,
            regional_summary=result["regional_summary"])

    field_summary = diag.summary()
    field_summary.update(sufficient_statistics_file=statistics_name,
        sufficient_statistics_raw_array_bytes_estimate=diag.estimated_accumulator_bytes,
        sufficient_statistics_file_status="UNASSESSED_CHAIN_STATISTICS_FOR_POSTPROCESSING",
        scales=scale_records)
    report = dict(status="N256_CHAIN_FINISHED_REQUIRES_DIAGNOSTICS",
        R2_complete=False, heldout_scored=False,
        initialization_policy="independent_prior", long_chain=True,
        source_commit="test-commit", target="conditional-v6 synthetic fixture",
        N=2, box_cMpc_h=384., observation_target_revision="test-revision",
        source_conditioning_rule="test-rule",
        field_mcse_contract=dict(batch_sizes=[2, 4], minimum_batches=2),
        chain_id=f"test-chain-{index}",
        chain_rng_design=dict(chain_index=index, master_seed=20261104,
            chain_count=4, bit_generator="PCG64", streams={
                "prior": dict(stream_id=0, spawn_key=[index, 0]),
                "audit": dict(stream_id=1, spawn_key=[index, 1]),
                "transition": dict(stream_id=2, spawn_key=[index, 2])}),
        field_mcse=field_summary)
    path = directory / "result.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


class FourChainFieldMCSEReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.train_keys = np.asarray([0, 15], dtype=np.int64)
        self.reports = []
        for index in range(4):
            directory = self.root / f"chain-{index}"
            directory.mkdir()
            self.reports.append(_write_chain(directory, index, self.train_keys))

    def tearDown(self):
        self.temp.cleanup()

    def test_pools_only_after_reporting_chain_disagreement(self):
        result = evaluate_four_chain_field_mcse(list(reversed(self.reports)),
            train_keys=self.train_keys)
        self.assertEqual(result["status"],
            "FOUR_CHAIN_FIELD_MCSE_DIAGNOSTIC_NOT_CONVERGENCE_EVIDENCE")
        self.assertFalse(result["R2_complete"])
        self.assertFalse(result["heldout_scored"])
        self.assertFalse(result["posterior_promoted"])
        self.assertEqual(result["chain_indices"], [0, 1, 2, 3])
        self.assertEqual(result["pooled_retained_states"], 32)
        self.assertEqual(result["pooled_accepted_transitions"], 16)
        self.assertIn("training_count_cells",
            result["cross_chain_field_disagreement"]["density"]["regional_summary"]
                ["mean_map_disagreement_over_within_sd"])
        self.assertEqual(result["pooled_field_mcse"]["4"]["status"],
            "FIELD_MCSE_AVAILABLE_NOT_CONVERGENCE_EVIDENCE")
        self.assertTrue(all(item["zip_member_crc_verified"]
            for item in result["statistics_files"]))
        self.assertTrue(all(item["selected_members_crc_verified"]
            for item in result["map_files"]))

    def test_refuses_wrong_training_keys(self):
        with self.assertRaisesRegex(ValueError, "training-key identity"):
            evaluate_four_chain_field_mcse(self.reports,
                train_keys=np.asarray([1, 15], dtype=np.int64))

    def test_refuses_duplicate_chain_identity_and_target_mismatch(self):
        report_path = self.reports[-1]
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["chain_id"] = "test-chain-0"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "chain-indexed streams"):
            evaluate_four_chain_field_mcse(self.reports,
                train_keys=self.train_keys)

        report["chain_id"] = "test-chain-3"
        report["target"] = "different target"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "contract mismatch: target"):
            evaluate_four_chain_field_mcse(self.reports,
                train_keys=self.train_keys)

    def test_refuses_duplicate_or_missing_artifacts(self):
        with self.assertRaisesRegex(ValueError, "four distinct"):
            evaluate_four_chain_field_mcse(self.reports[:3] + [self.reports[0]],
                train_keys=self.train_keys)
        stats = self.reports[0].parent / "field_mcse_sufficient_statistics_unassessed.npz"
        stats.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "statistics or map"):
            evaluate_four_chain_field_mcse(self.reports,
                train_keys=self.train_keys)


if __name__ == "__main__":
    unittest.main()
