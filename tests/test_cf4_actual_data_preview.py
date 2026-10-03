import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from cf4_actual_data_preview import ROOT, read_plan, validate_counts, bound_arrays, predictive_products


class ActualDataTest(unittest.TestCase):
    def test_longer_plan_preserves_model_and_gates(self):
        old = read_plan(ROOT/"config/cf4_actual_data_corrected_v2.json")
        new = read_plan(ROOT/"config/cf4_actual_data_longer_v3.json")
        for key in ("data", "grid", "selection_correction", "model_decision", "assessment"):
            self.assertEqual(old[key], new[key])
        self.assertEqual(new["sampler"], {**old["sampler"], "draws_per_chain": 2048})
        self.assertEqual(new["input_root"], old["output_root"])
        self.assertNotEqual(new["input_root"], new["output_root"])

    def inputs(self):
        train = np.ones((6,2,2,2), dtype=np.int64)
        held = np.zeros_like(train); held[:,0,0,0] = 1
        return dict(counts_train=train, counts_holdout=held, counts_all=train+held,
                    raw_selection_exposure=np.ones_like(train, dtype=float)), dict(
                    training_counts=48, holdout_counts=6, retained_counts=54)

    def test_actual_split_and_support(self):
        arrays, expected = self.inputs()
        validate_counts(arrays, expected)
        arrays["raw_selection_exposure"][0,0,0,0] = 0
        with self.assertRaises(ValueError):
            validate_counts(arrays, expected)
        arrays, expected = self.inputs()
        arrays["counts_train"] = arrays["counts_train"].astype(float)
        with self.assertRaises(ValueError):
            validate_counts(arrays, expected)

    def test_binding_has_real_velocity_and_no_truth(self):
        arrays, _ = self.inputs()
        model = SimpleNamespace(response=arrays["raw_selection_exposure"])
        design = dict(holdout=np.array([False, True]), variance=np.ones(2))
        result = (model, design, None, None, {}, arrays["counts_train"], arrays["counts_holdout"], np.array([123.,-456.]), None)
        saved = bound_arrays(result)
        np.testing.assert_array_equal(saved["radial_observed"], [123.,-456.])
        self.assertFalse(any("truth" in key or "mock" in key for key in saved))

    def test_predictive_moments_not_truth_metrics(self):
        arrays, _ = self.inputs()
        model = SimpleNamespace(response=arrays["raw_selection_exposure"])
        design = dict(holdout=np.array([False, True]), variance=np.array([4.,9.]), pos=np.zeros((2,3)))
        lam = np.ones_like(model.response)*5
        # Two identical mean predictions: predictive velocity variance is measurement variance.
        with tempfile.TemporaryDirectory() as directory:
            report = predictive_products(Path(directory), model, design, arrays["counts_train"],
                arrays["counts_holdout"], np.array([2.,6.]), 2*lam, 2*lam**2, np.zeros(2), np.zeros(2), 2)
            with np.load(Path(directory)/"observation_predictions.npz", allow_pickle=False) as saved:
                np.testing.assert_allclose(saved["radial_standardized_residual"], [1.,2.])
                np.testing.assert_array_equal(saved["radial_posterior_variance"], np.zeros(2))
            self.assertFalse(any("truth" in k or "coverage" in k or "RMSE" in k for k in report))


if __name__ == "__main__":
    unittest.main()
