import unittest

import numpy as np

from cf4_r2_independent_chain_start import (
    NoFinitePriorStart,
    PriorStartEvaluationError,
    ZeroCountLikelihoodStart,
    is_retryable_zero_count_likelihood,
    make_independent_chain_streams,
    select_prior_start,
)


class IndependentChainStartTests(unittest.TestCase):
    def test_retry_classification_is_exact_and_start_only(self):
        self.assertTrue(is_retryable_zero_count_likelihood(-np.inf,
            [-np.inf, -20.], order=2, gradient=False, retry_enabled=True))
        for score, parts, order, gradient, enabled in (
                (-np.inf, [-np.inf, np.inf], 2, False, True),
                (-np.inf, [0., -np.inf], 2, False, True),
                (-np.inf, [-np.inf, -20.], 1, False, True),
                (-np.inf, [-np.inf, -20.], 2, True, True),
                (-np.inf, [-np.inf, -20.], 2, False, False),
                (np.nan, [-np.inf, -20.], 2, False, True)):
            self.assertFalse(is_retryable_zero_count_likelihood(score, parts,
                order=order, gradient=gradient, retry_enabled=enabled))

    def test_streams_are_reproducible_and_purpose_separated(self):
        first, metadata = make_independent_chain_streams(20261104, 2)
        repeated, _ = make_independent_chain_streams(20261104, 2)
        other_chain, _ = make_independent_chain_streams(20261104, 3)
        self.assertEqual(metadata["chain_count"], 4)
        self.assertEqual(metadata["streams"]["prior"]["spawn_key"], [2, 0])
        samples = {name: rng.standard_normal(16)
                   for name, rng in first.items()}
        repeated_samples = {name: rng.standard_normal(16)
                            for name, rng in repeated.items()}
        other_samples = {name: rng.standard_normal(16)
                         for name, rng in other_chain.items()}
        for name in samples:
            np.testing.assert_array_equal(samples[name], repeated_samples[name])
            self.assertFalse(np.array_equal(samples[name], other_samples[name]))
        for left, right in (("prior", "audit"), ("prior", "transition"),
                            ("audit", "transition")):
            self.assertFalse(np.array_equal(samples[left], samples[right]))

    def test_only_exact_zero_count_likelihood_is_retried(self):
        rng = np.random.default_rng(7)
        seen = []

        def probe(state):
            seen.append(state.copy())
            if len(seen) < 3:
                raise ZeroCountLikelihoodStart(count_score=-np.inf,
                    raw_score=-12.5, reason="observed count has zero intensity")
            return float(np.sum(state**2) / 2)

        selected = select_prior_start(rng, 32, probe)
        self.assertEqual(len(seen), 3)
        self.assertEqual([row["status"] for row in selected.attempts],
                         ["ZERO_COUNT_LIKELIHOOD", "ZERO_COUNT_LIKELIHOOD", "FINITE"])
        self.assertEqual(selected.as_report()["selected_attempt"], 3)
        self.assertEqual(len({row["state_sha256"] for row in selected.attempts}), 3)
        np.testing.assert_array_equal(selected.state, seen[-1])

    def test_four_zero_count_starts_exhaust_frozen_cap(self):
        seen = []

        def probe(state):
            seen.append(state.copy())
            raise ZeroCountLikelihoodStart(count_score=-np.inf,
                raw_score=-1., reason="zero observed-count intensity")

        with self.assertRaises(NoFinitePriorStart) as caught:
            select_prior_start(np.random.default_rng(11), 8, probe)
        self.assertEqual(len(seen), 4)
        self.assertEqual(len(caught.exception.attempts), 4)
        self.assertEqual(caught.exception.as_report()["status"],
                         "NO_FINITE_PRIOR_START_WITHIN_RETRY_CAP")

    def test_numerical_or_support_error_aborts_without_retry(self):
        calls = []

        def probe(_state):
            calls.append(1)
            raise ValueError("support builder failed")

        with self.assertRaises(PriorStartEvaluationError) as caught:
            select_prior_start(np.random.default_rng(13), 8, probe)
        self.assertEqual(len(calls), 1)
        self.assertIn("support builder failed", caught.exception.attempts[0]["reason"])

    def test_nonfinite_energy_aborts_without_retry(self):
        calls = []

        def probe(_state):
            calls.append(1)
            return np.inf

        with self.assertRaises(PriorStartEvaluationError):
            select_prior_start(np.random.default_rng(17), 8, probe)
        self.assertEqual(len(calls), 1)

    def test_invalid_chain_identity_and_retry_cap_fail_closed(self):
        with self.assertRaises(ValueError):
            make_independent_chain_streams(1, 4)
        with self.assertRaises(ValueError):
            make_independent_chain_streams(1, 0, chain_count=3)
        with self.assertRaises(ValueError):
            select_prior_start(np.random.default_rng(1), 8, lambda _q: 1.,
                               max_attempts=5)


if __name__ == "__main__":
    unittest.main()
