import inspect
import math

import numpy as np

import cf4_aggregate_evidence_smc as smc_module
from cf4_aggregate_evidence_oracle import AggregateEvidenceControllerOracle
from cf4_aggregate_evidence_smc import (
    antipodal_vmf_logpdf,
    conditional_ess,
    conditional_parent_probabilities,
    genealogical_ess,
    initialize_particles,
    mh_log_acceptance,
    particle_ess,
    pool_parent_probabilities,
    propose_particle,
    replicate_parent_probability,
    run_smc_replicate,
    sample_antipodal_vmf_axis,
    select_temperature_increment,
    systematic_resampling,
    update_weights_and_normalizer,
)


def test_conditional_ess_and_endpoint_rule_for_null_likelihood():
    weights = np.full(16, 1.0 / 16.0)
    log_z = np.zeros(16)
    assert conditional_ess(weights, log_z, 0.0) == 16.0
    assert conditional_ess(weights, log_z, 1.0) == 16.0
    assert select_temperature_increment(0.0, weights, log_z) == 1.0


def test_temperature_bisection_returns_largest_certified_feasible_increment():
    weights = np.full(64, 1.0 / 64.0)
    log_z = np.linspace(-20.0, 20.0, 64)
    delta = select_temperature_increment(0.0, weights, log_z)
    target = 0.8 * 64
    assert 0.0 < delta < 1.0
    assert conditional_ess(weights, log_z, delta) >= target
    assert conditional_ess(weights, log_z, delta + 1.1e-10) < target


def test_weight_and_normalizer_update_matches_direct_arithmetic():
    weights = np.asarray([0.2, 0.3, 0.5])
    log_z = np.log(np.asarray([1.0, 2.0, 4.0]))
    updated, log_increment = update_weights_and_normalizer(
        weights, log_z, 0.5
    )
    direct = weights * np.exp(0.5 * log_z)
    expected_increment = direct.sum()
    np.testing.assert_allclose(
        updated, direct / expected_increment, rtol=0.0, atol=1e-15
    )
    assert abs(log_increment - math.log(expected_increment)) < 1e-15


class ZeroRng:
    def random(self):
        return 0.0


def test_systematic_resampling_uses_lowest_cdf_index_on_ties():
    selected = systematic_resampling(np.full(4, 0.25), ZeroRng())
    np.testing.assert_array_equal(selected, [0, 0, 1, 2])


def test_resampling_and_mh_spawn_streams_match_frozen_golden_values():
    weights = np.arange(1, 65, dtype=np.float64)
    weights /= np.sum(weights)
    resampling_rng = np.random.Generator(np.random.PCG64DXSM(
        np.random.SeedSequence(2026082301, spawn_key=(1, 0))
    ))
    selected = systematic_resampling(weights, resampling_rng)
    np.testing.assert_array_equal(
        selected[:20],
        [5, 9, 12, 14, 16, 18, 20, 21, 23, 24,
         25, 26, 28, 29, 30, 31, 32, 33, 34, 35],
    )
    np.testing.assert_array_equal(
        selected[-10:], [59, 59, 60, 60, 61, 61, 62, 62, 63, 63]
    )
    proposal = propose_particle(
        np.asarray([0.0, -6.0, 4.0]),
        np.asarray([1.0, 0.0, 0.0]),
        2026082301,
        0,
        0,
        0,
    )
    np.testing.assert_array_equal(proposal[0], [0.0, -6.0, 4.0])
    np.testing.assert_array_equal(
        proposal[1],
        [0.9926475824644964, -0.0869877722236861, 0.08416593438530241],
    )
    assert proposal[2:5] == ("axis_local", None, 0)
    assert float(proposal[5].random()) == 0.07913990096708112


def test_genealogical_ess_uses_terminal_initial_ancestor_fractions():
    assert genealogical_ess(np.arange(8), 8) == 8.0
    assert genealogical_ess(np.zeros(8, dtype=int), 8) == 1.0
    labels = np.asarray([0, 0, 1, 1, 2, 2, 3, 3])
    assert genealogical_ess(labels, 8) == 4.0


def test_particle_initialization_is_seed_reproducible_and_replicate_independent():
    q1, a1 = initialize_particles(2026082301, 32)
    q2, a2 = initialize_particles(2026082301, 32)
    q3, a3 = initialize_particles(2026082302, 32)
    np.testing.assert_array_equal(q1, q2)
    np.testing.assert_array_equal(a1, a2)
    assert not np.array_equal(q1, q3)
    assert not np.array_equal(a1, a3)
    np.testing.assert_allclose(np.linalg.norm(a1, axis=1), 1.0, atol=2e-16)
    maximum = np.argmax(np.abs(a1), axis=1)
    assert np.all(a1[np.arange(len(a1)), maximum] >= 0.0)


def test_frozen_mh_acceptance_has_the_correct_prior_cancellation():
    current = np.asarray([0.0, -6.0, 4.0])
    proposed = current + np.asarray([6.0, 0.0, 0.0])
    evidence_only = min(0.0, 0.7 * (2.0 - 1.0))
    assert mh_log_acceptance(
        "axis_local", current, proposed, 1.0, 2.0, 0.7
    ) == evidence_only
    assert mh_log_acceptance(
        "prior_independence", current, proposed, 1.0, 2.0, 0.7
    ) == evidence_only
    q_acceptance = mh_log_acceptance(
        "q_local", current, proposed, 1.0, 2.0, 0.7
    )
    assert q_acceptance < evidence_only
    assert mh_log_acceptance(
        "joint_local", current, proposed, 1.0, 2.0, 0.7
    ) == q_acceptance


def test_antipodal_vmf_scales_are_normalized_symmetric_and_have_exact_moment():
    direction = np.asarray([0.3, -0.4, 0.5])
    direction /= np.linalg.norm(direction)
    grid = np.linspace(-1.0, 1.0, 200001)
    axes = np.column_stack((
        np.sqrt(np.maximum(0.0, 1.0 - grid**2)),
        np.zeros_like(grid),
        grid,
    ))
    z_direction = np.asarray([0.0, 0.0, 1.0])
    for index, kappa in enumerate((100.0, 10.0, 1.0)):
        log_density = antipodal_vmf_logpdf(axes, z_direction, kappa)
        integral = 2.0 * math.pi * np.trapezoid(np.exp(log_density), grid)
        assert abs(integral - 1.0) < 1e-7
        np.testing.assert_allclose(
            antipodal_vmf_logpdf(axes[::1000], z_direction, kappa),
            antipodal_vmf_logpdf(-axes[::1000], z_direction, kappa),
            rtol=0.0,
            atol=2e-14,
        )
        rng = np.random.Generator(np.random.PCG64DXSM(
            np.random.SeedSequence(2026082305, spawn_key=(21, index))
        ))
        sample = np.stack([
            sample_antipodal_vmf_axis(rng, direction, kappa)
            for _ in range(20000)
        ])
        np.testing.assert_allclose(np.linalg.norm(sample, axis=1), 1.0, atol=3e-16)
        maximum = np.argmax(np.abs(sample), axis=1)
        assert np.all(sample[np.arange(len(sample)), maximum] >= 0.0)
        expected = 1.0 - math.tanh(0.5 * kappa) / kappa
        observed = float(np.mean(np.abs(sample @ direction)))
        assert abs(observed - expected) < 0.006


def null_parent_evaluator(parent_count):
    def evaluate(keys):
        return keys, np.zeros((len(keys), parent_count), dtype=np.float64)

    return evaluate


def test_null_likelihood_smc_jumps_to_one_and_is_reproducible():
    first_oracle = AggregateEvidenceControllerOracle(null_parent_evaluator(256))
    first = run_smc_replicate(
        2026082301,
        first_oracle,
        particle_count=64,
        maximum_temperature_stages=4,
    )
    second_oracle = AggregateEvidenceControllerOracle(null_parent_evaluator(256))
    second = run_smc_replicate(
        2026082301,
        second_oracle,
        particle_count=64,
        maximum_temperature_stages=4,
    )
    np.testing.assert_array_equal(first.beta_history, [0.0, 1.0])
    assert abs(first.log_normalizer) <= 1e-12
    assert first.genealogical_ess == 64.0
    assert first.resampling_ancestors == []
    assert abs(particle_ess(first.weights) - 64.0) < 2e-14
    assert len(first.move_history) == 1
    assert len(first.move_history[0]) == 4
    total_proposals = sum(
        sum(sweep["proposal_count"].values())
        for sweep in first.move_history[0]
    )
    assert total_proposals == 64 * 4
    for name in (
        "midpoint_mpc_h", "axis", "keys", "weights", "log_z_bar",
        "ancestor_labels", "beta_history", "conditional_ess_history",
        "particle_ess_history", "log_normalizer_increment",
    ):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))


def test_positive_remaining_beta_rejects_zero_increment(monkeypatch):
    oracle = AggregateEvidenceControllerOracle(null_parent_evaluator(256))
    monkeypatch.setattr(
        smc_module, "select_temperature_increment", lambda *args, **kwargs: 0.0
    )
    with np.testing.assert_raises_regex(RuntimeError, "requires positive delta"):
        run_smc_replicate(
            2026082301,
            oracle,
            particle_count=8,
            maximum_temperature_stages=1,
        )


def test_terminal_parent_reconstruction_and_replicate_pooling_identity():
    parent_log_z = np.log(np.asarray([
        [1.0, 3.0],
        [2.0, 2.0],
    ]))
    conditional = conditional_parent_probabilities(parent_log_z)
    np.testing.assert_allclose(conditional.sum(axis=1), 1.0, atol=1e-15)
    probability = replicate_parent_probability(
        np.asarray([0.25, 0.75]), parent_log_z
    )
    expected = 0.25 * np.asarray([0.25, 0.75]) + 0.75 * np.asarray([0.5, 0.5])
    np.testing.assert_allclose(probability, expected, atol=1e-15)
    replicate = np.asarray([probability, [0.8, 0.2]])
    pooled, log_i_bar = pool_parent_probabilities(
        np.log(np.asarray([2.0, 1.0])), replicate
    )
    direct = (2.0 * replicate[0] + replicate[1]) / 3.0
    np.testing.assert_allclose(pooled, direct, atol=1e-15)
    assert abs(log_i_bar - math.log(1.5)) < 1e-15
    with np.testing.assert_raises_regex(ValueError, "terminal weights"):
        replicate_parent_probability(np.asarray([0.25, np.nan]), parent_log_z)


def test_controller_api_cannot_receive_parent_or_cf4_fields():
    parameters = set(inspect.signature(run_smc_replicate).parameters)
    assert parameters == {
        "master_seed",
        "oracle",
        "particle_count",
        "target_cess_fraction",
        "resampling_ess_fraction",
        "maximum_temperature_stages",
        "sweeps_per_stage",
    }
    assert not any("parent" in name.lower() or "cf4" in name.lower()
                   for name in parameters)
