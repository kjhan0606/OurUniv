import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cf4_bgc_fixed_design_smoke as smoke  # noqa: E402


def _prepared(vobs):
    rows = 4
    return {
        "raw_idx": np.arange(rows),
        "pgc": np.arange(10, 10 + rows),
        "cz": np.linspace(2000.0, 5000.0, rows),
        "pos": np.arange(rows * 3, dtype=float).reshape(rows, 3) + 1.0,
        "rhat": np.tile([1.0, 0.0, 0.0], (rows, 1)),
        "vobs": np.asarray(vobs),
        "sig_measure": np.full(rows, 100.0),
        "variance": np.full(rows, 10000.0),
        "likelihood_kind": np.full(rows, 2, dtype=np.int8),
        "B": np.arange(rows * 4, dtype=float).reshape(rows, 4),
        "q_std": np.array([150.0, 150.0, 150.0, 3.0]),
        "holdout": np.array([False, False, True, False]),
    }


def _tiny_design():
    return smoke.fixed_design_from_prepared(_prepared([1.0, 2.0, 3.0, 4.0]))


def _publication_result():
    return {
        "schema": "ouruniv-cf4-bgc-fixed-design-single-mock-smoke-result-v2",
        "status": "COMPLETE_IMPLEMENTATION_SMOKE_NO_SCIENCE_CLAIM",
        "implementation": {"commit": "a" * 40},
        "bin_manifest": {"manifest_body_sha256": "b" * 64},
        "selection_semantics": "observed_grouped_CF4_fixed_design_conditioned",
        "population_selection_mock": False,
        "population_selection_function_validation_performed": False,
        "observed_catalog_vobs_used_as_posterior_datum": False,
        "mock_datum_formula": "u_mock=A*s_truth+B*q_truth+epsilon",
        "development_truth_seed_consumed": smoke.TRUTH_SEED,
        "development_truth_seed_count_consumed": 1,
        "seeds": {
            "truth_white": smoke.TRUTH_SEED,
            "truth_nuisance": smoke.NUISANCE_TRUTH_SEED,
            "likelihood_noise": smoke.NOISE_SEED,
            "posterior_draws": list(smoke.POSTERIOR_DRAW_SEEDS),
            "preconditioner": smoke.PRECONDITIONER_SEED,
            "adjoint": smoke.ADJOINT_SEED,
        },
        "N32_canonical_independent_real_analysis_mode_count": 8538,
        "N32_non_nyquist_theta_analysis_mode_count": 8535,
        "global_merged_bin_availability": [
            {
                "merged_bin_index": index,
                "N32_canonical_independent_real_mode_count": 8538 if index == 0 else 0,
            }
            for index in range(33)
        ],
        "theta_global_merged_bin_availability": [
            {
                "merged_bin_index": index,
                "N32_non_nyquist_canonical_independent_real_mode_count": (
                    8535 if index == 0 else 0
                ),
            }
            for index in range(33)
        ],
        "growth_rate": 0.5,
        "delta_theta_normalization": {
            "stored_theta_semantics": (
                "reconstructed_from_stored_velocity_not_copied_from_delta"
            ),
            "non_nyquist_consistency_pass": True,
            "Nyquist_plane_modes_excluded_from_theta_metrics": True,
            "non_nyquist_relative_errors": [0.0] * 6,
        },
        "catalog_design": {"selected_rows": 3, "train_rows": 2, "holdout_rows": 1},
        "numerical_gates": {
            "all_pass": True,
            "delta_theta_non_nyquist_pass": True,
            "delta_theta_non_nyquist_relative_errors": [0.0] * 6,
        },
        "delta_metrics": {
            "posterior_mean_source": "explicit_analytic_posterior_mean",
            "mock_count": 1,
            "posterior_draw_count": 4,
        },
        "theta_metrics": {
            "posterior_mean_source": "explicit_analytic_posterior_mean",
            "mock_count": 1,
            "posterior_draw_count": 4,
        },
        "velocity_posterior_product": {
            "per_cell_full_3x3_covariance_reconstructable_from_draw_axis": True,
            "scalar_sigma_v_substitution_allowed": False,
        },
        **smoke.no_claim_policy(),
    }


def _publication_arrays():
    scalar = np.zeros((smoke.N,) * 3)
    vector = np.zeros((3,) + scalar.shape)
    return {
        "truth_white": scalar.copy(),
        "posterior_mean_white": scalar.copy(),
        "posterior_draws_white": np.zeros((4,) + scalar.shape),
        "truth_delta": scalar,
        "truth_theta": scalar.copy(),
        "posterior_mean_delta": scalar.copy(),
        "posterior_mean_theta": scalar.copy(),
        "posterior_delta": np.zeros((4,) + scalar.shape),
        "posterior_theta": np.zeros((4,) + scalar.shape),
        "truth_velocity": vector,
        "posterior_mean_velocity": vector.copy(),
        "posterior_draw_velocity": np.zeros((4,) + vector.shape),
        "truth_nuisance_q": np.zeros(4),
        "posterior_mean_nuisance_q": np.zeros(4),
        "posterior_draws_nuisance_q": np.zeros((4, 4)),
        "mock_datum": np.zeros(3),
        "truth_forward_radial_signal": np.zeros(3),
        "truth_velocity_CIC_radial_signal": np.zeros(3),
        "train_raw_idx": np.arange(2),
        "holdout_raw_idx": np.arange(2, 3),
    }


def _rewrite_published_result(output, mutate):
    result_path = output / "result.json"
    artifact_path = output / "manifest.json"
    complete_path = output / "COMPLETE"
    result = json.loads(result_path.read_text())
    mutate(result)
    result_payload = smoke.canonical_json_bytes(result)
    result_path.write_bytes(result_payload)
    artifact = json.loads(artifact_path.read_text())
    artifact["payloads"]["result.json"] = {
        "sha256": smoke._sha256(result_payload),
        "bytes": len(result_payload),
    }
    artifact_payload = smoke.canonical_json_bytes(artifact)
    artifact_path.write_bytes(artifact_payload)
    complete = json.loads(complete_path.read_text())
    complete["manifest_sha256"] = smoke._sha256(artifact_payload)
    complete_path.write_bytes(smoke.canonical_json_bytes(complete))


def test_frozen_configuration_provenance_and_seed_contract():
    args = smoke.frozen_args(ROOT / "data/cf4_clean.npz")
    assert (args.N, args.box_size) == (32, 384.0)
    assert (args.Om, args.Ob, args.h, args.A_s_1e9, args.ns) == (
        0.31,
        0.05,
        0.746,
        1.63,
        0.96,
    )
    assert (args.bgc_window, args.bgc_cz_min, args.bgc_cz_max) == (801, 1500.0, 18000.0)
    assert (args.bgc_pool_cz_min, args.bgc_pool_cz_max) == (500.0, 30000.0)
    assert (args.error_scale, args.sigma_nl, args.h0_prior, args.bulk_prior) == (
        0.9,
        0.0,
        3.0,
        150.0,
    )
    assert (args.holdout, args.split_seed, args.cg_tol, args.cg_maxiter) == (
        0.2,
        20260823,
        3.0e-5,
        500,
    )
    assert args.holdout_by_raw_index_hash is True
    assert args.precond_probes == 4
    streams = {
        smoke.TRUTH_SEED,
        smoke.NUISANCE_TRUTH_SEED,
        smoke.NOISE_SEED,
        smoke.PRECONDITIONER_SEED,
        smoke.ADJOINT_SEED,
        *smoke.POSTERIOR_DRAW_SEEDS,
    }
    assert len(streams) == 9
    assert smoke.TRUTH_SEED == 2026083000
    validation_seeds = set(range(2026083064, 2026083320))
    nontruth_streams = streams - {smoke.TRUTH_SEED}
    assert not (nontruth_streams & validation_seeds)
    assert not any(2026083000 <= seed < 2026090000 for seed in nontruth_streams)
    assert smoke.verify_frozen_provenance(ROOT / "data/cf4_clean.npz")


def test_fixed_design_discards_real_vobs_and_preserves_fixed_conditioning():
    first = smoke.fixed_design_from_prepared(_prepared([1.0, 2.0, 3.0, 4.0]))
    second = smoke.fixed_design_from_prepared(
        _prepared([9.0e99, -8.0e99, 7.0e99, -6.0e99])
    )
    assert "vobs" not in first
    assert first.keys() == second.keys()
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_mock_datum_uses_truth_nuisance_noise_and_is_deterministic():
    design = _tiny_design()

    def forward(field):
        return np.array([field.sum(), field[0].sum(), field[1].sum(), field[2].sum()])

    first = smoke.generate_mock_datum(forward, design, shape=(3, 3, 3))
    second = smoke.generate_mock_datum(forward, design, shape=(3, 3, 3))
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    expected = first["signal"] + design["B"] @ first["q_truth"] + first["epsilon"]
    np.testing.assert_allclose(first["u_mock"], expected)


def test_delta_and_velocity_divergence_kernel_away_from_nyquist_planes():
    rng = np.random.default_rng(4)
    white = rng.standard_normal((8, 8, 8))
    transfer = np.ones_like(white)
    transfer[0, 0, 0] = 0.0
    delta = smoke.white_to_delta(white, transfer)
    coordinate = np.arange(8) * 10.0
    x, y, z = np.meshgrid(coordinate, coordinate, coordinate, indexing="ij")
    delta = (
        np.cos(2.0 * np.pi * x / 80.0)
        + 0.3 * np.sin(4.0 * np.pi * y / 80.0)
        - 0.2 * np.cos(6.0 * np.pi * z / 80.0)
    )
    velocity = smoke.delta_to_velocity(delta, growth_rate=0.5, box_size=80.0)
    theta = smoke.velocity_to_normalized_divergence(
        velocity, growth_rate=0.5, box_size=80.0
    )
    frequency = 2.0 * np.pi * np.fft.fftfreq(8, d=10.0)
    kx, ky, kz = np.meshgrid(frequency, frequency, frequency, indexing="ij")
    divergence = np.fft.ifftn(
        1j
        * (
            kx * np.fft.fftn(velocity[0])
            + ky * np.fft.fftn(velocity[1])
            + kz * np.fft.fftn(velocity[2])
        )
    ).real
    np.testing.assert_allclose(-divergence / 50.0, delta, atol=1e-12)
    np.testing.assert_allclose(theta, delta, atol=1e-12)


def test_cic_truth_velocity_matches_equivalent_radial_forward():
    grid = 6
    box = 60.0
    growth = 0.55
    transfer = np.ones((grid,) * 3)
    transfer[0, 0, 0] = 0.0
    positions = np.array([[0.0, 0.0, 0.0], [15.0, 25.0, 35.0], [59.0, 1.0, 8.0]])
    radial = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    def forward(white):
        delta = smoke.white_to_delta(white, transfer)
        velocity = smoke.delta_to_velocity(delta, growth, box)
        return smoke.cic_sample_radial_velocity(velocity, positions, radial, box)

    design = _tiny_design()
    for key in ("raw_idx", "pgc", "cz", "sig_measure", "variance", "likelihood_kind", "holdout"):
        design[key] = design[key][:3]
    design["pos"] = positions
    design["rhat"] = radial
    design["B"] = design["B"][:3]
    mock = smoke.generate_mock_datum(forward, design, shape=(grid,) * 3)
    delta = smoke.white_to_delta(mock["s_truth"], transfer)
    stored = smoke.delta_to_velocity(delta, growth, box)
    sampled = smoke.cic_sample_radial_velocity(stored, positions, radial, box)
    np.testing.assert_allclose(sampled, mock["signal"], rtol=1e-13, atol=1e-13)


def test_manifest_global_plan_records_all_33_bins_and_missing_high_k():
    body, body_sha, _ = smoke.load_bin_manifest(
        ROOT / "config/cf4_kf_bin_manifest_v1.json"
    )
    plan = smoke.global_merged_mode_plan(body)
    assert len(plan["availability"]) == 33
    assert body_sha == "f78e0991efd928ebdd999c9a91856ccd71a296a335b813176a02de842449e0cf"
    counts = [
        row["N32_canonical_independent_real_mode_count"]
        for row in plan["availability"]
    ]
    assert any(count > 0 for count in counts)
    assert any(count == 0 for count in counts)
    assert sum(counts) == plan["canonical_independent_real_analysis_mode_count"]
    assert sum(counts) == 8538
    selected = set(
        zip(
            *[
                axis.tolist()
                for axis in np.unravel_index(
                    plan["flat_independent_field_indices"], (smoke.N,) * 3
                )
            ]
        )
    )
    signed = np.rint(np.fft.fftfreq(smoke.N) * smoke.N).astype(int)

    def signed_tuple(index):
        return tuple(int(signed[item]) for item in index)

    signed_selected = {signed_tuple(index) for index in selected}
    for vector in signed_selected:
        conjugate = tuple(
            -smoke.N // 2 if -component == smoke.N // 2 else -component
            for component in vector
        )
        if conjugate != vector:
            assert conjugate not in signed_selected
    assert plan["availability"][-1]["status"] == (
        "NOT_EVALUATED_NO_N32_CANONICAL_INDEPENDENT_REAL_MODES"
    )


def test_one_mock_metrics_are_strictly_fail_closed_and_delta_theta_match():
    body, body_sha, _ = smoke.load_bin_manifest(
        ROOT / "config/cf4_kf_bin_manifest_v1.json"
    )
    plan = smoke.global_merged_mode_plan(body, grid_size=8)
    rng = np.random.default_rng(12)
    truth = rng.standard_normal((8, 8, 8))
    draws = truth[None, ...] + rng.standard_normal((4, 8, 8, 8)) * 0.1
    analytic_mean = truth * 0.75
    transfer = np.ones_like(truth)
    transfer[0, 0, 0] = 0.0
    delta, theta, fields = smoke.evaluate_delta_theta_metrics(
        truth, draws, analytic_mean, transfer, 0.5, plan, body_sha
    )
    assert delta["mock_count"] == theta["mock_count"] == 1
    assert not any(delta["strict_gate_intersection_with_geometry"])
    assert not any(theta["strict_gate_intersection_with_geometry"])
    assert delta["development_science_metric_allowed"] is False
    assert delta["posterior_mean_source"] == "explicit_analytic_posterior_mean"
    assert theta["posterior_mean_source"] == "explicit_analytic_posterior_mean"
    np.testing.assert_allclose(delta["response"], [0.75] * len(delta["response"]))
    np.testing.assert_array_equal(
        fields["truth_theta"],
        smoke.velocity_to_normalized_divergence(fields["truth_velocity"], 0.5),
    )
    assert max(fields["delta_theta_non_nyquist_relative_errors"]) < 1.0e-12
    assert fields["theta_non_nyquist_analysis_mode_count"] < plan[
        "canonical_independent_real_analysis_mode_count"
    ]


def test_numerical_threshold_edges_and_no_claim_policy():
    gate = smoke._numerical_gate(5.0e-5, 1.0e-4, [1.0e-4] * 4)
    assert gate["all_pass"] is True
    assert smoke._numerical_gate(5.00001e-5, 1.0e-4, [1.0e-4] * 4)["all_pass"] is False
    policy = smoke.no_claim_policy()
    assert policy["development_64_mock_execution_performed"] is False
    assert policy["untouched_validation_256_mock_execution_performed"] is False
    assert policy["frontier_evaluated"] is False
    assert policy["science_metric_or_claim_allowed"] is False


def test_deterministic_npz_is_key_order_invariant():
    first = smoke.deterministic_npz_bytes({"b": np.arange(4), "a": np.eye(2)})
    second = smoke.deterministic_npz_bytes({"a": np.eye(2), "b": np.arange(4)})
    assert first == second


def test_publish_validate_refuses_overwrite_and_preserves_vector_ensemble(tmp_path):
    output = tmp_path / "smoke"
    result = _publication_result()
    arrays = _publication_arrays()
    smoke.publish(output, result, arrays)
    assert {path.name for path in output.iterdir()} == smoke.EXPECTED_OUTPUT_FILES
    assert smoke.validate_output(output)["status"] == "PASS"
    with np.load(output / "fields.npz", allow_pickle=False) as fields:
        assert fields["posterior_draw_velocity"].shape == (4, 3, 32, 32, 32)
        assert not any("sigma_v" in key for key in fields.files)
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        smoke.publish(output, result, arrays)
    assert not (tmp_path / ".smoke.staging").exists()


def test_publish_refuses_preexisting_sibling_staging(tmp_path):
    stage = tmp_path / ".smoke.staging"
    stage.mkdir()
    marker = stage / "owned-by-another-process"
    marker.write_text("preserve")
    with pytest.raises(FileExistsError, match="existing staging"):
        smoke.publish(tmp_path / "smoke", _publication_result(), _publication_arrays())
    assert marker.read_text() == "preserve"


def test_validate_rejects_scalar_velocity_substitution(tmp_path):
    output = tmp_path / "bad-smoke"
    arrays = _publication_arrays()
    arrays["sigma_v"] = np.ones((smoke.N,) * 3)
    smoke.publish(output, _publication_result(), arrays)
    with pytest.raises(smoke.SmokeError, match="scalar sigma_v"):
        smoke.validate_output(output)


def test_validate_tolerates_last_bit_roundoff_in_recorded_theta_error(tmp_path):
    output = tmp_path / "portable-smoke"
    smoke.publish(output, _publication_result(), _publication_arrays())

    def perturb(result):
        result["delta_theta_normalization"]["non_nyquist_relative_errors"][0] = 1e-31
        result["numerical_gates"]["delta_theta_non_nyquist_relative_errors"][0] = 1e-31

    _rewrite_published_result(output, perturb)
    assert smoke.validate_output(output)["status"] == "PASS"


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda result: result.__setitem__("selection_semantics", "wrong"),
            "selection semantics",
        ),
        (
            lambda result: result.__setitem__("population_selection_mock", True),
            "population-selection mock",
        ),
        (
            lambda result: result.__setitem__(
                "observed_catalog_vobs_used_as_posterior_datum", True
            ),
            "observed vobs",
        ),
        (
            lambda result: result["numerical_gates"].__setitem__("all_pass", False),
            "numerical gates",
        ),
        (
            lambda result: result.__setitem__("frontier_evaluated", True),
            "no-claim flag",
        ),
    ],
)
def test_validate_rejects_semantic_gate_and_claim_tampering(
    tmp_path, mutate, match
):
    output = tmp_path / "tampered-smoke"
    smoke.publish(output, _publication_result(), _publication_arrays())
    _rewrite_published_result(output, mutate)
    with pytest.raises(smoke.SmokeError, match=match):
        smoke.validate_output(output)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda arrays: arrays.pop("posterior_draws_white"), "posterior_draws_white"),
        (
            lambda arrays: arrays["posterior_delta"].__setitem__((0, 0, 0, 0), np.nan),
            "nonfinite: posterior_delta",
        ),
        (
            lambda arrays: arrays["posterior_theta"].__setitem__((0, 0, 0, 0), 1.0),
            "stored theta is not reconstructed from velocity",
        ),
    ],
)
def test_validate_rejects_missing_nonfinite_or_inconsistent_ensembles(
    tmp_path, mutation, match
):
    arrays = _publication_arrays()
    mutation(arrays)
    output = tmp_path / "bad-ensemble"
    smoke.publish(output, _publication_result(), arrays)
    with pytest.raises(smoke.SmokeError, match=match):
        smoke.validate_output(output)


def test_validate_rejects_artifact_and_complete_schema_tampering(tmp_path):
    output = tmp_path / "schema-smoke"
    smoke.publish(output, _publication_result(), _publication_arrays())
    artifact_path = output / "manifest.json"
    complete_path = output / "COMPLETE"
    artifact = json.loads(artifact_path.read_text())
    artifact["schema"] = "wrong"
    artifact_payload = smoke.canonical_json_bytes(artifact)
    artifact_path.write_bytes(artifact_payload)
    complete = json.loads(complete_path.read_text())
    complete["manifest_sha256"] = smoke._sha256(artifact_payload)
    complete_path.write_bytes(smoke.canonical_json_bytes(complete))
    with pytest.raises(smoke.SmokeError, match="artifact manifest schema"):
        smoke.validate_output(output)
    artifact["schema"] = (
        "ouruniv-cf4-bgc-fixed-design-single-mock-smoke-artifact-manifest-v2"
    )
    artifact_payload = smoke.canonical_json_bytes(artifact)
    artifact_path.write_bytes(artifact_payload)
    complete["manifest_sha256"] = smoke._sha256(artifact_payload)
    complete["schema"] = "wrong"
    complete_path.write_bytes(smoke.canonical_json_bytes(complete))
    with pytest.raises(smoke.SmokeError, match="COMPLETE schema"):
        smoke.validate_output(output)


def test_failed_staged_publication_removes_only_owned_new_directory(
    tmp_path, monkeypatch
):
    output = tmp_path / "failed-smoke"
    original = smoke._write_exclusive
    calls = 0

    def fail_second(path, payload):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic publication failure")
        original(path, payload)

    monkeypatch.setattr(smoke, "_write_exclusive", fail_second)
    with pytest.raises(OSError, match="synthetic publication failure"):
        smoke.publish(output, _publication_result(), _publication_arrays())
    assert not output.exists()
    assert not (tmp_path / ".failed-smoke.staging").exists()


def test_cli_source_contains_no_run_side_effect_or_real_vobs_datum():
    source = (ROOT / "src/cf4_bgc_fixed_design_smoke.py").read_text()
    assert "linear.prepare_bgc_catalog" in source
    assert 'prepared["vobs"]' not in source
    assert 'mock["u_mock"][train]' in source
    assert 'design["vobs"]' not in source
    assert "posterior_draw_velocity" in source
    assert "truth_radial_forward_CIC_relative_error" in source
    assert "os.link" not in source
    assert "renameat2" not in source
    assert "os.rename(stage, output)" in source
