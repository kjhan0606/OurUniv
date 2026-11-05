from hong2021_v83_development_gate import prospective_pass
from hong2021_v83_sample import expected_attrs


def test_prospective_pass_requires_both_blocks_global_and_numerics() -> None:
    assert prospective_pass(0.03, 0.03, 0.051, True)
    assert not prospective_pass(0.025, 0.5, 0.9, True)
    assert not prospective_pass(0.5, 0.025, 0.9, True)
    assert not prospective_pass(0.5, 0.5, 0.05, True)
    assert not prospective_pass(0.5, 0.5, 0.9, False)


def test_expected_attrs_distinguish_spatial_candidate_from_control() -> None:
    arguments = (830, "c" * 64, "d" * 64, "e" * 64, "f" * 40, "a" * 64)
    candidate = expected_attrs("candidate", *arguments)
    control = expected_attrs("control", *arguments)
    differing = {key for key in candidate if candidate[key] != control[key]}
    assert differing == {"sampler", "sampler_steps"}
    assert candidate["sampler_steps"] == 40
    assert control["sampler_steps"] == 0
