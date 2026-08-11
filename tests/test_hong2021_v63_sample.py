from hong2021_v63_sample import ARMS, ENSEMBLE_SCHEMA, METHOD


def test_v63_sampling_identity() -> None:
    assert ENSEMBLE_SCHEMA == (
        "hong2021-v63-conditional-log-physical-moment-ensemble-v1"
    )
    assert METHOD.startswith("train_only_conditional_log_physical_moment")
    assert ARMS == (
        "bounded_query_local_mixture_copula",
        "rolled_parameter_control",
    )

