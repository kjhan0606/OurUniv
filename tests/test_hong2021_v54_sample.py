from hong2021_v54_sample import ARMS, ENSEMBLE_SCHEMA


def test_v54_sampling_identity() -> None:
    assert ENSEMBLE_SCHEMA.endswith("bounded-mixture-ensemble-v1")
    assert ARMS == ("bounded_query_local_mixture_copula", "rolled_parameter_control")
