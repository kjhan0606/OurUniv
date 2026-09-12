from hong2021_v54_train_gate import RATIO_MAXIMUM, RATIO_MINIMUM


def test_frozen_train_mechanism_band() -> None:
    assert RATIO_MINIMUM == 2.0 / 3.0
    assert RATIO_MAXIMUM == 1.5
    assert RATIO_MINIMUM <= 1.0 <= RATIO_MAXIMUM
