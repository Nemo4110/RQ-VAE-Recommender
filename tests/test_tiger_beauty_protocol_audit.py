import pytest

from audit_tiger_beauty_protocol import validation_target_probability


def test_validation_target_probability_matches_current_subsample_boundaries() -> None:
    assert validation_target_probability(3) == pytest.approx(37 / 38)
    assert validation_target_probability(4) == pytest.approx(18 / 19)
    assert validation_target_probability(20) < validation_target_probability(4)


def test_validation_target_probability_rejects_invalid_history_length() -> None:
    with pytest.raises(ValueError, match="at least three"):
        validation_target_probability(2)
