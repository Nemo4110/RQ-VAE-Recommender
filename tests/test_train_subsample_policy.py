from data.processed import build_subsample_sequence


def test_subsample_sequence_preserves_existing_validation_append_by_default() -> None:
    assert build_subsample_sequence([10, 11, 12], 13, include_future_item=True) == [10, 11, 12, 13]


def test_subsample_sequence_can_strictly_exclude_validation_target() -> None:
    assert build_subsample_sequence([10, 11, 12], 13, include_future_item=False) == [10, 11, 12]
