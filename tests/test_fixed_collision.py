import pytest
import torch

from modules.tokenizer.fixed_collision import (
    FixedCollisionOverflow,
    analyze_three_token_ids,
    build_fixed_four_token_ids,
)


def test_builds_unique_four_token_ids_in_corpus_order() -> None:
    three = torch.tensor(
        [[1, 2, 3], [1, 2, 3], [4, 5, 6], [1, 2, 3]],
        dtype=torch.long,
    )

    result = build_fixed_four_token_ids(three, collision_cardinality=256)

    assert result.four_token_ids.tolist() == [
        [1, 2, 3, 0],
        [1, 2, 3, 1],
        [4, 5, 6, 0],
        [1, 2, 3, 2],
    ]
    assert result.stats.unique_three_token_count == 2
    assert result.stats.max_bucket_size == 3
    assert result.stats.bucket_size_histogram == {1: 1, 3: 1}
    assert torch.unique(result.four_token_ids, dim=0).shape[0] == 4


def test_rejects_bucket_larger_than_fixed_cardinality() -> None:
    three = torch.zeros((257, 3), dtype=torch.long)

    with pytest.raises(FixedCollisionOverflow, match="257.*256"):
        build_fixed_four_token_ids(three, collision_cardinality=256)


def test_rejects_rq_token_outside_codebook() -> None:
    three = torch.tensor([[0, 0, 256]], dtype=torch.long)

    with pytest.raises(ValueError, match="RQ token must be in"):
        analyze_three_token_ids(three, rq_codebook_size=256)


def test_rejects_dynamic_collision_cardinality() -> None:
    three = torch.tensor([[0, 0, 0]], dtype=torch.long)

    with pytest.raises(ValueError, match="collision cardinality must equal 256"):
        build_fixed_four_token_ids(three, collision_cardinality=257)
