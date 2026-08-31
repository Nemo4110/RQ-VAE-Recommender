import pytest
import torch

from modules.tiger_policy import validate_full_semantic_ids
from modules.tiger_policy import validate_semantic_id_source
from modules.tokenizer.semids import generate_unique_random_semantic_ids


def test_random_semantic_ids_are_deterministic_unique_and_in_range() -> None:
    first = generate_unique_random_semantic_ids(
        item_count=100, width=4, cardinality=8, seed=29
    )
    second = generate_unique_random_semantic_ids(
        item_count=100, width=4, cardinality=8, seed=29
    )
    assert torch.equal(first, second)
    assert first.shape == (100, 4)
    assert int(first.min()) >= 0
    assert int(first.max()) < 8
    assert torch.unique(first, dim=0).shape[0] == 100


def test_random_semantic_ids_reject_capacity_overflow() -> None:
    with pytest.raises(ValueError, match="too small"):
        generate_unique_random_semantic_ids(item_count=17, width=2, cardinality=4, seed=0)


def test_random_id_summary_does_not_label_the_last_digit_as_a_collision_suffix() -> None:
    ids = generate_unique_random_semantic_ids(
        item_count=100, width=4, cardinality=8, seed=29
    )
    summary = validate_full_semantic_ids(ids, semantic_id_source="random")
    assert summary["full_id_unique"]
    assert summary["random_id_cardinality"] == 8
    assert summary["suffix_cardinality"] == 0
    assert summary["max_collision_bucket"] == 0


def test_random_checkpoint_source_contract_rejects_mapping_mismatch() -> None:
    with pytest.raises(ValueError, match="random_id_seed"):
        validate_semantic_id_source(
            "random",
            "random",
            checkpoint_random_seed=17,
            requested_random_seed=29,
            checkpoint_random_cardinality=255,
            requested_random_cardinality=255,
        )
    with pytest.raises(ValueError, match="semantic_id_source"):
        validate_semantic_id_source("rqvae", "random")
