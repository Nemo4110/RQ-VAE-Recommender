import pytest
import torch

from modules.tokenizer.semids import generate_lsh_semantic_ids
from modules.tiger_policy import validate_full_semantic_ids
from modules.tiger_policy import validate_semantic_id_source


def test_lsh_semantic_ids_are_deterministic_and_in_range() -> None:
    vectors = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 1.0], [2.0, -3.0]])
    first = generate_lsh_semantic_ids(vectors, width=4, num_hyperplanes=3, seed=29)
    second = generate_lsh_semantic_ids(vectors, width=4, num_hyperplanes=3, seed=29)
    assert torch.equal(first, second)
    assert first.shape == (4, 4)
    assert int(first.min()) >= 0
    assert int(first.max()) < 8


def test_lsh_source_contract_rejects_hyperplane_mismatch() -> None:
    with pytest.raises(ValueError, match="lsh_num_hyperplanes"):
        validate_semantic_id_source(
            "lsh",
            "lsh",
            checkpoint_lsh_seed=29,
            requested_lsh_seed=29,
            checkpoint_lsh_num_hyperplanes=8,
            requested_lsh_num_hyperplanes=7,
        )


def test_lsh_full_id_summary_does_not_label_digit_as_suffix() -> None:
    codes = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    summary = validate_full_semantic_ids(codes, semantic_id_source="lsh")
    assert summary["full_id_unique"]
    assert summary["lsh_code_cardinality"] == 5
    assert summary["suffix_cardinality"] == 0
