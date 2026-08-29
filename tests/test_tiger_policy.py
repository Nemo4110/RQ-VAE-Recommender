from __future__ import annotations

import pytest
import torch

from data.schemas import TokenizedSeqBatch
from evaluate.tiger_native import build_item_lookup
from evaluate.tiger_native import NativeItemTopKAccumulator
from evaluate.tiger_native import item_level_metrics
from evaluate.tiger_native import map_semantic_ids_to_items
from modules.tiger_policy import DATASET_CAPPED_USER_BINS
from modules.tiger_policy import HISTORICAL_NATIVE
from modules.tiger_policy import MODULO_HASHED_BUCKET
from modules.tiger_policy import NATIVE_SEMANTIC_ID
from modules.tiger_policy import PAPER_FULL_ID
from modules.tiger_policy import TIGERPolicyConfig
from modules.tiger_policy import select_semantic_id_tokens
from modules.tiger_policy import token_cardinalities
from modules.tiger_policy import token_type_offsets
from modules.tiger_policy import user_bucket_indices
from modules.tiger_policy import validate_checkpoint_policy
from modules.tiger_policy import validate_full_semantic_ids


def test_paper_policy_requires_three_layers_and_two_thousand_user_bins() -> None:
    TIGERPolicyConfig(
        token_policy=PAPER_FULL_ID,
        paper_aligned=True,
        num_user_bins=2000,
        user_token_policy=MODULO_HASHED_BUCKET,
        user_bin_mode="paper_fixed",
        evaluator_policy=NATIVE_SEMANTIC_ID,
        rqvae_n_layers=3,
    ).validate()

    with pytest.raises(ValueError, match="num_user_bins=2000"):
        TIGERPolicyConfig(
            token_policy=PAPER_FULL_ID,
            paper_aligned=True,
            num_user_bins=None,
            user_bin_mode="paper_fixed",
            rqvae_n_layers=3,
        ).validate()

    with pytest.raises(ValueError, match="exactly three"):
        TIGERPolicyConfig(
            token_policy=PAPER_FULL_ID,
            paper_aligned=True,
            num_user_bins=2000,
            user_bin_mode="paper_fixed",
            rqvae_n_layers=4,
        ).validate()


def test_dataset_capped_user_bins_resolve_to_dataset_specific_upper_bound() -> None:
    policy = TIGERPolicyConfig(
        user_bin_mode=DATASET_CAPPED_USER_BINS,
        user_bin_cap=2000,
    )
    resolved = policy.for_dataset(731)
    assert resolved.effective_num_user_bins == 731
    assert resolved.metadata()["dataset_user_count"] == 731

    capped = TIGERPolicyConfig(
        token_policy=PAPER_FULL_ID,
        num_user_bins=1200,
        user_bin_mode=DATASET_CAPPED_USER_BINS,
        user_bin_cap=2000,
    ).for_dataset(731)
    assert capped.effective_num_user_bins == 731
    capped.validate()

    with pytest.raises(ValueError, match="paper-aligned mode"):
        TIGERPolicyConfig(
            paper_aligned=True,
            num_user_bins=731,
            user_bin_mode=DATASET_CAPPED_USER_BINS,
        ).validate()


def test_token_selection_is_explicit_for_flattened_item_groups() -> None:
    source = torch.tensor([[1, 2, 3, 0, 4, 5, 6, 1]])
    historical = select_semantic_id_tokens(
        source,
        n_layers=3,
        token_policy=HISTORICAL_NATIVE,
        source_item_width=4,
    )
    paper = select_semantic_id_tokens(
        source,
        n_layers=3,
        token_policy=PAPER_FULL_ID,
        source_item_width=4,
    )
    assert historical.tolist() == [[1, 2, 3, 4, 5, 6]]
    assert paper.tolist() == source.tolist()


def test_token_offsets_follow_per_position_cardinalities() -> None:
    assert token_type_offsets((256, 256, 256, 2)) == (0, 256, 512, 768)


def test_native_item_mapping_is_separate_from_teacher_forced_scoring() -> None:
    catalog = torch.tensor([[1, 2, 3, 0], [1, 2, 3, 1]], dtype=torch.long)
    generated = torch.tensor(
        [[[1, 2, 3, 1], [9, 9, 9, 9]], [[1, 2, 3, 1], [1, 2, 3, 0]]],
        dtype=torch.long,
    )
    assert build_item_lookup(catalog)[(1, 2, 3, 0)] == 0
    item_ids, invalid = map_semantic_ids_to_items(generated, catalog)
    assert item_ids.tolist() == [[1, -1], [1, 0]]
    assert invalid.tolist() == [[False, True], [False, False]]
    metrics = item_level_metrics(
        torch.tensor([1, 0]), item_ids, invalid_mask=invalid, ks=(1, 2)
    )
    assert metrics["invalid_id_count"] == 1
    assert metrics["h@1"] == 0.5
    assert metrics["h@2"] == 1.0
    accumulator = NativeItemTopKAccumulator(catalog, ks=(1, 2))
    accumulator.accumulate(
        torch.tensor([[1, 2, 3, 1], [1, 2, 3, 0]]), generated
    )
    accumulated = accumulator.reduce()
    assert accumulated["invalid_id_count"] == 1
    assert accumulated["h@1"] == 0.5
    assert accumulated["h@2"] == 1.0


def test_paper_cardinalities_keep_suffix_separate_from_rqvae_codebooks() -> None:
    codebooks = torch.tensor(
        [[1, 2, 3, 0], [1, 2, 3, 1], [1, 4, 5, 0]], dtype=torch.long
    )
    assert token_cardinalities(
        codebooks,
        n_layers=3,
        rqvae_codebook_size=256,
        token_policy=PAPER_FULL_ID,
    ) == (256, 256, 256, 2)
    assert token_cardinalities(
        codebooks[:, :3],
        n_layers=3,
        rqvae_codebook_size=256,
        token_policy=HISTORICAL_NATIVE,
    ) == (256, 256, 256)
    summary = validate_full_semantic_ids(codebooks)
    assert summary["full_id_unique"] is True
    assert summary["suffix_cardinality"] == 2


def test_modulo_user_bucket_is_stable_and_bounded() -> None:
    ids = torch.tensor([-1, 0, 2000, 4001])
    assert user_bucket_indices(ids, 2000).tolist() == [1999, 0, 0, 1]


def test_paper_checkpoint_requires_matching_policy_metadata() -> None:
    policy = TIGERPolicyConfig(
        token_policy=PAPER_FULL_ID,
        paper_aligned=True,
        num_user_bins=2000,
        user_bin_mode="paper_fixed",
    )
    with pytest.raises(ValueError, match="missing TIGER policy metadata"):
        validate_checkpoint_policy(None, policy)
    with pytest.raises(ValueError, match="token_policy"):
        validate_checkpoint_policy({"token_policy": HISTORICAL_NATIVE}, policy)


def _batch(width: int) -> TokenizedSeqBatch:
    return TokenizedSeqBatch(
        user_ids=torch.tensor([[7], [2007]], dtype=torch.long),
        sem_ids=torch.tensor(
            [[1, 2, 3, 0, 4, 5, 6, 1], [2, 3, 4, 0, 5, 6, 7, 1]],
            dtype=torch.long,
        ),
        sem_ids_fut=torch.tensor(
            [[4, 5, 6, 1], [5, 6, 7, 1]], dtype=torch.long
        ),
        seq_mask=torch.ones(2, 8, dtype=torch.bool),
        token_type_ids=torch.arange(width).repeat(2, 2),
        token_type_ids_fut=torch.arange(width).repeat(2, 1),
    )


def test_model_supports_full_id_policy_on_cpu() -> None:
    pytest.importorskip("transformers")
    from modules.model import EncoderDecoderRetrievalModel

    model = EncoderDecoderRetrievalModel(
        codebooks=torch.tensor(
            [[1, 2, 3, 0], [1, 2, 3, 1], [2, 3, 4, 0]], dtype=torch.long
        ),
        num_hierarchies=4,
        num_embeddings_per_hierarchy=256,
        token_cardinalities=(256, 256, 256, 2),
        token_policy=PAPER_FULL_ID,
        source_sem_ids_dim=4,
        t5_d_model=8,
        t5_num_heads=2,
        t5_d_ff=16,
        t5_num_layers=1,
        should_add_sep_token=False,
        num_user_bins=2000,
    )
    output = model(_batch(4))
    assert output.loss.ndim == 0
    assert len(output.loss_d) == 4
    assert model.item_sid_embedding_table.num_embeddings == 770
