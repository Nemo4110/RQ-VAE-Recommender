from types import SimpleNamespace

import pytest
import torch

from modules.tokenizer.fixed_collision import FixedCollisionOverflow
from modules.tokenizer.semids import SemanticIdTokenizer


def test_tokenizer_strict_mode_rejects_public_style_overflow(monkeypatch) -> None:
    tokenizer = SemanticIdTokenizer(
        input_dim=8,
        output_dim=4,
        hidden_dims=[6],
        codebook_size=256,
        n_layers=3,
        n_cat_feats=0,
        collision_token_cardinality=256,
    )
    three = torch.zeros((257, 3), dtype=torch.long)
    monkeypatch.setattr(
        tokenizer,
        "_precompute_three_token_ids",
        lambda movie_dataset: three,
    )

    with pytest.raises(FixedCollisionOverflow, match="257.*256"):
        tokenizer.precompute_corpus_ids(SimpleNamespace())
