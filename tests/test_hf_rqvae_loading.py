from __future__ import annotations

import inspect

import pytest
import torch

from modules.rqvae import RqVae
from modules.tokenizer.semids import SemanticIdTokenizer


def build_model() -> RqVae:
    return RqVae(
        input_dim=8,
        embed_dim=4,
        hidden_dims=[6],
        codebook_size=5,
        codebook_kmeans_init=False,
        n_layers=3,
        n_cat_features=0,
    )


def tokenizer_kwargs() -> dict:
    return {
        "input_dim": 8,
        "output_dim": 4,
        "hidden_dims": [6],
        "codebook_size": 5,
        "n_layers": 3,
        "n_cat_feats": 0,
    }


def test_rejects_legacy_and_hub_sources_together(tmp_path) -> None:
    with pytest.raises(ValueError, match="at most one RQ-VAE source"):
        SemanticIdTokenizer(
            **tokenizer_kwargs(),
            rqvae_weights_path=str(tmp_path / "legacy.pt"),
            rqvae_hf_model_path=str(tmp_path / "hub"),
        )


def test_loads_local_hub_model_without_network(tmp_path) -> None:
    source = build_model().eval()
    source.save_pretrained(tmp_path)
    tokenizer = SemanticIdTokenizer(
        **tokenizer_kwargs(),
        rqvae_hf_model_path=str(tmp_path),
    )
    result = tokenizer.rq_vae.get_semantic_ids(torch.randn(2, 8))
    assert result.sem_ids.shape == (2, 3)
    assert tokenizer.rq_vae.training is False
    for key, expected in source.state_dict().items():
        torch.testing.assert_close(tokenizer.rq_vae.state_dict()[key], expected)


def test_rejects_hub_architecture_mismatch(tmp_path) -> None:
    build_model().save_pretrained(tmp_path)
    with pytest.raises(ValueError, match="architecture mismatch"):
        SemanticIdTokenizer(
            **{**tokenizer_kwargs(), "codebook_size": 6},
            rqvae_hf_model_path=str(tmp_path),
        )


def test_legacy_checkpoint_loading_remains_available(tmp_path) -> None:
    source = build_model().eval()
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"iter": 7, "model": source.state_dict()}, checkpoint)
    tokenizer = SemanticIdTokenizer(
        **tokenizer_kwargs(),
        rqvae_weights_path=str(checkpoint),
    )
    for key, expected in source.state_dict().items():
        torch.testing.assert_close(tokenizer.rq_vae.state_dict()[key], expected)


def test_decoder_train_exposes_hub_path() -> None:
    import train_decoder

    assert "pretrained_rqvae_hf_path" in inspect.signature(
        train_decoder.train
    ).parameters

