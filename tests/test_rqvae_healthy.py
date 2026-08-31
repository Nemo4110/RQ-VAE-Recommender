from types import SimpleNamespace

import pytest
import torch
from torch import nn

from modules.rqvae_healthy import EmaCodebook
from modules.rqvae_healthy import append_collision_suffixes
from modules.rqvae_healthy import apply_kaiming_relu_initialization
from modules.rqvae_healthy import reset_dead_codes
from modules.rqvae_healthy import summarize_rqvae_health
from train_rqvae_healthy import _run_metadata
from train_rqvae_healthy import historical_epoch_permutation


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.n_embed = 3
        self.embedding = nn.Embedding(3, 2)


def test_reset_dead_codes_replaces_only_unused_entries() -> None:
    layer = _Layer()
    layer.embedding.weight.data.fill_(-1)
    reset_counts = reset_dead_codes(
        [layer],
        [torch.tensor([[1.0, 2.0], [3.0, 4.0]])],
        [torch.tensor([0, 0])],
        generator=torch.Generator().manual_seed(7),
    )
    assert reset_counts == [2]
    assert layer.embedding.weight[0].tolist() == [-1.0, -1.0]
    assert not torch.all(layer.embedding.weight[1:] == -1)


def test_ema_codebook_updates_codebook_without_embedding_gradients() -> None:
    layer = _Layer()
    layer.embedding.weight.data.zero_()
    ema = EmaCodebook([layer], decay=0.0)
    ema.update(
        [layer],
        [torch.tensor([[2.0, 4.0], [6.0, 8.0]])],
        [torch.tensor([1, 1])],
    )
    # EMA Laplace smoothing intentionally changes the exact count-normalized mean
    # by about 1e-4 for this tiny two-item fixture.
    assert torch.allclose(
        layer.embedding.weight[1], torch.tensor([4.0, 6.0]), atol=2e-4
    )
    assert torch.allclose(layer.embedding.weight[0], torch.zeros(2))


def test_ema_corpus_assignment_bootstrap_matches_codebook_mass() -> None:
    layer = _Layer()
    layer.embedding.weight.data.copy_(
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    )
    ema = EmaCodebook([layer], decay=0.99)
    ema.bootstrap_from_assignment_counts([layer], [torch.tensor([4.0, 2.0, 0.0])])
    assert torch.equal(ema.cluster_sizes[0], torch.tensor([4.0, 2.0, 0.0]))
    assert torch.equal(
        ema.embedding_averages[0],
        torch.tensor([[4.0, 8.0], [6.0, 8.0], [0.0, 0.0]]),
    )


def test_ema_unit_pseudocount_preserves_initial_codebook_scale() -> None:
    layer = _Layer()
    layer.embedding.weight.data.copy_(
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    )
    ema = EmaCodebook([layer], decay=0.99)
    ema.update(
        [layer],
        [torch.tensor([[2.0, 4.0], [6.0, 8.0]])],
        [torch.tensor([1, 1])],
    )
    assert torch.allclose(ema.cluster_sizes[0], torch.tensor([0.99, 1.01, 0.99]))
    assert torch.max(torch.abs(layer.embedding.weight)) < 10.0


def test_ema_checkpoint_state_rejects_mismatched_layer_shape() -> None:
    layer = _Layer()
    ema = EmaCodebook([layer], decay=0.9)
    state = ema.state_dict()
    state["cluster_sizes"] = [torch.ones(4)]
    with pytest.raises(ValueError, match="shape"):
        ema.load_state_dict(state)


def test_gin_binding_targets_imported_healthy_train_function() -> None:
    import gin
    from train_rqvae_healthy import train

    gin.clear_config()
    try:
        gin.parse_config("train_rqvae_healthy.train.arm_name = 'configured_arm'")
        assert gin.get_bindings(train)["arm_name"] == "configured_arm"
    finally:
        gin.clear_config()


def test_ema_codebook_state_round_trip_preserves_accumulators() -> None:
    layer = _Layer()
    ema = EmaCodebook([layer], decay=0.9)
    ema.update([layer], [torch.ones(2, 2)], [torch.tensor([0, 1])])
    restored = EmaCodebook([layer], decay=0.9)
    restored.load_state_dict(ema.state_dict())
    assert torch.equal(restored.cluster_sizes[0], ema.cluster_sizes[0])
    assert torch.equal(restored.embedding_averages[0], ema.embedding_averages[0])


def test_collision_suffixes_are_first_seen_and_full_ids_are_unique() -> None:
    semantic_ids = torch.tensor([[1, 2, 3], [1, 2, 3], [2, 2, 2], [1, 2, 3]])
    full_ids = append_collision_suffixes(semantic_ids, suffix_capacity=3)
    assert full_ids.tolist() == [[1, 2, 3, 0], [1, 2, 3, 1], [2, 2, 2, 0], [1, 2, 3, 2]]
    health = summarize_rqvae_health(
        semantic_ids, codebook_size=4, suffix_capacity=3, minimum_usage=0.25
    )
    assert health["max_collision_bucket"] == 3
    assert health["unique_full_id_count"] == 4
    assert health["full_id_unique"]
    assert health["collision_capacity_passed"]


def test_collision_gate_rejects_oversized_bucket() -> None:
    semantic_ids = torch.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    with pytest.raises(ValueError, match="exceeds"):
        append_collision_suffixes(semantic_ids, suffix_capacity=2)
    health = summarize_rqvae_health(
        semantic_ids, codebook_size=4, suffix_capacity=2, minimum_usage=0.25
    )
    assert health["max_collision_bucket"] == 3
    assert not health["collision_capacity_passed"]
    assert not health["full_id_unique"]
    assert not health["paper_gate_passed"]


def test_kaiming_relu_initialization_reinitializes_linear_weights_and_biases() -> None:
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    for module in model.modules():
        if isinstance(module, nn.Linear):
            module.weight.data.zero_()
            module.bias.data.fill_(1.0)
    assert apply_kaiming_relu_initialization(model) == 2
    for module in model.modules():
        if isinstance(module, nn.Linear):
            assert torch.count_nonzero(module.weight)
            assert torch.equal(module.bias, torch.zeros_like(module.bias))


def test_eager_loss_exposes_one_batch_vector_per_quantizer_layer() -> None:
    from modules.quantize import QuantizeForwardMode
    from modules.rqvae import RqVae
    from train_rqvae_healthy import eager_rqvae_loss

    model = RqVae(
        input_dim=4,
        embed_dim=2,
        hidden_dims=[3],
        codebook_size=3,
        codebook_kmeans_init=False,
        codebook_mode=QuantizeForwardMode.STE,
        n_layers=3,
        n_cat_features=0,
    )
    batch_size = 5
    loss, reconstruction_loss, rqvae_loss, residuals, semantic_ids = eager_rqvae_loss(
        model, torch.randn(batch_size, 4), gumbel_temperature=0.2
    )
    assert torch.isfinite(loss)
    assert torch.isfinite(reconstruction_loss)
    assert torch.isfinite(rqvae_loss)
    assert len(residuals) == 3
    assert len(semantic_ids) == 3
    assert all(residual.shape == (batch_size, 2) for residual in residuals)
    assert all(ids.shape == (batch_size,) for ids in semantic_ids)

def test_historical_epoch_permutation_matches_seeded_torch_randperm() -> None:
    expected = torch.randperm(11, generator=torch.Generator().manual_seed(30))
    assert torch.equal(historical_epoch_permutation(11, 29, 2), expected)
    with pytest.raises(ValueError, match="positive"):
        historical_epoch_permutation(0, 29, 1)


def test_reset_dead_codes_supports_without_replacement_donors() -> None:
    layer = _Layer()
    layer.embedding.weight.data.fill_(-1)
    residuals = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    reset_dead_codes(
        [layer],
        [residuals],
        [torch.tensor([0, 0, 0])],
        generator=torch.Generator().manual_seed(7),
        donor_sampling="without_replacement",
    )
    assert torch.unique(layer.embedding.weight[1:], dim=0).shape[0] == 2
    with pytest.raises(ValueError, match="donor_sampling"):
        reset_dead_codes(
            [layer], [residuals], [torch.tensor([0, 0, 0])], donor_sampling="invalid"
        )

def test_run_metadata_uses_configured_dataset_identity() -> None:
    model = SimpleNamespace(
        input_dim=768,
        embed_dim=32,
        hidden_dims=[512, 256, 128],
        codebook_size=256,
        n_layers=3,
    )
    metadata = _run_metadata(
        config={"use_ema_codebook": False, "ema_bootstrap_from_corpus": False},
        dataset_folder="dataset/amazon-p5-st5",
        dataset_split="beauty",
        dataset_item_count=12101,
        model=model,
    )
    assert metadata["dataset"] == {
        "name": "amazon-p5-st5",
        "root": "dataset/amazon-p5-st5",
        "split": "beauty",
        "item_count": 12101,
    }



def test_ema_zero_mass_initialization_matches_historical_e2_first_update() -> None:
    layer = _Layer()
    layer.embedding.weight.data.copy_(
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    )
    ema = EmaCodebook([layer], decay=0.5, initialization="zero_mass")
    assert torch.equal(ema.cluster_sizes[0], torch.zeros(3))
    assert torch.equal(ema.embedding_averages[0], layer.embedding.weight)
    ema.update([layer], [torch.tensor([[2.0, 4.0]])], [torch.tensor([1])])
    assert torch.allclose(
        layer.embedding.weight[1], torch.tensor([5.0, 8.0]), atol=4e-4
    )
    restored = EmaCodebook([layer], decay=0.5, initialization="zero_mass")
    restored.load_state_dict(ema.state_dict())
    assert torch.equal(restored.cluster_sizes[0], ema.cluster_sizes[0])
    with pytest.raises(ValueError, match="initialization"):
        EmaCodebook([layer], decay=0.5).load_state_dict(ema.state_dict())


def test_ema_rejects_unknown_initialization() -> None:
    with pytest.raises(ValueError, match="initialization"):
        EmaCodebook([_Layer()], decay=0.99, initialization="unknown")
