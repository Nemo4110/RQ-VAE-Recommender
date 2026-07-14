from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

from modules.quantize import QuantizeForwardMode
from modules.rqvae import RqVae
from paper_rqvae import (
    build_epoch_dataloader,
    build_paper_optimizer,
    load_training_checkpoint,
    save_training_checkpoint,
    steps_per_epoch,
)


def build_model() -> RqVae:
    return RqVae(
        input_dim=8,
        embed_dim=4,
        hidden_dims=[6],
        codebook_size=256,
        codebook_kmeans_init=True,
        codebook_mode=QuantizeForwardMode.STE,
        n_layers=3,
        commitment_weight=0.25,
        n_cat_features=0,
    )


def test_optimizer_is_paper_adagrad() -> None:
    optimizer = build_paper_optimizer(build_model(), learning_rate=0.4)

    assert isinstance(optimizer, torch.optim.Adagrad)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.4)
    assert optimizer.param_groups[0]["weight_decay"] == 0


def test_epoch_loader_keeps_every_item_and_last_partial_batch() -> None:
    dataset = TensorDataset(torch.arange(2050))
    generator = torch.Generator().manual_seed(7)
    loader = build_epoch_dataloader(dataset, batch_size=1024, generator=generator)

    batches = list(loader)
    observed = torch.cat([batch[0] for batch in batches])
    assert len(batches) == 3
    assert sorted(observed.tolist()) == list(range(2050))


def test_literal_beauty_epoch_has_twelve_steps() -> None:
    assert steps_per_epoch(item_count=12101, batch_size=1024) == 12


def test_first_forward_initializes_each_codebook_only_once(monkeypatch) -> None:
    model = build_model().train()
    calls = [0, 0, 0]
    for index, layer in enumerate(model.layers):
        original = layer._kmeans_init

        def wrapped(x, *, _index=index, _original=original):
            calls[_index] += 1
            return _original(x)

        monkeypatch.setattr(layer, "_kmeans_init", wrapped)

    model.get_semantic_ids(torch.randn(1024, 8))
    model.get_semantic_ids(torch.randn(1024, 8))
    assert calls == [1, 1, 1]


def test_checkpoint_restores_generator_optimizer_and_kmeans_flags(
    tmp_path: Path,
) -> None:
    model = build_model()
    model.get_semantic_ids(torch.randn(1024, 8))
    optimizer = build_paper_optimizer(model, learning_rate=0.4)
    generator = torch.Generator().manual_seed(19)
    torch.randperm(20, generator=generator)
    expected_generator_state = generator.get_state().clone()
    checkpoint = tmp_path / "checkpoint.pt"

    save_training_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        generator=generator,
        completed_epoch=7,
        optimizer_step=84,
        seed=19,
        config={"epochs": 20000},
    )

    restored_model = build_model()
    restored_optimizer = build_paper_optimizer(restored_model, learning_rate=0.4)
    restored_generator = torch.Generator()
    state = load_training_checkpoint(
        checkpoint,
        model=restored_model,
        optimizer=restored_optimizer,
        generator=restored_generator,
    )

    assert state == {"start_epoch": 8, "optimizer_step": 84, "seed": 19}
    assert torch.equal(restored_generator.get_state(), expected_generator_state)
    assert all(layer.kmeans_initted for layer in restored_model.layers)
