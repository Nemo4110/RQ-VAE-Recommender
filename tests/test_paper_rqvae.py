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
from train_rqvae_paper import build_paper_model
from train_rqvae_paper import checkpoint_path_for
from train_rqvae_paper import validate_paper_run



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


@pytest.mark.parametrize(
    ("run_mode", "epochs", "max_items", "resume_checkpoint"),
    [
        ("smoke", 2, 2048, None),
        ("bounded", 100, None, None),
        ("full", 20000, None, None),
        ("resume_probe", 3, 2048, "checkpoint.pt"),
    ],
)
def test_paper_run_matrix_accepts_only_planned_stages(
    run_mode: str,
    epochs: int,
    max_items: int | None,
    resume_checkpoint: str | None,
) -> None:
    validate_paper_run(
        run_mode=run_mode,
        epochs=epochs,
        max_items=max_items,
        batch_size=1024,
        learning_rate=0.4,
        num_workers=0,
        amp=False,
        resume_checkpoint=resume_checkpoint,
        vae_input_dim=768,
        vae_hidden_dims=[512, 256, 128],
        vae_embed_dim=32,
        vae_codebook_size=256,
        vae_n_layers=3,
        commitment_weight=0.25,
    )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("batch_size", 512, "batch size"),
        ("learning_rate", 0.1, "learning rate"),
        ("num_workers", 1, "num_workers"),
        ("amp", True, "amp"),
        ("vae_codebook_size", 257, "codebook size"),
        ("vae_n_layers", 4, "layers"),
    ],
)
def test_paper_run_rejects_protocol_drift(
    field: str,
    value: object,
    match: str,
) -> None:
    kwargs = {
        "run_mode": "smoke",
        "epochs": 2,
        "max_items": 2048,
        "batch_size": 1024,
        "learning_rate": 0.4,
        "num_workers": 0,
        "amp": False,
        "resume_checkpoint": None,
        "vae_input_dim": 768,
        "vae_hidden_dims": [512, 256, 128],
        "vae_embed_dim": 32,
        "vae_codebook_size": 256,
        "vae_n_layers": 3,
        "commitment_weight": 0.25,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=match):
        validate_paper_run(**kwargs)


def test_paper_model_matches_tiger_rqvae_architecture() -> None:
    model = build_paper_model(
        input_dim=768,
        hidden_dims=[512, 256, 128],
        embed_dim=32,
        codebook_size=256,
        n_layers=3,
        commitment_weight=0.25,
    )

    assert model.input_dim == 768
    assert model.hidden_dims == [512, 256, 128]
    assert model.embed_dim == 32
    assert model.codebook_size == 256
    assert model.n_layers == 3
    assert model.n_cat_feats == 0
    assert all(layer.n_embed == 256 for layer in model.layers)
    assert all(layer.forward_mode is QuantizeForwardMode.STE for layer in model.layers)
    assert all(layer.do_kmeans_init for layer in model.layers)


def test_checkpoint_path_contains_epoch_and_optimizer_step(tmp_path: Path) -> None:
    path = checkpoint_path_for(tmp_path, completed_epoch=2, optimizer_step=4)

    assert path == tmp_path / "checkpoint_epoch_00002_step_0000004.pt"
