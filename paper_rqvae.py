from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Adagrad
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def build_paper_optimizer(model: nn.Module, learning_rate: float = 0.4) -> Adagrad:
    if learning_rate != 0.4:
        raise ValueError("paper-strict RQ-VAE learning rate must equal 0.4")
    return Adagrad(model.parameters(), lr=learning_rate, weight_decay=0.0)


def build_epoch_dataloader(
    dataset: Dataset,
    batch_size: int,
    generator: torch.Generator,
) -> DataLoader:
    if batch_size != 1024:
        raise ValueError("paper-strict RQ-VAE batch size must equal 1024")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        generator=generator,
    )


def steps_per_epoch(item_count: int, batch_size: int) -> int:
    if item_count <= 0 or batch_size <= 0:
        raise ValueError("item_count and batch_size must be positive")
    return math.ceil(item_count / batch_size)


def save_training_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    completed_epoch: int,
    optimizer_step: int,
    seed: int,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "completed_epoch": completed_epoch,
            "optimizer_step": optimizer_step,
            "seed": seed,
            "data_generator_state": generator.get_state(),
            "config": config,
        },
        path,
    )


def load_training_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
) -> dict[str, int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    generator.set_state(checkpoint["data_generator_state"])
    for layer in model.layers:
        layer.kmeans_initted = True
    return {
        "start_epoch": int(checkpoint["completed_epoch"]) + 1,
        "optimizer_step": int(checkpoint["optimizer_step"]),
        "seed": int(checkpoint["seed"]),
    }
