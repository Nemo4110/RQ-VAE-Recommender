from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Adagrad
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from modules.tokenizer.fixed_collision import FixedCollisionOverflow
from modules.tokenizer.fixed_collision import analyze_three_token_ids
from modules.tokenizer.fixed_collision import build_fixed_four_token_ids


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


def compute_corpus_diagnostics(
    three_token_ids: torch.Tensor,
    losses_finite: bool = True,
) -> dict[str, Any]:
    stats = analyze_three_token_ids(three_token_ids, rq_codebook_size=256)
    unique_four_token_count = 0
    collision_gate_passed = False
    try:
        fixed = build_fixed_four_token_ids(
            three_token_ids,
            collision_cardinality=256,
        )
        unique_four_token_count = int(fixed.four_token_ids.shape[0])
        collision_gate_passed = True
    except FixedCollisionOverflow:
        collision_gate_passed = False
    usage_gate_passed = all(value >= 0.80 for value in stats.codebook_usage)
    return {
        "item_count": stats.item_count,
        "unique_three_token_count": stats.unique_three_token_count,
        "unique_four_token_count": unique_four_token_count,
        "max_bucket_size": stats.max_bucket_size,
        "bucket_size_histogram": stats.bucket_size_histogram,
        "codebook_usage": list(stats.codebook_usage),
        "fixed_collision_cardinality": 256,
        "losses_finite": bool(losses_finite),
        "collision_gate_passed": collision_gate_passed,
        "usage_gate_passed": usage_gate_passed,
        "paper_gate_passed": (
            collision_gate_passed and usage_gate_passed and bool(losses_finite)
        ),
    }


def publish_final_checkpoint(
    checkpoint_path: Path,
    link_path: Path,
    *,
    diagnostics: dict[str, Any],
) -> None:
    if not diagnostics.get("paper_gate_passed", False):
        raise ValueError("paper gate must pass before publishing final checkpoint")
    if link_path.exists() or link_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {link_path}")
    link_path.symlink_to(checkpoint_path.resolve())
