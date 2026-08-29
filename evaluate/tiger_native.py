"""CPU-safe helpers for the fork's native semantic-ID evaluator contract.

This module only maps generated complete semantic IDs to frozen catalog item IDs.
It does not implement the Temporal-v1 teacher-forced full-catalog scorer.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def build_item_lookup(codebooks: Tensor) -> dict[tuple[int, ...], int]:
    """Build a deterministic complete-ID-to-item lookup for a frozen catalog."""

    if codebooks.ndim != 2 or codebooks.shape[1] < 1:
        raise ValueError("codebooks must have shape [items, token_width]")
    rows = [tuple(int(token) for token in row) for row in codebooks.tolist()]
    if len(set(rows)) != len(rows):
        raise ValueError("full semantic IDs must be unique for item-level evaluation")
    return {row: item_id for item_id, row in enumerate(rows)}


def map_semantic_ids_to_items(
    generated_semantic_ids: Tensor,
    codebooks: Tensor,
) -> tuple[Tensor, Tensor]:
    """Map ``[B, K, D]`` generated IDs to item IDs and invalid masks."""

    if generated_semantic_ids.ndim != 3:
        raise ValueError("generated semantic IDs must have shape [batch, top_k, tokens]")
    if generated_semantic_ids.shape[-1] != codebooks.shape[-1]:
        raise ValueError("generated and catalog token widths must match")
    lookup = build_item_lookup(codebooks)
    item_ids = torch.full(
        generated_semantic_ids.shape[:2],
        -1,
        dtype=torch.long,
        device=generated_semantic_ids.device,
    )
    for batch_index, row in enumerate(generated_semantic_ids.detach().cpu().tolist()):
        for rank, semantic_id in enumerate(row):
            item_id = lookup.get(tuple(int(token) for token in semantic_id))
            if item_id is not None:
                item_ids[batch_index, rank] = item_id
    return item_ids, item_ids.lt(0)


def item_level_metrics(
    actual_item_ids: Tensor,
    generated_item_ids: Tensor,
    invalid_mask: Tensor | None = None,
    ks: tuple[int, ...] = (5, 10),
) -> dict[str, float | int]:
    """Compute native item-level hit/NDCG metrics without teacher forcing."""

    if actual_item_ids.ndim != 1 or generated_item_ids.ndim != 2:
        raise ValueError(
            "actual IDs must be [batch] and generated IDs must be [batch, top_k]"
        )
    if actual_item_ids.shape[0] != generated_item_ids.shape[0]:
        raise ValueError("actual and generated batch sizes must match")
    if invalid_mask is None:
        invalid_mask = generated_item_ids.lt(0)
    if invalid_mask.shape != generated_item_ids.shape:
        raise ValueError("invalid_mask must match generated item IDs")
    valid_predictions = (~invalid_mask).sum().item()
    result: dict[str, float | int] = {
        "prediction_count": int(generated_item_ids.numel()),
        "invalid_id_count": int(invalid_mask.sum().item()),
        "invalid_id_rate": float(invalid_mask.float().mean().item()),
    }
    for k in ks:
        if k <= 0:
            raise ValueError("metric cutoffs must be positive")
        top = generated_item_ids[:, :k]
        matches = top.eq(actual_item_ids.unsqueeze(1)) & ~invalid_mask[:, :k]
        found = matches.any(dim=1)
        ranks = matches.float().argmax(dim=1) + 1
        result[f"h@{k}"] = float(found.float().mean().item())
        result[f"ndcg@{k}"] = float(
            torch.where(
                found,
                1.0 / torch.log2(ranks.float() + 1.0),
                torch.zeros_like(ranks.float()),
            )
            .mean()
            .item()
        )
    result["valid_prediction_count"] = int(valid_predictions)
    return result


class NativeItemTopKAccumulator:
    """Accumulate item-level metrics for complete generated semantic IDs."""

    def __init__(self, codebooks: Tensor, ks: tuple[int, ...] = (1, 5, 10)) -> None:
        self.codebooks = codebooks
        self.ks = tuple(ks)
        if not self.ks or any(k <= 0 for k in self.ks):
            raise ValueError("metric cutoffs must be positive")
        self.reset()

    def reset(self) -> None:
        self.total = 0
        self.prediction_count = 0
        self.invalid_id_count = 0
        self.hits = {k: 0 for k in self.ks}
        self.ndcg = {k: 0.0 for k in self.ks}

    def accumulate(self, actual_semantic_ids: Tensor, generated_semantic_ids: Tensor) -> None:
        if actual_semantic_ids.ndim != 2:
            raise ValueError("actual semantic IDs must have shape [batch, tokens]")
        generated_items, invalid = map_semantic_ids_to_items(
            generated_semantic_ids, self.codebooks
        )
        actual_items, actual_invalid = map_semantic_ids_to_items(
            actual_semantic_ids.unsqueeze(1), self.codebooks
        )
        if actual_invalid.any():
            raise ValueError("actual semantic IDs must resolve to catalog items")
        actual_items = actual_items[:, 0]
        self.prediction_count += int(generated_items.numel())
        for actual_item, ranked_items, ranked_invalid in zip(
            actual_items.tolist(), generated_items.tolist(), invalid.tolist(), strict=True
        ):
            self.total += 1
            self.invalid_id_count += sum(ranked_invalid)
            for k in self.ks:
                rank = next(
                    (
                        index
                        for index, (item, is_invalid) in enumerate(
                            zip(ranked_items[:k], ranked_invalid[:k], strict=True),
                            start=1,
                        )
                        if not is_invalid and item == actual_item
                    ),
                    None,
                )
                if rank is not None:
                    self.hits[k] += 1
                    self.ndcg[k] += 1.0 / math.log2(rank + 1.0)

    def reduce(self) -> dict[str, float | int]:
        if self.total == 0:
            raise ValueError("cannot reduce native item metrics before accumulation")
        result: dict[str, float | int] = {
            "invalid_id_count": self.invalid_id_count,
            "invalid_id_rate": self.invalid_id_count / max(1, self.prediction_count),
            "prediction_count": self.prediction_count,
        }
        for k in self.ks:
            result[f"h@{k}"] = self.hits[k] / self.total
            result[f"ndcg@{k}"] = self.ndcg[k] / self.total
        return result
