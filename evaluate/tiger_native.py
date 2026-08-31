"""CPU-safe helpers for the fork's native semantic-ID evaluator contract.

This module only maps generated complete semantic IDs to frozen catalog item IDs.
It does not implement the Temporal-v1 teacher-forced full-catalog scorer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

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


class _PaperCandidateDecoder(Protocol):
    """Minimum decoder interface required for candidate-only beam evaluation."""

    num_hierarchies: int
    decoder_mlp: Sequence
    decoder_head_mode: str

    def encoder_forward_pass(
        self,
        attention_mask: Tensor,
        input_ids: Tensor,
        user_id: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]: ...

    def decoder_forward_pass(
        self,
        future_ids: Tensor | None,
        encoder_output: Tensor,
        attention_mask_for_encoder: Tensor,
        use_cache: bool = False,
        future_ids_are_global: bool = False,
    ) -> Tensor: ...


def _stable_descending_indices(scores: Tensor, limit: int) -> Tensor:
    return torch.argsort(scores, dim=-1, descending=True, stable=True)[..., :limit]


@torch.no_grad()
def unconstrained_autoregressive_beam_search(
    model: _PaperCandidateDecoder,
    attention_mask: Tensor,
    input_ids: Tensor,
    *,
    beam_size: int,
    user_id: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Generate complete IDs without any catalog-prefix masking.

    This is a candidate evaluator only: it does not claim an exact unpublished
    TIGER decoding setup.  All positions are expanded over their complete head
    vocabulary, complete IDs are mapped after decoding, and invalid IDs remain
    observable for later filtering/accounting.
    """

    if beam_size <= 0:
        raise ValueError("beam_size must be positive")
    if model.num_hierarchies != 4:
        raise ValueError("unconstrained paper-candidate evaluation requires four tokens")

    encoder_output, encoder_mask = model.encoder_forward_pass(
        attention_mask=attention_mask,
        input_ids=input_ids,
        user_id=user_id,
    )
    batch_size = input_ids.shape[0]
    generated: Tensor | None = None
    beam_scores: Tensor | None = None

    for hierarchy in range(model.num_hierarchies):
        if generated is None:
            beam_count = 1
            current_encoder = encoder_output
            current_mask = encoder_mask
            decoder_ids = None
        else:
            beam_count = generated.shape[1]
            current_encoder = encoder_output.repeat_interleave(beam_count, dim=0)
            current_mask = encoder_mask.repeat_interleave(beam_count, dim=0)
            decoder_ids = generated.reshape(batch_size * beam_count, hierarchy)

        shared_vocab = getattr(model, "decoder_head_mode", "per_position_heads") == "shared_vocab"
        decoder_kwargs = {
            "future_ids": decoder_ids,
            "encoder_output": current_encoder,
            "attention_mask_for_encoder": current_mask,
            "use_cache": False,
        }
        if shared_vocab:
            decoder_kwargs["future_ids_are_global"] = True
        decoder_output = model.decoder_forward_pass(**decoder_kwargs)
        if hasattr(model, "decoder_logits"):
            logits = model.decoder_logits(
                decoder_output[:, -1, :], hierarchy, full_vocab=shared_vocab
            )
        else:
            logits = model.decoder_mlp[hierarchy](decoder_output[:, -1, :])
        token_log_probs = torch.log_softmax(logits, dim=-1).reshape(
            batch_size, beam_count, -1
        )
        if beam_scores is not None:
            token_log_probs = token_log_probs + beam_scores.unsqueeze(-1)

        token_cardinality = token_log_probs.shape[-1]
        flat_scores = token_log_probs.reshape(batch_size, beam_count * token_cardinality)
        next_beam_count = min(beam_size, flat_scores.shape[-1])
        selected_flat = _stable_descending_indices(flat_scores, next_beam_count)
        selected_scores = torch.gather(flat_scores, 1, selected_flat)
        parent_indices = selected_flat // token_cardinality
        next_tokens = selected_flat % token_cardinality

        if generated is None:
            generated = next_tokens.unsqueeze(-1)
        else:
            parent_codes = torch.gather(
                generated,
                1,
                parent_indices.unsqueeze(-1).expand(-1, -1, hierarchy),
            )
            generated = torch.cat([parent_codes, next_tokens.unsqueeze(-1)], dim=-1)
        beam_scores = selected_scores

    assert generated is not None
    assert beam_scores is not None
    return generated, beam_scores


@dataclass
class PaperCandidateTopKAccumulator:
    """Evaluate unconstrained four-token beams after complete-ID filtering.

    ``invalid_id_*`` records raw top-K beams, matching the paper's reported
    notion that a generated complete ID can be absent from the item catalog.
    The ``expanded_beam_*`` fields retain the corresponding counts before the
    top-K raw-beam cutoff.  Ranking metrics use distinct valid mapped items.
    """

    codebooks: Tensor
    top_k: int = 10
    ks: tuple[int, ...] = (1, 5, 10)
    raw_ks: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.codebooks.ndim != 2 or self.codebooks.shape[1] != 4:
            raise ValueError("paper-candidate evaluation requires [items, four tokens]")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        self.ks = tuple(self.ks)
        if not self.ks or any(k <= 0 for k in self.ks):
            raise ValueError("metric cutoffs must be positive")
        if max(self.ks) > self.top_k:
            raise ValueError("top_k must cover every requested metric cutoff")
        self.raw_ks = tuple(sorted(set(self.raw_ks or self.ks) | {self.top_k}))
        if any(k > self.top_k for k in self.raw_ks):
            raise ValueError("raw invalid cutoffs must not exceed top_k")
        self.lookup = build_item_lookup(self.codebooks)
        self.reset()

    def reset(self) -> None:
        self.total = 0
        self.raw_prediction_counts = {k: 0 for k in self.raw_ks}
        self.raw_invalid_id_counts = {k: 0 for k in self.raw_ks}
        self.expanded_prediction_count = 0
        self.expanded_invalid_id_count = 0
        self.underfilled_user_count = 0
        self.hits = {k: 0 for k in self.ks}
        self.ndcg = {k: 0.0 for k in self.ks}

    def accumulate(
        self,
        actual_semantic_ids: Tensor,
        generated_semantic_ids: Tensor,
        beam_scores: Tensor | None = None,
    ) -> None:
        if actual_semantic_ids.ndim != 2 or actual_semantic_ids.shape[1] != 4:
            raise ValueError("actual semantic IDs must have shape [batch, four tokens]")
        if generated_semantic_ids.ndim != 3 or generated_semantic_ids.shape[-1] != 4:
            raise ValueError("generated semantic IDs must have shape [batch, beams, four tokens]")
        if generated_semantic_ids.shape[0] != actual_semantic_ids.shape[0]:
            raise ValueError("actual and generated batch sizes must match")
        if beam_scores is not None and beam_scores.shape != generated_semantic_ids.shape[:2]:
            raise ValueError("beam_scores must have shape [batch, beams]")

        actual_rows = actual_semantic_ids.detach().cpu().tolist()
        generated_rows = generated_semantic_ids.detach().cpu().tolist()
        score_rows = beam_scores.detach().cpu().tolist() if beam_scores is not None else None
        required_valid_items = max(self.ks)

        for batch_index, (actual_code, beams) in enumerate(
            zip(actual_rows, generated_rows, strict=True)
        ):
            actual_item = self.lookup.get(tuple(int(token) for token in actual_code))
            if actual_item is None:
                raise ValueError("actual semantic IDs must resolve to frozen catalog items")
            if score_rows is None:
                ordered_indices = list(range(len(beams)))
            else:
                ordered_indices = sorted(
                    range(len(beams)),
                    key=lambda index: (-float(score_rows[batch_index][index]), index),
                )

            self.total += 1
            self.expanded_prediction_count += len(ordered_indices)
            invalid_by_rank: list[bool] = []
            ranked_items: list[int] = []
            seen_items: set[int] = set()
            for rank_index, beam_index in enumerate(ordered_indices):
                item_id = self.lookup.get(
                    tuple(int(token) for token in beams[beam_index])
                )
                if item_id is None:
                    self.expanded_invalid_id_count += 1
                    invalid_by_rank.append(True)
                    continue
                invalid_by_rank.append(False)
                if item_id not in seen_items:
                    ranked_items.append(item_id)
                    seen_items.add(item_id)

            for k in self.raw_ks:
                raw_count = min(k, len(invalid_by_rank))
                self.raw_prediction_counts[k] += raw_count
                self.raw_invalid_id_counts[k] += sum(invalid_by_rank[:raw_count])
            if len(ranked_items) < required_valid_items:
                self.underfilled_user_count += 1
            for k in self.ks:
                rank = next(
                    (index for index, item in enumerate(ranked_items[:k], start=1) if item == actual_item),
                    None,
                )
                if rank is not None:
                    self.hits[k] += 1
                    self.ndcg[k] += 1.0 / math.log2(rank + 1.0)

    def reduce(self) -> dict[str, float | int]:
        if self.total == 0:
            raise ValueError("cannot reduce paper-candidate metrics before accumulation")
        raw_top_k_predictions = self.raw_prediction_counts[self.top_k]
        raw_top_k_invalid = self.raw_invalid_id_counts[self.top_k]
        result: dict[str, float | int | dict[str, dict[str, float | int]]] = {
            "prediction_count": raw_top_k_predictions,
            "invalid_id_count": raw_top_k_invalid,
            "invalid_id_rate": raw_top_k_invalid / max(1, raw_top_k_predictions),
            "raw_invalid_by_k": {
                str(k): {
                    "prediction_count": self.raw_prediction_counts[k],
                    "invalid_id_count": self.raw_invalid_id_counts[k],
                    "invalid_id_rate": self.raw_invalid_id_counts[k]
                    / max(1, self.raw_prediction_counts[k]),
                }
                for k in self.raw_ks
            },
            "expanded_beam_prediction_count": self.expanded_prediction_count,
            "expanded_beam_invalid_id_count": self.expanded_invalid_id_count,
            "expanded_beam_invalid_id_rate": self.expanded_invalid_id_count
            / max(1, self.expanded_prediction_count),
            "underfilled_user_count": self.underfilled_user_count,
        }
        for k in self.ks:
            result[f"h@{k}"] = self.hits[k] / self.total
            result[f"ndcg@{k}"] = self.ndcg[k] / self.total
        return result
