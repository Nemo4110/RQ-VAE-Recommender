from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import torch
from torch import Tensor


class FixedCollisionOverflow(ValueError):
    pass


@dataclass(frozen=True)
class CollisionStats:
    item_count: int
    unique_three_token_count: int
    max_bucket_size: int
    bucket_size_histogram: dict[int, int]
    codebook_usage: tuple[float, float, float]


@dataclass(frozen=True)
class FixedCollisionResult:
    four_token_ids: Tensor
    stats: CollisionStats


def analyze_three_token_ids(
    three_token_ids: Tensor,
    rq_codebook_size: int = 256,
) -> CollisionStats:
    codes = three_token_ids.detach().cpu().long()
    if codes.ndim != 2 or codes.shape[1] != 3:
        raise ValueError("RQ semantic IDs must have shape [items, 3]")
    if codes.shape[0] == 0:
        raise ValueError("RQ semantic IDs must contain at least one item")
    if rq_codebook_size != 256:
        raise ValueError("paper-strict RQ codebook size must equal 256")
    if int(codes.min().item()) < 0 or int(codes.max().item()) >= rq_codebook_size:
        raise ValueError("RQ token must be in [0, 255]")

    _, counts = torch.unique(codes, dim=0, return_counts=True)
    histogram = dict(
        sorted(Counter(int(value) for value in counts.tolist()).items())
    )
    usage = tuple(
        torch.unique(codes[:, position]).numel() / rq_codebook_size
        for position in range(3)
    )
    return CollisionStats(
        item_count=codes.shape[0],
        unique_three_token_count=counts.shape[0],
        max_bucket_size=int(counts.max().item()),
        bucket_size_histogram=histogram,
        codebook_usage=usage,
    )


def build_fixed_four_token_ids(
    three_token_ids: Tensor,
    collision_cardinality: int = 256,
) -> FixedCollisionResult:
    if collision_cardinality != 256:
        raise ValueError("paper-strict collision cardinality must equal 256")
    codes = three_token_ids.detach().cpu().long()
    stats = analyze_three_token_ids(codes, rq_codebook_size=256)
    if stats.max_bucket_size > collision_cardinality:
        raise FixedCollisionOverflow(
            f"maximum collision bucket {stats.max_bucket_size} exceeds fixed "
            f"cardinality {collision_cardinality}"
        )

    seen: dict[tuple[int, int, int], int] = {}
    fourth = torch.empty((codes.shape[0], 1), dtype=torch.long)
    for item_index, code in enumerate(codes.tolist()):
        key = tuple(int(token) for token in code)
        collision_index = seen.get(key, 0)
        fourth[item_index, 0] = collision_index
        seen[key] = collision_index + 1

    four_token_ids = torch.cat([codes, fourth], dim=1)
    if int(four_token_ids[:, -1].max().item()) >= collision_cardinality:
        raise FixedCollisionOverflow("fourth token exceeds fixed cardinality 256")
    if torch.unique(four_token_ids, dim=0).shape[0] != four_token_ids.shape[0]:
        raise ValueError("complete four-token IDs must be unique")
    return FixedCollisionResult(four_token_ids=four_token_ids, stats=stats)
