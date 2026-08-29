"""Explicit TIGER token, user-token, and evaluator policies.

The RQ-VAE remains three-layer.  A fourth semantic-ID position, when used,
is a post-training collision-resolution suffix and not a trainable codebook.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Sequence

import torch


HISTORICAL_NATIVE = "historical/native"
PAPER_FULL_ID = "paper/full-id"
MODULO_HASHED_BUCKET = "modulo_hashed_bucket"
EXPLICIT_USER_BINS = "explicit"
PAPER_FIXED_USER_BINS = "paper_fixed"
DATASET_CAPPED_USER_BINS = "dataset_capped"
NATIVE_SEMANTIC_ID = "native_semantic_id"
TEACHER_FORCED_FULL_CATALOG = "teacher_forced_full_catalog"

TokenPolicy = Literal["historical/native", "paper/full-id"]


@dataclass(frozen=True)
class TIGERPolicyConfig:
    """Validated protocol settings for a TIGER decoder run."""

    token_policy: TokenPolicy = HISTORICAL_NATIVE
    paper_aligned: bool = False
    num_user_bins: int | None = None
    user_token_policy: str = MODULO_HASHED_BUCKET
    user_bin_mode: str = EXPLICIT_USER_BINS
    user_bin_cap: int = 2000
    dataset_user_count: int | None = None
    evaluator_policy: str = NATIVE_SEMANTIC_ID
    rqvae_n_layers: int = 3

    def validate(self) -> None:
        if self.token_policy not in (HISTORICAL_NATIVE, PAPER_FULL_ID):
            raise ValueError(
                f"unsupported TIGER token_policy={self.token_policy!r}; "
                f"expected {HISTORICAL_NATIVE!r} or {PAPER_FULL_ID!r}"
            )
        if self.rqvae_n_layers != 3 and (
            self.paper_aligned or self.token_policy == PAPER_FULL_ID
        ):
            raise ValueError(
                "TIGER paper-aligned/full-id mode requires exactly three RQ-VAE layers"
            )
        if self.num_user_bins is not None and self.num_user_bins <= 0:
            raise ValueError("num_user_bins must be positive when provided")
        if self.user_bin_mode not in (
            EXPLICIT_USER_BINS,
            PAPER_FIXED_USER_BINS,
            DATASET_CAPPED_USER_BINS,
        ):
            raise ValueError(f"unsupported user_bin_mode={self.user_bin_mode!r}")
        if self.user_bin_cap <= 0:
            raise ValueError("user_bin_cap must be positive")
        if self.dataset_user_count is not None and self.dataset_user_count <= 0:
            raise ValueError("dataset_user_count must be positive when provided")
        if self.paper_aligned:
            if self.user_bin_mode != PAPER_FIXED_USER_BINS or self.num_user_bins != 2000:
                raise ValueError(
                    "TIGER paper-aligned mode requires user_bin_mode=paper_fixed "
                    "and num_user_bins=2000"
                )
            if self.user_token_policy != MODULO_HASHED_BUCKET:
                raise ValueError(
                    "the current implementation only supports modulo_hashed_bucket "
                    "for paper-aligned user bins"
                )
        if self.token_policy == PAPER_FULL_ID and self.user_token_policy != MODULO_HASHED_BUCKET:
            raise ValueError(
                "the current implementation only supports modulo_hashed_bucket "
                "for full-id user bins"
            )
        if self.user_bin_mode == DATASET_CAPPED_USER_BINS:
            if (
                self.dataset_user_count is not None
                and self.num_user_bins is not None
                and self.num_user_bins > min(self.user_bin_cap, self.dataset_user_count)
            ):
                raise ValueError(
                    "num_user_bins cannot exceed the dataset user-bin cap or user count"
                )
        if self.evaluator_policy not in (
            NATIVE_SEMANTIC_ID,
            TEACHER_FORCED_FULL_CATALOG,
        ):
            raise ValueError(f"unsupported evaluator_policy={self.evaluator_policy!r}")

    def for_dataset(self, dataset_user_count: int) -> "TIGERPolicyConfig":
        """Resolve dataset-capped bins without changing paper-fixed semantics."""

        if dataset_user_count <= 0:
            raise ValueError("dataset_user_count must be positive")
        resolved = min(self.user_bin_cap, dataset_user_count)
        if self.user_bin_mode == DATASET_CAPPED_USER_BINS:
            if self.num_user_bins is not None:
                resolved = min(self.num_user_bins, resolved)
            return replace(
                self, num_user_bins=resolved, dataset_user_count=dataset_user_count
            )
        return replace(self, dataset_user_count=dataset_user_count)

    @property
    def effective_num_user_bins(self) -> int | None:
        if self.user_bin_mode == DATASET_CAPPED_USER_BINS:
            if self.dataset_user_count is None:
                raise ValueError("dataset_user_count is required for dataset_capped user bins")
            return min(
                self.num_user_bins or self.user_bin_cap,
                self.user_bin_cap,
                self.dataset_user_count,
            )
        return self.num_user_bins

    @property
    def uses_collision_suffix(self) -> bool:
        return self.token_policy == PAPER_FULL_ID

    def metadata(self) -> dict[str, object]:
        self.validate()
        return {
            "token_policy": self.token_policy,
            "paper_aligned": self.paper_aligned,
            "user_token_policy": self.user_token_policy,
            "num_user_bins": self.num_user_bins,
            "effective_num_user_bins": self.effective_num_user_bins,
            "user_bin_mode": self.user_bin_mode,
            "user_bin_cap": self.user_bin_cap,
            "dataset_user_count": self.dataset_user_count,
            "evaluator_policy": self.evaluator_policy,
            "rqvae_n_layers": self.rqvae_n_layers,
            "collision_suffix": "post_training_deterministic"
            if self.uses_collision_suffix
            else "excluded_by_policy",
        }


def validate_checkpoint_policy(
    checkpoint_policy: dict[str, object] | None,
    expected_policy: TIGERPolicyConfig,
) -> None:
    """Reject checkpoints whose recorded protocol cannot match this run.

    Historical checkpoints created before policy metadata existed remain loadable;
    paper/full-id checkpoints must carry metadata so token semantics cannot be guessed.
    """

    expected_policy.validate()
    if checkpoint_policy is None:
        if expected_policy.uses_collision_suffix or expected_policy.paper_aligned:
            raise ValueError(
                "paper-aligned/full-id decoder checkpoint is missing TIGER policy metadata"
            )
        return
    expected = expected_policy.metadata()
    for key in (
        "token_policy",
        "paper_aligned",
        "user_token_policy",
        "num_user_bins",
        "effective_num_user_bins",
        "user_bin_mode",
        "user_bin_cap",
        "dataset_user_count",
        "evaluator_policy",
    ):
        if checkpoint_policy.get(key) != expected[key]:
            raise ValueError(
                f"decoder checkpoint TIGER policy mismatch for {key}: "
                f"{checkpoint_policy.get(key)!r} != {expected[key]!r}"
            )


def select_semantic_id_tokens(
    semantic_ids: torch.Tensor,
    *,
    n_layers: int,
    token_policy: TokenPolicy,
    source_item_width: int | None = None,
) -> torch.Tensor:
    """Select explicit decoder tokens from flattened or row-wise semantic IDs.

    ``semantic_ids`` may be ``[B, N * D]`` or ``[B, D]``.  The selection is
    policy-driven and deliberately does not silently truncate a suffix.
    """

    if semantic_ids.ndim not in (2, 3):
        raise ValueError("semantic_ids must have shape [B, D] or [B, N, D]")
    if n_layers != 3 and token_policy == PAPER_FULL_ID:
        raise ValueError("paper/full-id policy requires n_layers=3")
    expected_width = n_layers + (1 if token_policy == PAPER_FULL_ID else 0)
    item_width = source_item_width or (n_layers + 1)
    if item_width < expected_width:
        raise ValueError("source_item_width cannot be smaller than selected token width")

    if semantic_ids.ndim == 2:
        if semantic_ids.shape[1] % item_width != 0:
            raise ValueError(
                f"flattened semantic ID width {semantic_ids.shape[1]} is not divisible by "
                f"source item width {item_width}"
            )
        batch, total = semantic_ids.shape
        reshaped = semantic_ids.view(batch, total // item_width, item_width)
        return reshaped[:, :, :expected_width].reshape(batch, -1)

    if semantic_ids.shape[-1] != item_width:
        raise ValueError(
            f"row-wise semantic IDs must include {item_width} source columns; "
            f"got {semantic_ids.shape[-1]}"
        )
    return semantic_ids[..., :expected_width]


def token_type_offsets(cardinalities: Sequence[int]) -> tuple[int, ...]:
    """Return cumulative embedding offsets for each decoder token position."""

    values = tuple(int(cardinality) for cardinality in cardinalities)
    if not values or any(cardinality <= 0 for cardinality in values):
        raise ValueError("token cardinalities must be a non-empty positive sequence")
    offsets = []
    current = 0
    for cardinality in values:
        offsets.append(current)
        current += cardinality
    return tuple(offsets)


def token_cardinalities(
    codebooks: torch.Tensor,
    *,
    n_layers: int,
    rqvae_codebook_size: int,
    token_policy: TokenPolicy,
) -> tuple[int, ...]:
    """Return per-position vocabularies for the selected decoder policy."""

    if codebooks.ndim != 2:
        raise ValueError("codebooks must have shape [items, tokens]")
    if codebooks.shape[1] < n_layers:
        raise ValueError("codebooks do not contain all RQ-VAE layers")
    if token_policy == HISTORICAL_NATIVE:
        return (rqvae_codebook_size,) * n_layers
    if codebooks.shape[1] < n_layers + 1:
        raise ValueError("paper/full-id policy requires a collision suffix column")
    suffix = codebooks[:, n_layers]
    if suffix.numel() == 0 or suffix.min().item() < 0:
        raise ValueError("collision suffix values must be non-negative")
    return (rqvae_codebook_size,) * n_layers + (int(suffix.max().item()) + 1,)


def validate_full_semantic_ids(codebooks: torch.Tensor) -> dict[str, int | bool]:
    """Validate and summarize post-training full-ID uniqueness."""

    if codebooks.ndim != 2 or codebooks.shape[1] < 4:
        raise ValueError("full semantic IDs must have at least four columns")
    unique_count = int(torch.unique(codebooks, dim=0).shape[0])
    return {
        "item_count": int(codebooks.shape[0]),
        "unique_full_id_count": unique_count,
        "full_id_unique": unique_count == int(codebooks.shape[0]),
        "suffix_cardinality": int(codebooks[:, 3].max().item()) + 1
        if codebooks.shape[0]
        else 0,
        "max_collision_bucket": int(codebooks[:, 3].max().item()) + 1
        if codebooks.shape[0]
        else 0,
    }


def user_bucket_indices(user_ids: torch.Tensor, num_user_bins: int) -> torch.Tensor:
    """Apply the repository's explicit modulo user-bucket mapping."""

    if num_user_bins <= 0:
        raise ValueError("num_user_bins must be positive")
    return torch.remainder(user_ids, num_user_bins)
