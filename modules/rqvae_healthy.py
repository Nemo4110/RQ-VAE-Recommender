"""Minimal anti-collapse helpers for the isolated TIGER RQ-VAE candidate lane."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from torch import nn


class EmaCodebook:
    """EMA state that replaces optimizer updates to RQ-VAE codebook weights."""

    def __init__(
        self,
        layers: Sequence[nn.Module],
        decay: float,
        *,
        initialization: str = "unit_pseudocount",
    ) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        if initialization not in ("unit_pseudocount", "zero_mass"):
            raise ValueError(
                "EMA initialization must be unit_pseudocount or zero_mass"
            )
        self.decay = decay
        self.initialization = initialization
        self.cluster_sizes = []
        self.embedding_averages = []
        for layer in layers:
            if not hasattr(layer, "n_embed") or not hasattr(layer, "embedding"):
                raise TypeError("EMA layers must expose n_embed and embedding")
            weight = layer.embedding.weight.detach()
            # ``unit_pseudocount`` preserves k-means codebook scale on the
            # first EMA update.  ``zero_mass`` reproduces the historical E2
            # source contract: zero assignment mass but initialized-codebook
            # embedding averages before the first update.
            initial_mass = (
                torch.ones(layer.n_embed, device=weight.device)
                if initialization == "unit_pseudocount"
                else torch.zeros(layer.n_embed, device=weight.device)
            )
            self.cluster_sizes.append(initial_mass)
            self.embedding_averages.append(weight.clone())

    def state_dict(self) -> dict[str, object]:
        return {
            "decay": self.decay,
            "initialization": self.initialization,
            "cluster_sizes": [value.detach().clone() for value in self.cluster_sizes],
            "embedding_averages": [
                value.detach().clone() for value in self.embedding_averages
            ],
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state.get("decay") != self.decay:
            raise ValueError("EMA checkpoint decay does not match the configured decay")
        saved_initialization = state.get("initialization", "unit_pseudocount")
        if saved_initialization != self.initialization:
            raise ValueError("EMA checkpoint initialization does not match the configured mode")
        cluster_sizes = state.get("cluster_sizes")
        embedding_averages = state.get("embedding_averages")
        if not isinstance(cluster_sizes, list) or not isinstance(embedding_averages, list):
            raise TypeError("EMA checkpoint is missing tensor state")
        if len(cluster_sizes) != len(self.cluster_sizes) or len(embedding_averages) != len(
            self.embedding_averages
        ):
            raise ValueError("EMA checkpoint layer count does not match")
        for current, loaded in zip(self.cluster_sizes, cluster_sizes, strict=True):
            if not isinstance(loaded, Tensor) or loaded.shape != current.shape:
                raise ValueError("EMA cluster-size tensor shape does not match")
            current.copy_(loaded.to(device=current.device, dtype=current.dtype))
        for current, loaded in zip(
            self.embedding_averages, embedding_averages, strict=True
        ):
            if not isinstance(loaded, Tensor) or loaded.shape != current.shape:
                raise ValueError("EMA embedding-average tensor shape does not match")
            current.copy_(loaded.to(device=current.device, dtype=current.dtype))

    @torch.no_grad()
    def bootstrap_from_assignment_counts(
        self, layers: Sequence[nn.Module], assignment_counts: Sequence[Tensor]
    ) -> None:
        """Seed EMA mass from the initialized corpus code assignments.

        This preserves each k-means codebook vector while giving its numerator
        and denominator the same observed-corpus mass before the first update.
        """

        if len(layers) != len(self.cluster_sizes) or len(assignment_counts) != len(
            self.cluster_sizes
        ):
            raise ValueError("EMA bootstrap layer count does not match")
        for layer, counts, cluster_sizes, embedding_averages in zip(
            layers, assignment_counts, self.cluster_sizes, self.embedding_averages, strict=True
        ):
            if counts.ndim != 1 or counts.shape[0] != layer.n_embed:
                raise ValueError("EMA bootstrap counts must have one entry per code")
            if torch.any(counts < 0):
                raise ValueError("EMA bootstrap counts must be non-negative")
            observed_counts = counts.to(device=cluster_sizes.device, dtype=cluster_sizes.dtype)
            cluster_sizes.copy_(observed_counts)
            embedding_averages.copy_(
                layer.embedding.weight.detach() * observed_counts.unsqueeze(1)
            )

    @torch.no_grad()
    def update(
        self,
        layers: Sequence[nn.Module],
        residuals: Sequence[Tensor],
        semantic_ids: Sequence[Tensor],
    ) -> None:
        if not (len(layers) == len(residuals) == len(semantic_ids)):
            raise ValueError("layers, residuals, and semantic_ids must have equal length")
        if len(layers) != len(self.cluster_sizes):
            raise ValueError("EMA state does not match the supplied layer count")
        for index, (layer, residual, ids) in enumerate(
            zip(layers, residuals, semantic_ids, strict=True)
        ):
            if residual.ndim != 2 or ids.ndim != 1 or residual.shape[0] != ids.shape[0]:
                raise ValueError("residuals must be [batch, dim] and IDs must be [batch]")
            if ids.numel() and (ids.min() < 0 or ids.max() >= layer.n_embed):
                raise ValueError("semantic IDs are outside the codebook vocabulary")
            assignments = torch.nn.functional.one_hot(ids, layer.n_embed).to(residual)
            counts = assignments.sum(dim=0)
            embedding_sum = assignments.transpose(0, 1) @ residual
            self.cluster_sizes[index].mul_(self.decay).add_(counts, alpha=1 - self.decay)
            self.embedding_averages[index].mul_(self.decay).add_(
                embedding_sum, alpha=1 - self.decay
            )
            total = self.cluster_sizes[index].sum()
            smoothed = (
                (self.cluster_sizes[index] + 1e-5)
                / (total + layer.n_embed * 1e-5)
                * total
            )
            layer.embedding.weight.copy_(
                self.embedding_averages[index] / smoothed.unsqueeze(1).clamp_min(1e-5)
            )


@torch.no_grad()
def reset_dead_codes(
    layers: Sequence[nn.Module],
    residuals: Sequence[Tensor],
    semantic_ids: Sequence[Tensor],
    *,
    generator: torch.Generator | None = None,
    donor_sampling: str = "with_replacement",
) -> list[int]:
    """Replace unused code vectors from current-batch residual donors."""

    if donor_sampling not in ("with_replacement", "without_replacement"):
        raise ValueError("donor_sampling must be with_replacement or without_replacement")
    if not (len(layers) == len(residuals) == len(semantic_ids)):
        raise ValueError("layers, residuals, and semantic_ids must have equal length")
    reset_counts: list[int] = []
    for layer, residual, ids in zip(layers, residuals, semantic_ids, strict=True):
        if residual.ndim != 2 or ids.ndim != 1 or residual.shape[0] != ids.shape[0]:
            raise ValueError("residuals must be [batch, dim] and IDs must be [batch]")
        if residual.shape[0] == 0:
            raise ValueError("cannot reset codes from an empty residual batch")
        used = torch.zeros(layer.n_embed, dtype=torch.bool, device=residual.device)
        used[ids.unique()] = True
        dead = (~used).nonzero(as_tuple=True)[0]
        if dead.numel() == 0:
            reset_counts.append(0)
            continue
        if donor_sampling == "without_replacement":
            donor_indices = torch.randperm(
                residual.shape[0], device=residual.device, generator=generator
            )[: dead.numel()]
        else:
            donor_indices = torch.randint(
                residual.shape[0],
                (dead.numel(),),
                device=residual.device,
                generator=generator,
            )
        donors = residual[donor_indices]
        layer.embedding.weight[dead] = donors
        reset_counts.append(int(dead.numel()))
    return reset_counts


@torch.no_grad()
def append_collision_suffixes(
    semantic_ids: Tensor,
    *,
    suffix_capacity: int,
) -> Tensor:
    """Append deterministic first-seen collision ranks to RQ-VAE token triples."""

    if semantic_ids.ndim != 2 or semantic_ids.shape[1] != 3:
        raise ValueError("semantic_ids must have shape [item_count, 3]")
    if suffix_capacity <= 0:
        raise ValueError("suffix_capacity must be positive")
    suffixes = torch.empty(
        semantic_ids.shape[0], dtype=semantic_ids.dtype, device=semantic_ids.device
    )
    next_suffix_by_triple: dict[tuple[int, int, int], int] = {}
    for item_index, triple in enumerate(semantic_ids.detach().cpu().tolist()):
        key = (int(triple[0]), int(triple[1]), int(triple[2]))
        suffix = next_suffix_by_triple.get(key, 0)
        if suffix >= suffix_capacity:
            raise ValueError(
                "a three-token collision bucket exceeds the fourth-token capacity: "
                f"{suffix + 1} > {suffix_capacity}"
            )
        suffixes[item_index] = suffix
        next_suffix_by_triple[key] = suffix + 1
    return torch.cat([semantic_ids, suffixes.unsqueeze(1)], dim=1)


@torch.no_grad()
def summarize_rqvae_health(
    semantic_ids: Tensor,
    *,
    codebook_size: int,
    suffix_capacity: int = 256,
    minimum_usage: float = 0.80,
) -> dict[str, int | float | bool | list[int] | list[float]]:
    """Compute fixed-four-token RQ-VAE health gates for an item corpus."""

    if semantic_ids.ndim != 2 or semantic_ids.shape[1] != 3:
        raise ValueError("semantic_ids must have shape [item_count, 3]")
    if codebook_size <= 0 or suffix_capacity <= 0:
        raise ValueError("codebook and suffix capacities must be positive")
    if not 0.0 < minimum_usage <= 1.0:
        raise ValueError("minimum_usage must be in (0, 1]")
    if semantic_ids.numel() and (
        semantic_ids.min() < 0 or semantic_ids.max() >= codebook_size
    ):
        raise ValueError("semantic IDs are outside the codebook vocabulary")

    item_count = int(semantic_ids.shape[0])
    used_counts = [
        int(semantic_ids[:, index].unique().numel()) for index in range(semantic_ids.shape[1])
    ]
    codebook_usage = [count / codebook_size for count in used_counts]
    _, bucket_sizes = torch.unique(semantic_ids, dim=0, return_counts=True)
    unique_three_token_count = int(bucket_sizes.numel())
    max_bucket_size = int(bucket_sizes.max().item()) if item_count else 0
    collision_capacity_passed = max_bucket_size <= suffix_capacity

    if collision_capacity_passed:
        full_ids = append_collision_suffixes(
            semantic_ids, suffix_capacity=suffix_capacity
        )
        unique_full_id_count = int(torch.unique(full_ids, dim=0).shape[0])
        full_id_unique = unique_full_id_count == item_count
    else:
        unique_full_id_count = 0
        full_id_unique = False

    usage_gate_passed = all(value >= minimum_usage for value in codebook_usage)
    return {
        "item_count": item_count,
        "rqvae_n_layers": int(semantic_ids.shape[1]),
        "codebook_size": codebook_size,
        "suffix_capacity": suffix_capacity,
        "minimum_usage": minimum_usage,
        "used_counts": used_counts,
        "codebook_usage": codebook_usage,
        "unique_three_token_count": unique_three_token_count,
        "max_collision_bucket": max_bucket_size,
        "unique_full_id_count": unique_full_id_count,
        "full_id_unique": full_id_unique,
        "collision_capacity_passed": collision_capacity_passed,
        "usage_gate_passed": usage_gate_passed,
        "hard_collapse": max_bucket_size > codebook_size * 10,
        "paper_gate_passed": collision_capacity_passed
        and full_id_unique
        and usage_gate_passed,
    }


@torch.no_grad()
def apply_kaiming_relu_initialization(model: nn.Module) -> int:
    """Re-initialize Linear weights for ReLU stacks and zero their biases."""

    count = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            count += 1
    if count == 0:
        raise ValueError("model contains no Linear modules to initialize")
    return count
