from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


INITIALIZATION_CONFIG_FIELDS = {
    "dataset_sha256",
    "dataset_item_count",
    "seed",
    "batch_size",
    "vae_input_dim",
    "vae_hidden_dims",
    "vae_embed_dim",
    "vae_codebook_size",
    "vae_n_layers",
    "commitment_weight",
}

_INVARIANT_EXCLUDED_FIELDS = {
    "optimizer_name",
    "learning_rate",
    "weight_decay",
    "optimizer_eps",
    "initial_accumulator_value",
    "adam_betas",
    "arm_label",
    "save_dir_root",
    "config_path",
    "resume_checkpoint",
}


@dataclass(frozen=True)
class OptimizerSpec:
    name: str
    learning_rate: float
    weight_decay: float
    eps: float
    initial_accumulator_value: float | None = None
    betas: tuple[float, float] | None = None

    @classmethod
    def adagrad_default(cls) -> OptimizerSpec:
        return cls("adagrad", 0.4, 0.0, 1e-10, 0.0, None)

    @classmethod
    def adagrad_accum01_eps1e7(cls) -> OptimizerSpec:
        return cls("adagrad", 0.4, 0.0, 1e-7, 0.1, None)

    @classmethod
    def adamw_author(cls) -> OptimizerSpec:
        return cls("adamw", 1e-3, 1e-4, 1e-8, None, (0.9, 0.999))


def build_diagnostic_optimizer(
    model: nn.Module,
    spec: OptimizerSpec,
) -> torch.optim.Optimizer:
    if spec in {
        OptimizerSpec.adagrad_default(),
        OptimizerSpec.adagrad_accum01_eps1e7(),
    }:
        return torch.optim.Adagrad(
            model.parameters(),
            lr=spec.learning_rate,
            weight_decay=spec.weight_decay,
            initial_accumulator_value=float(spec.initial_accumulator_value),
            eps=spec.eps,
        )
    if spec == OptimizerSpec.adamw_author():
        return torch.optim.AdamW(
            model.parameters(),
            lr=spec.learning_rate,
            weight_decay=spec.weight_decay,
            eps=spec.eps,
            betas=spec.betas,
        )
    raise ValueError(f"unsupported diagnostic optimizer treatment: {spec}")


def _json_default(value: object) -> object:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, torch.dtype):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def hash_json(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def optimizer_treatment_payload(spec: OptimizerSpec) -> dict[str, Any]:
    return asdict(spec)


def optimizer_treatment_hash(spec: OptimizerSpec) -> str:
    return hash_json(optimizer_treatment_payload(spec))


def optimizer_metadata(
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    return {
        "module": type(optimizer).__module__,
        "class": type(optimizer).__name__,
        "defaults": dict(optimizer.defaults),
        "param_groups": [
            {key: value for key, value in group.items() if key != "params"}
            for group in optimizer.param_groups
        ],
    }


def optimizer_metadata_hash(optimizer: torch.optim.Optimizer) -> str:
    return hash_json(optimizer_metadata(optimizer))


def complete_config_hash(config: Mapping[str, Any]) -> str:
    return hash_json(dict(config))


def invariant_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in config.items()
        if key not in _INVARIANT_EXCLUDED_FIELDS
    }


def invariant_config_hash(config: Mapping[str, Any]) -> str:
    return hash_json(invariant_config(config))


def initialization_config(config: Mapping[str, Any]) -> dict[str, Any]:
    missing = INITIALIZATION_CONFIG_FIELDS.difference(config)
    if missing:
        raise ValueError(f"initialization config missing fields: {sorted(missing)}")
    return {key: config[key] for key in sorted(INITIALIZATION_CONFIG_FIELDS)}


def initialization_config_hash(config: Mapping[str, Any]) -> str:
    return hash_json(initialization_config(config))
