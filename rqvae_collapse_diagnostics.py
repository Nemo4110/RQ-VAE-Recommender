from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


INITIALIZATION_CONFIG_FIELDS = frozenset(
    {
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
)

_INVARIANT_EXCLUDED_FIELDS = frozenset(
    {
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
)


def _normalize_optimizer_float(field_name: str, value: object) -> float:
    if type(value) not in (int, float):
        raise ValueError(
            f"optimizer config field {field_name} must be an int or float"
        )
    return float(value)


def _normalize_optimizer_betas(
    value: object,
) -> tuple[float, float] | None:
    if value is None:
        return None
    if type(value) not in (list, tuple) or len(value) != 2:
        raise ValueError(
            "optimizer config field adam_betas must be a two-value list or tuple"
        )
    return (
        _normalize_optimizer_float("adam_betas", value[0]),
        _normalize_optimizer_float("adam_betas", value[1]),
    )


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

    @classmethod
    def from_config_fields(
        cls,
        *,
        optimizer_name: object,
        learning_rate: object,
        weight_decay: object,
        optimizer_eps: object,
        initial_accumulator_value: object,
        adam_betas: object,
    ) -> OptimizerSpec:
        if type(optimizer_name) is not str:
            raise ValueError("optimizer config field optimizer_name must be a string")
        normalized_accumulator = (
            None
            if initial_accumulator_value is None
            else _normalize_optimizer_float(
                "initial_accumulator_value",
                initial_accumulator_value,
            )
        )
        spec = cls(
            name=optimizer_name,
            learning_rate=_normalize_optimizer_float(
                "learning_rate",
                learning_rate,
            ),
            weight_decay=_normalize_optimizer_float(
                "weight_decay",
                weight_decay,
            ),
            eps=_normalize_optimizer_float("optimizer_eps", optimizer_eps),
            initial_accumulator_value=normalized_accumulator,
            betas=_normalize_optimizer_betas(adam_betas),
        )
        return _canonical_optimizer_spec(spec)


def _require_exact_optimizer_spec_types(spec: OptimizerSpec) -> None:
    if type(spec.name) is not str:
        raise ValueError("optimizer spec field name must have exact runtime type str")
    for field_name in ("learning_rate", "weight_decay", "eps"):
        if type(getattr(spec, field_name)) is not float:
            raise ValueError(
                f"optimizer spec field {field_name} must have exact runtime type float"
            )
    if (
        spec.initial_accumulator_value is not None
        and type(spec.initial_accumulator_value) is not float
    ):
        raise ValueError(
            "optimizer spec field initial_accumulator_value must be None or "
            "have exact runtime type float"
        )
    if spec.betas is None:
        return
    if type(spec.betas) is not tuple:
        raise ValueError(
            "optimizer spec field betas must be None or have exact runtime type tuple"
        )
    if len(spec.betas) != 2 or any(type(value) is not float for value in spec.betas):
        raise ValueError(
            "optimizer spec field betas must contain exactly two exact floats"
        )


def _canonical_optimizer_spec(spec: OptimizerSpec) -> OptimizerSpec:
    _require_exact_optimizer_spec_types(spec)
    for candidate in (
        OptimizerSpec.adagrad_default(),
        OptimizerSpec.adagrad_accum01_eps1e7(),
        OptimizerSpec.adamw_author(),
    ):
        if spec == candidate:
            return candidate
    raise ValueError(f"unsupported diagnostic optimizer treatment: {spec}")


def build_diagnostic_optimizer(
    model: nn.Module,
    spec: OptimizerSpec,
) -> torch.optim.Optimizer:
    canonical_spec = _canonical_optimizer_spec(spec)
    if canonical_spec.name == "adagrad":
        return torch.optim.Adagrad(
            model.parameters(),
            lr=canonical_spec.learning_rate,
            weight_decay=canonical_spec.weight_decay,
            initial_accumulator_value=float(
                canonical_spec.initial_accumulator_value
            ),
            eps=canonical_spec.eps,
        )
    return torch.optim.AdamW(
        model.parameters(),
        lr=canonical_spec.learning_rate,
        weight_decay=canonical_spec.weight_decay,
        eps=canonical_spec.eps,
        betas=canonical_spec.betas,
    )


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
        allow_nan=False,
        default=_json_default,
    )


def hash_json(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def optimizer_treatment_payload(spec: OptimizerSpec) -> dict[str, Any]:
    return asdict(_canonical_optimizer_spec(spec))


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
