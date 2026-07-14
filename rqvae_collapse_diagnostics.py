from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import struct
import tempfile
from collections import defaultdict
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any
from typing import TypedDict

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from modules.tokenizer.fixed_collision import FixedCollisionOverflow
from modules.tokenizer.fixed_collision import build_fixed_four_token_ids


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


class RngState(TypedDict):
    python: object
    numpy: tuple[object, ...]
    torch_cpu: torch.Tensor
    torch_cuda: list[torch.Tensor]


_COMMON_ARTIFACT_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "dataset",
        "seed",
        "initialization_config",
        "initialization_config_hash",
        "initial_model_state",
        "initial_model_hash",
        "post_kmeans_model_state",
        "post_kmeans_model_hash",
        "post_kmeans_codebook_hash",
        "rng_before_model_initialization",
        "rng_before_model_initialization_hash",
        "rng_at_training_start",
        "rng_at_training_start_hash",
        "initial_batch",
        "epoch_plan_hashes",
        "epoch_plan_rolling_hash",
        "compatibility",
        "common_initialization_hash",
    }
)

_COMPATIBILITY_FIELDS = frozenset(
    {
        "dataset_sha256",
        "dataset_item_count",
        "seed",
        "initialization_config_hash",
        "initial_model_hash",
        "post_kmeans_model_hash",
        "post_kmeans_codebook_hash",
        "initial_batch_indices_hash",
        "initial_batch_input_hash",
        "rng_at_training_start_hash",
        "epoch_plan_rolling_hash",
    }
)


def hash_tensor(tensor: torch.Tensor) -> str:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor value must be a torch.Tensor")
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _update_hash_part(
    digest: Any,
    tag: bytes,
    payload: bytes = b"",
) -> None:
    digest.update(len(tag).to_bytes(4, "big"))
    digest.update(tag)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _update_nested_hash(digest: Any, value: object) -> None:
    if isinstance(value, torch.Tensor):
        _update_hash_part(digest, b"torch.Tensor", hash_tensor(value).encode("ascii"))
        return
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("NumPy object arrays cannot be hashed deterministically")
        array = np.ascontiguousarray(value)
        _update_hash_part(digest, b"numpy.dtype", array.dtype.str.encode("ascii"))
        _update_hash_part(
            digest,
            b"numpy.shape",
            canonical_json(list(array.shape)).encode("utf-8"),
        )
        _update_hash_part(digest, b"numpy.bytes", array.tobytes(order="C"))
        return
    if isinstance(value, Mapping):
        items = sorted(
            ((str(key), item) for key, item in value.items()),
            key=lambda pair: pair[0],
        )
        string_keys = [key for key, _ in items]
        if len(string_keys) != len(set(string_keys)):
            raise ValueError("mapping keys must be unique after string conversion")
        _update_hash_part(
            digest,
            b"mapping",
            len(items).to_bytes(8, "big"),
        )
        for key, item in items:
            _update_hash_part(digest, b"mapping.key", key.encode("utf-8"))
            _update_nested_hash(digest, item)
        return
    if isinstance(value, list):
        _update_hash_part(digest, b"list", len(value).to_bytes(8, "big"))
        for item in value:
            _update_nested_hash(digest, item)
        return
    if isinstance(value, tuple):
        _update_hash_part(digest, b"tuple", len(value).to_bytes(8, "big"))
        for item in value:
            _update_nested_hash(digest, item)
        return
    if isinstance(value, Path):
        _update_hash_part(digest, b"path", str(value).encode("utf-8"))
        return
    if isinstance(value, np.generic):
        _update_nested_hash(digest, value.item())
        return
    if value is None:
        _update_hash_part(digest, b"none")
        return
    if isinstance(value, bool):
        _update_hash_part(digest, b"bool", b"1" if value else b"0")
        return
    if isinstance(value, int):
        _update_hash_part(digest, b"int", str(value).encode("ascii"))
        return
    if isinstance(value, float):
        _update_hash_part(digest, b"float", struct.pack("!d", value))
        return
    if isinstance(value, str):
        _update_hash_part(digest, b"str", value.encode("utf-8"))
        return
    if isinstance(value, bytes):
        _update_hash_part(digest, b"bytes", value)
        return
    raise TypeError(f"unsupported nested state value: {type(value).__name__}")


def hash_nested_state(value: object) -> str:
    digest = hashlib.sha256()
    _update_nested_hash(digest, value)
    return digest.hexdigest()


def seed_all(seed: int) -> None:
    if type(seed) is not int:
        raise ValueError("seed must be an int")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _extract_item_tensor(value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "x") and isinstance(value.x, torch.Tensor):
        return value.x
    if isinstance(value, Mapping) and isinstance(value.get("x"), torch.Tensor):
        return value["x"]
    if (
        isinstance(value, (list, tuple))
        and len(value) == 1
        and isinstance(value[0], torch.Tensor)
    ):
        return value[0]
    raise TypeError("dataset items must be tensors or expose a tensor-valued x field")


def load_items_by_index(dataset: Dataset, indices: Tensor) -> Tensor:
    if not isinstance(indices, Tensor):
        raise TypeError("indices must be a torch.Tensor")
    index_tensor = indices.detach().to(dtype=torch.long, device="cpu")
    if index_tensor.ndim != 1:
        raise ValueError("indices must be one-dimensional")
    try:
        return _extract_item_tensor(dataset[index_tensor])
    except (IndexError, TypeError, ValueError):
        return torch.stack(
            [
                _extract_item_tensor(dataset[int(index)])
                for index in index_tensor.tolist()
            ]
        )


def clone_cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        str(key): value.detach().to(device="cpu").clone()
        for key, value in model.state_dict().items()
    }


def _clone_cpu_state_mapping(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if not isinstance(state, Mapping):
        raise TypeError("model state must be a mapping")
    cloned: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if type(key) is not str:
            raise TypeError("model state keys must be strings")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"model state value {key} must be a tensor")
        cloned[key] = value.detach().to(device="cpu").clone()
    return cloned


def hash_state_dict(state: Mapping[str, torch.Tensor]) -> str:
    return hash_nested_state(
        dict(sorted(_clone_cpu_state_mapping(state).items()))
    )


def _clone_numpy_rng_state(
    state: tuple[object, ...],
) -> tuple[object, ...]:
    return tuple(
        value.copy() if isinstance(value, np.ndarray) else copy.deepcopy(value)
        for value in state
    )


def _clone_rng_state(state: Mapping[str, object]) -> RngState:
    _validate_rng_state(state)
    return {
        "python": copy.deepcopy(state["python"]),
        "numpy": _clone_numpy_rng_state(state["numpy"]),
        "torch_cpu": state["torch_cpu"].detach().to(device="cpu").clone(),
        "torch_cuda": [
            value.detach().to(device="cpu").clone()
            for value in state["torch_cuda"]
        ],
    }


def capture_rng_state() -> RngState:
    numpy_state = np.random.get_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return {
        "python": copy.deepcopy(random.getstate()),
        "numpy": _clone_numpy_rng_state(numpy_state),
        "torch_cpu": torch.get_rng_state().detach().to(device="cpu").clone(),
        "torch_cuda": [
            value.detach().to(device="cpu").clone() for value in cuda_states
        ],
    }


def _validate_rng_state(state: Mapping[str, object]) -> None:
    if not isinstance(state, Mapping):
        raise TypeError("RNG state must be a mapping")
    expected_fields = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(state) != expected_fields:
        raise ValueError("RNG state has an invalid schema")
    if not isinstance(state["numpy"], tuple):
        raise TypeError("NumPy RNG state must be a tuple")
    if not isinstance(state["torch_cpu"], torch.Tensor):
        raise TypeError("CPU torch RNG state must be a tensor")
    if not isinstance(state["torch_cuda"], list) or any(
        not isinstance(value, torch.Tensor) for value in state["torch_cuda"]
    ):
        raise TypeError("CUDA torch RNG state must be a list of tensors")

    python_probe = random.Random()
    python_probe.setstate(state["python"])
    numpy_probe = np.random.RandomState()
    numpy_probe.set_state(state["numpy"])
    cpu_probe = torch.Generator(device="cpu")
    cpu_probe.set_state(state["torch_cpu"].detach().to(device="cpu"))

    cuda_states = state["torch_cuda"]
    if cuda_states:
        if not torch.cuda.is_available():
            raise ValueError("CUDA RNG state cannot be restored without CUDA")
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError("CUDA RNG state count does not match visible devices")
        for index, cuda_state in enumerate(cuda_states):
            probe = torch.Generator(device=f"cuda:{index}")
            probe.set_state(cuda_state.detach().to(device="cpu"))


def restore_rng_state(state: RngState) -> None:
    _validate_rng_state(state)
    random.setstate(copy.deepcopy(state["python"]))
    np.random.set_state(_clone_numpy_rng_state(state["numpy"]))
    torch.set_rng_state(state["torch_cpu"].detach().to(device="cpu").clone())
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(
            [
                value.detach().to(device="cpu").clone()
                for value in state["torch_cuda"]
            ]
        )


def hash_rng_state(state: RngState) -> str:
    _validate_rng_state(state)
    return hash_nested_state(state)


def _require_positive_int(field_name: str, value: object) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field_name} must be a positive int")
    return value


def epoch_permutation(item_count: int, seed: int, epoch: int) -> torch.Tensor:
    item_count = _require_positive_int("item_count", item_count)
    epoch = _require_positive_int("epoch", epoch)
    if type(seed) is not int:
        raise ValueError("seed must be an int")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + epoch - 1)
    return torch.randperm(item_count, generator=generator, device="cpu")


def epoch_plan_hashes(
    item_count: int,
    seed: int,
    epochs: int,
) -> list[str]:
    epochs = _require_positive_int("epochs", epochs)
    return [
        hash_tensor(epoch_permutation(item_count, seed, epoch))
        for epoch in range(1, epochs + 1)
    ]


def rolling_epoch_plan_hash(
    item_count: int,
    seed: int,
    epochs: int,
) -> str:
    return hash_json(epoch_plan_hashes(item_count, seed, epochs))


def _post_kmeans_codebook_state(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    codebooks = {
        key: value
        for key, value in state.items()
        if key.startswith("layers.") and key.endswith(".embedding.weight")
    }
    if not codebooks:
        raise ValueError("post-kmeans model state contains no RQ-VAE codebooks")
    return codebooks


def _hash_codebook_state(state: Mapping[str, torch.Tensor]) -> str:
    return hash_json(
        [
            hash_tensor(tensor)
            for _, tensor in sorted(_post_kmeans_codebook_state(state).items())
        ]
    )


def _validate_state_layouts(
    initial_state: Mapping[str, torch.Tensor],
    post_state: Mapping[str, torch.Tensor],
) -> None:
    if set(initial_state) != set(post_state):
        raise ValueError("initial and post-kmeans model state keys differ")
    for key in initial_state:
        if initial_state[key].shape != post_state[key].shape:
            raise ValueError(f"model state shape changed for {key}")
        if initial_state[key].dtype != post_state[key].dtype:
            raise ValueError(f"model state dtype changed for {key}")


def _normalize_epoch_hashes(epoch_hashes: Sequence[str]) -> list[str]:
    if isinstance(epoch_hashes, (str, bytes)) or not isinstance(
        epoch_hashes,
        Sequence,
    ):
        raise TypeError("epoch_hashes must be a sequence")
    normalized = list(epoch_hashes)
    if len(normalized) != 500:
        raise ValueError("epoch_hashes must contain exactly 500 hashes")
    if any(
        type(value) is not str or len(value) != 64
        for value in normalized
    ):
        raise ValueError("epoch_hashes must contain SHA-256 hex strings")
    return normalized


def _normalize_used_code_counts(value: Sequence[int]) -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(
            "initial_batch_used_code_counts must be a sequence of ints"
        )
    counts = list(value)
    if any(type(count) is not int or count < 0 for count in counts):
        raise ValueError("initial batch used code counts must be nonnegative ints")
    return counts


def _validated_initialization_config_payload(
    payload: Mapping[str, Any],
    *,
    dataset_sha256: str,
    dataset_item_count: int,
    seed: int,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("initialization config payload must be a mapping")
    if set(payload) != INITIALIZATION_CONFIG_FIELDS:
        raise ValueError(
            "initialization config payload must contain exactly the fixed fields"
        )
    normalized = copy.deepcopy(dict(payload))
    required_matches = {
        "dataset_sha256": dataset_sha256,
        "dataset_item_count": dataset_item_count,
        "seed": seed,
    }
    for field_name, expected in required_matches.items():
        actual = normalized[field_name]
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(
                f"initialization config {field_name} does not agree "
                "with common initialization metadata"
            )
    return normalized


def _validated_initial_batch_indices(
    indices: Tensor,
    *,
    dataset_item_count: int,
    seed: int,
) -> Tensor:
    if not isinstance(indices, Tensor):
        raise TypeError("initial batch indices must be a torch.Tensor")
    value = indices.detach().to(device="cpu")
    if value.ndim != 1:
        raise ValueError("initial batch indices must be one-dimensional")
    if value.dtype is not torch.long:
        raise ValueError("initial batch indices must use torch.long")
    if value.numel() != 1024:
        raise ValueError("initial batch indices must contain exactly 1024 values")
    expected = epoch_permutation(dataset_item_count, seed, 1)[:1024]
    if expected.numel() != 1024:
        raise ValueError(
            "initial batch requires a dataset with at least 1024 items"
        )
    if not torch.equal(value, expected):
        raise ValueError(
            "initial batch indices must equal the epoch permutation prefix"
        )
    return value.contiguous().clone()


def _validated_initial_batch_inputs(
    inputs: Tensor,
    *,
    indices: Tensor,
) -> Tensor:
    if not isinstance(inputs, Tensor):
        raise TypeError("initial batch inputs must be a torch.Tensor")
    if inputs.ndim == 0 or inputs.shape[0] != 1024:
        raise ValueError("initial batch inputs must contain exactly 1024 rows")
    if inputs.shape[0] != indices.numel():
        raise ValueError("initial batch input rows must match index rows")
    return inputs.detach().to(device="cpu").clone()


def _compatibility_payload(
    *,
    dataset_sha256: str,
    dataset_item_count: int,
    seed: int,
    initialization_config_hash_value: str,
    initial_model_hash: str,
    post_kmeans_model_hash: str,
    post_kmeans_codebook_hash: str,
    initial_batch_indices_hash: str,
    initial_batch_input_hash: str,
    rng_at_training_start_hash: str,
    epoch_plan_rolling_hash: str,
) -> dict[str, object]:
    return {
        "dataset_sha256": dataset_sha256,
        "dataset_item_count": dataset_item_count,
        "seed": seed,
        "initialization_config_hash": initialization_config_hash_value,
        "initial_model_hash": initial_model_hash,
        "post_kmeans_model_hash": post_kmeans_model_hash,
        "post_kmeans_codebook_hash": post_kmeans_codebook_hash,
        "initial_batch_indices_hash": initial_batch_indices_hash,
        "initial_batch_input_hash": initial_batch_input_hash,
        "rng_at_training_start_hash": rng_at_training_start_hash,
        "epoch_plan_rolling_hash": epoch_plan_rolling_hash,
    }


def save_common_initialization(
    path: Path,
    *,
    initial_model_state: Mapping[str, Tensor],
    post_kmeans_model_state: Mapping[str, Tensor],
    rng_before_model_initialization: RngState,
    rng_at_training_start: RngState,
    initial_batch_indices: Tensor,
    initial_batch_inputs: Tensor,
    initial_batch_used_code_counts: Sequence[int],
    dataset_sha256: str,
    dataset_item_count: int,
    seed: int,
    initialization_config_payload: Mapping[str, Any],
    epoch_hashes: Sequence[str],
    epoch_plan_rolling_hash: str,
) -> dict[str, Any]:
    destination = Path(path)
    if type(dataset_sha256) is not str:
        raise TypeError("dataset_sha256 must be a string")
    dataset_item_count = _require_positive_int(
        "dataset_item_count",
        dataset_item_count,
    )
    if type(seed) is not int:
        raise ValueError("seed must be an int")
    config_payload = _validated_initialization_config_payload(
        initialization_config_payload,
        dataset_sha256=dataset_sha256,
        dataset_item_count=dataset_item_count,
        seed=seed,
    )

    initial_state = _clone_cpu_state_mapping(initial_model_state)
    post_state = _clone_cpu_state_mapping(post_kmeans_model_state)
    _validate_state_layouts(initial_state, post_state)
    rng_before = _clone_rng_state(rng_before_model_initialization)
    rng_training = _clone_rng_state(rng_at_training_start)
    indices = _validated_initial_batch_indices(
        initial_batch_indices,
        dataset_item_count=dataset_item_count,
        seed=seed,
    )
    inputs = _validated_initial_batch_inputs(
        initial_batch_inputs,
        indices=indices,
    )
    used_code_counts = _normalize_used_code_counts(
        initial_batch_used_code_counts
    )
    codebook_state = _post_kmeans_codebook_state(post_state)
    if len(used_code_counts) != len(codebook_state):
        raise ValueError("used code counts must match the number of codebooks")
    normalized_epoch_hashes = _normalize_epoch_hashes(epoch_hashes)
    deterministic_epoch_hashes = epoch_plan_hashes(
        dataset_item_count,
        seed,
        500,
    )
    if normalized_epoch_hashes != deterministic_epoch_hashes:
        raise ValueError("epoch hashes do not match the deterministic epoch plan")
    calculated_rolling_hash = hash_json(normalized_epoch_hashes)
    if calculated_rolling_hash != epoch_plan_rolling_hash:
        raise ValueError("epoch plan rolling hash mismatch")

    config_hash = hash_json(config_payload)
    initial_model_hash = hash_state_dict(initial_state)
    post_model_hash = hash_state_dict(post_state)
    codebook_hash = _hash_codebook_state(post_state)
    rng_before_hash = hash_rng_state(rng_before)
    rng_training_hash = hash_rng_state(rng_training)
    indices_hash = hash_tensor(indices)
    input_hash = hash_tensor(inputs)
    compatibility = _compatibility_payload(
        dataset_sha256=dataset_sha256,
        dataset_item_count=dataset_item_count,
        seed=seed,
        initialization_config_hash_value=config_hash,
        initial_model_hash=initial_model_hash,
        post_kmeans_model_hash=post_model_hash,
        post_kmeans_codebook_hash=codebook_hash,
        initial_batch_indices_hash=indices_hash,
        initial_batch_input_hash=input_hash,
        rng_at_training_start_hash=rng_training_hash,
        epoch_plan_rolling_hash=calculated_rolling_hash,
    )
    artifact = {
        "schema_version": 1,
        "artifact_kind": "rqvae_common_initialization",
        "dataset": {
            "name": "amazon-beauty",
            "item_count": dataset_item_count,
            "sha256": dataset_sha256,
        },
        "seed": seed,
        "initialization_config": config_payload,
        "initialization_config_hash": config_hash,
        "initial_model_state": initial_state,
        "initial_model_hash": initial_model_hash,
        "post_kmeans_model_state": post_state,
        "post_kmeans_model_hash": post_model_hash,
        "post_kmeans_codebook_hash": codebook_hash,
        "rng_before_model_initialization": rng_before,
        "rng_before_model_initialization_hash": rng_before_hash,
        "rng_at_training_start": rng_training,
        "rng_at_training_start_hash": rng_training_hash,
        "initial_batch": {
            "indices": indices,
            "indices_hash": indices_hash,
            "input_tensor_hash": input_hash,
            "used_code_counts": used_code_counts,
        },
        "epoch_plan_hashes": normalized_epoch_hashes,
        "epoch_plan_rolling_hash": calculated_rolling_hash,
        "compatibility": compatibility,
        "common_initialization_hash": hash_json(compatibility),
    }

    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(artifact, temporary_path)
        os.link(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return artifact


def _require_exact_fields(
    payload: Mapping[str, object],
    expected_fields: frozenset[str],
    label: str,
) -> None:
    if set(payload) != expected_fields:
        raise ValueError(f"{label} has an invalid schema")


def _require_equal(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch")


def _validate_model_load_target(
    model: nn.Module,
    initial_state: Mapping[str, torch.Tensor],
    post_state: Mapping[str, torch.Tensor],
    initial_model_hash: str,
) -> None:
    current_state = clone_cpu_state_dict(model)
    _validate_state_layouts(current_state, post_state)
    if hash_state_dict(current_state) != initial_model_hash:
        raise ValueError("model does not match the common initial model hash")
    if set(current_state) != set(initial_state):
        raise ValueError("model state keys do not match the stored initial state")
    layers = getattr(model, "layers", None)
    if layers is None or any(
        not hasattr(layer, "kmeans_initted") for layer in layers
    ):
        raise ValueError("model does not expose RQ-VAE kmeans flags")


def load_common_initialization(
    path: Path,
    *,
    model: nn.Module,
    expected_dataset_sha256: str,
    expected_dataset_item_count: int,
    expected_seed: int,
    expected_initialization_config_hash: str,
    expected_epoch_plan_rolling_hash: str,
) -> dict[str, Any]:
    artifact = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(artifact, Mapping):
        raise ValueError("common initialization artifact must be a mapping")
    _require_exact_fields(
        artifact,
        _COMMON_ARTIFACT_FIELDS,
        "common initialization artifact",
    )
    _require_equal("schema version", artifact["schema_version"], 1)
    _require_equal(
        "artifact kind",
        artifact["artifact_kind"],
        "rqvae_common_initialization",
    )

    dataset = artifact["dataset"]
    if not isinstance(dataset, Mapping) or set(dataset) != {
        "name",
        "item_count",
        "sha256",
    }:
        raise ValueError("dataset metadata has an invalid schema")
    _require_equal("dataset name", dataset["name"], "amazon-beauty")
    _require_equal(
        "dataset SHA-256",
        dataset["sha256"],
        expected_dataset_sha256,
    )
    _require_equal(
        "dataset item count",
        dataset["item_count"],
        expected_dataset_item_count,
    )
    _require_equal("seed", artifact["seed"], expected_seed)

    initial_state = _clone_cpu_state_mapping(artifact["initial_model_state"])
    post_state = _clone_cpu_state_mapping(artifact["post_kmeans_model_state"])
    _validate_state_layouts(initial_state, post_state)
    calculated_initial_model_hash = hash_state_dict(initial_state)
    calculated_post_model_hash = hash_state_dict(post_state)
    calculated_codebook_hash = _hash_codebook_state(post_state)
    _require_equal(
        "initial model hash",
        artifact["initial_model_hash"],
        calculated_initial_model_hash,
    )
    _require_equal(
        "post-kmeans model hash",
        artifact["post_kmeans_model_hash"],
        calculated_post_model_hash,
    )
    _require_equal(
        "post-kmeans codebook hash",
        artifact["post_kmeans_codebook_hash"],
        calculated_codebook_hash,
    )

    if type(artifact["initialization_config"]) is not dict:
        raise TypeError("initialization config must be a dict")
    config_payload = _validated_initialization_config_payload(
        artifact["initialization_config"],
        dataset_sha256=dataset["sha256"],
        dataset_item_count=dataset["item_count"],
        seed=artifact["seed"],
    )
    config_hash = hash_json(config_payload)
    _require_equal(
        "initialization config hash",
        artifact["initialization_config_hash"],
        config_hash,
    )
    _require_equal(
        "expected initialization config hash",
        config_hash,
        expected_initialization_config_hash,
    )

    rng_before = _clone_rng_state(artifact["rng_before_model_initialization"])
    rng_training = _clone_rng_state(artifact["rng_at_training_start"])
    calculated_rng_before_hash = hash_rng_state(rng_before)
    calculated_rng_training_hash = hash_rng_state(rng_training)
    _require_equal(
        "pre-initialization RNG hash",
        artifact["rng_before_model_initialization_hash"],
        calculated_rng_before_hash,
    )
    _require_equal(
        "training-start RNG hash",
        artifact["rng_at_training_start_hash"],
        calculated_rng_training_hash,
    )

    initial_batch = artifact["initial_batch"]
    if not isinstance(initial_batch, Mapping):
        raise ValueError("initial batch must be a mapping")
    _require_exact_fields(
        initial_batch,
        frozenset(
            {
                "indices",
                "indices_hash",
                "input_tensor_hash",
                "used_code_counts",
            }
        ),
        "initial batch",
    )
    if not isinstance(initial_batch["indices"], Tensor):
        raise TypeError("initial batch indices must be a torch.Tensor")
    calculated_indices_hash = hash_tensor(initial_batch["indices"])
    _require_equal(
        "initial batch indices hash",
        initial_batch["indices_hash"],
        calculated_indices_hash,
    )
    validated_indices = _validated_initial_batch_indices(
        initial_batch["indices"],
        dataset_item_count=dataset["item_count"],
        seed=artifact["seed"],
    )
    if not torch.equal(validated_indices, initial_batch["indices"]):
        raise ValueError("initial batch indices changed during validation")
    if type(initial_batch["input_tensor_hash"]) is not str:
        raise TypeError("initial batch input hash must be a string")
    if type(initial_batch["used_code_counts"]) is not list:
        raise TypeError("initial batch used_code_counts must be a list")
    used_code_counts = _normalize_used_code_counts(
        initial_batch["used_code_counts"]
    )
    if len(used_code_counts) != len(_post_kmeans_codebook_state(post_state)):
        raise ValueError(
            "initial batch used_code_counts must match the number of codebooks"
        )

    if type(artifact["epoch_plan_hashes"]) is not list:
        raise TypeError("epoch_plan_hashes must be a list")
    normalized_epoch_hashes = _normalize_epoch_hashes(
        artifact["epoch_plan_hashes"]
    )
    deterministic_epoch_hashes = epoch_plan_hashes(
        dataset["item_count"],
        artifact["seed"],
        500,
    )
    if normalized_epoch_hashes != deterministic_epoch_hashes:
        raise ValueError("epoch hashes do not match the deterministic epoch plan")
    calculated_epoch_rolling_hash = hash_json(normalized_epoch_hashes)
    _require_equal(
        "epoch plan rolling hash",
        artifact["epoch_plan_rolling_hash"],
        calculated_epoch_rolling_hash,
    )
    _require_equal(
        "expected epoch plan rolling hash",
        calculated_epoch_rolling_hash,
        expected_epoch_plan_rolling_hash,
    )

    compatibility = artifact["compatibility"]
    if not isinstance(compatibility, Mapping):
        raise ValueError("compatibility payload must be a mapping")
    _require_exact_fields(
        compatibility,
        _COMPATIBILITY_FIELDS,
        "compatibility payload",
    )
    expected_compatibility = _compatibility_payload(
        dataset_sha256=dataset["sha256"],
        dataset_item_count=dataset["item_count"],
        seed=artifact["seed"],
        initialization_config_hash_value=config_hash,
        initial_model_hash=calculated_initial_model_hash,
        post_kmeans_model_hash=calculated_post_model_hash,
        post_kmeans_codebook_hash=calculated_codebook_hash,
        initial_batch_indices_hash=calculated_indices_hash,
        initial_batch_input_hash=initial_batch["input_tensor_hash"],
        rng_at_training_start_hash=calculated_rng_training_hash,
        epoch_plan_rolling_hash=calculated_epoch_rolling_hash,
    )
    _require_equal(
        "compatibility payload",
        dict(compatibility),
        expected_compatibility,
    )
    _require_equal(
        "common initialization hash",
        artifact["common_initialization_hash"],
        hash_json(expected_compatibility),
    )

    _validate_model_load_target(
        model,
        initial_state,
        post_state,
        calculated_initial_model_hash,
    )

    model.load_state_dict(post_state, strict=True)
    for layer in model.layers:
        layer.kmeans_initted = True
    restore_rng_state(rng_training)
    return dict(artifact)



class FailureStatus(str, Enum):
    COMPLETE = "complete"
    NUMERICAL_FAILURE = "numerical_failure"
    OOM = "oom"
    INVARIANT_MISMATCH = "invariant_mismatch"
    DEADLINE_STOP = "deadline_stop"
    OUTPUT_EXISTS = "output_exists"
    INTERRUPTED = "interrupted"
    RUNTIME_ERROR = "runtime_error"


class CollapseClass(str, Enum):
    NEVER = "never"
    TRANSIENT_RECOVERED = "transient_recovered"
    FINAL_ONLY = "final_only"
    SUSTAINED = "sustained"
    NOT_EVALUABLE = "not_evaluable"


def _require_nonempty_matrix(field_name: str, values: Tensor) -> Tensor:
    if not isinstance(values, Tensor):
        raise TypeError(f"{field_name} must be a torch.Tensor")
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"{field_name} must be a nonempty two-dimensional tensor")
    return values.detach().float()


def _linear_norm_quantiles(
    values: Tensor,
    quantiles: Sequence[float],
) -> list[float]:
    norms = values.norm(dim=1)
    return [
        float(torch.quantile(norms, quantile, interpolation="linear").item())
        for quantile in quantiles
    ]


def input_distribution_stats(inputs: Tensor) -> dict[str, Any]:
    values = _require_nonempty_matrix("inputs", inputs)
    norms = values.norm(dim=1)
    return {
        "global_variance": float(values.var(unbiased=False).item()),
        "mean_feature_variance": float(
            values.var(dim=0, unbiased=False).mean().item()
        ),
        "norm_mean": float(norms.mean().item()),
        "norm_std": float(norms.std(unbiased=False).item()),
        "norm_min": float(norms.min().item()),
        "norm_max": float(norms.max().item()),
    }


def encoder_distribution_stats(encoded: Tensor) -> dict[str, Any]:
    values = _require_nonempty_matrix("encoded", encoded)
    p00, p50, p95, p99, p100 = _linear_norm_quantiles(
        values,
        (0.0, 0.50, 0.95, 0.99, 1.0),
    )
    return {
        "global_variance": float(values.var(unbiased=False).item()),
        "norm_p00": p00,
        "norm_p50": p50,
        "norm_p95": p95,
        "norm_p99": p99,
        "norm_p100": p100,
    }


def residual_distribution_stats(
    residuals_by_layer: Sequence[Tensor],
) -> list[dict[str, Any]]:
    if isinstance(residuals_by_layer, (str, bytes)) or not isinstance(
        residuals_by_layer,
        Sequence,
    ):
        raise TypeError("residuals_by_layer must be a sequence")
    stats: list[dict[str, Any]] = []
    for layer_index, residual in enumerate(residuals_by_layer):
        values = _require_nonempty_matrix(
            f"residuals_by_layer[{layer_index}]",
            residual,
        )
        norms = values.norm(dim=1)
        p50, p95 = _linear_norm_quantiles(values, (0.50, 0.95))
        stats.append(
            {
                "variance": float(values.var(unbiased=False).item()),
                "norm_mean": float(norms.mean().item()),
                "norm_p50": p50,
                "norm_p95": p95,
            }
        )
    if not stats:
        raise ValueError("residuals_by_layer must not be empty")
    return stats


def _nearest_rank(values: Tensor, quantile: float) -> int:
    ordered = values.detach().to(device="cpu", dtype=torch.long).sort().values
    rank = max(1, math.ceil(quantile * ordered.numel()))
    return int(ordered[rank - 1].item())


def assignment_diagnostics(
    semantic_ids: Tensor,
    *,
    codebook_size: int = 256,
) -> dict[str, Any]:
    if type(codebook_size) is not int or codebook_size != 256:
        raise ValueError("diagnostic codebook_size must equal paper-strict 256")
    if not isinstance(semantic_ids, Tensor):
        raise TypeError("semantic_ids must be a torch.Tensor")
    codes = semantic_ids.detach().to(device="cpu", dtype=torch.long)
    if codes.ndim != 2 or codes.shape[1] != 3 or codes.shape[0] == 0:
        raise ValueError("semantic_ids must have shape [items, 3] and be nonempty")
    if int(codes.min().item()) < 0 or int(codes.max().item()) >= codebook_size:
        raise ValueError("semantic IDs must be within the codebook range")

    item_count = int(codes.shape[0])
    layer_stats: list[dict[str, Any]] = []
    for layer_index in range(codes.shape[1]):
        counts = torch.bincount(codes[:, layer_index], minlength=codebook_size)
        probabilities = counts.float() / counts.sum()
        nonzero = probabilities[probabilities > 0]
        entropy = float(-(nonzero * nonzero.log()).sum().item())
        used_count = int((counts > 0).sum().item())
        layer_stats.append(
            {
                "layer_index": layer_index,
                "used_count": used_count,
                "used_ratio": used_count / codebook_size,
                "entropy": entropy,
                "normalized_entropy": entropy / math.log(codebook_size),
                "top1_mass": float(probabilities.max().item()),
                "top10_mass": float(
                    probabilities.topk(min(10, len(probabilities))).values.sum().item()
                ),
            }
        )

    _, bucket_sizes = torch.unique(codes, dim=0, return_counts=True)
    unique_three_token_count = int(bucket_sizes.numel())
    max_bucket_size = int(bucket_sizes.max().item())
    collision_gate_passed = True
    try:
        fixed = build_fixed_four_token_ids(
            codes,
            collision_cardinality=256,
        )
        unique_four_token_count = int(
            torch.unique(fixed.four_token_ids, dim=0).shape[0]
        )
    except FixedCollisionOverflow:
        collision_gate_passed = False
        unique_four_token_count = 0

    used_counts = [int(layer["used_count"]) for layer in layer_stats]
    used_ratios = [float(layer["used_ratio"]) for layer in layer_stats]
    hard_collapse = (
        max_bucket_size / item_count >= 0.90
        or any(
            used_count * 100 <= codebook_size
            for used_count in used_counts
        )
    )
    local_promising = (
        all(ratio >= 0.20 for ratio in used_ratios)
        and unique_three_token_count >= 1024
        and max_bucket_size <= 256
    )
    paper_gate_passed = (
        all(ratio >= 0.80 for ratio in used_ratios)
        and max_bucket_size <= 256
        and unique_four_token_count == 12101
    )
    return {
        "item_count": item_count,
        "codebook_size": codebook_size,
        "layers": layer_stats,
        "used_counts": used_counts,
        "used_ratios": used_ratios,
        "unique_three_token_count": unique_three_token_count,
        "unique_four_token_count": unique_four_token_count,
        "max_bucket_size": max_bucket_size,
        "fixed_collision_cardinality": 256,
        "collision_gate_passed": collision_gate_passed,
        "bucket_size_p50": _nearest_rank(bucket_sizes, 0.50),
        "bucket_size_p95": _nearest_rank(bucket_sizes, 0.95),
        "bucket_size_p99": _nearest_rank(bucket_sizes, 0.99),
        "hard_collapse": hard_collapse,
        "local_promising": local_promising,
        "paper_gate_passed": paper_gate_passed,
    }


def parameter_groups(
    model: nn.Module,
) -> dict[str, list[tuple[str, nn.Parameter]]]:
    keys = [
        "encoder",
        "decoder",
        "layers.0.embedding.weight",
        "layers.1.embedding.weight",
        "layers.2.embedding.weight",
    ]
    groups: dict[str, list[tuple[str, nn.Parameter]]] = {
        key: [] for key in keys
    }
    unassigned: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("encoder."):
            group_name = "encoder"
        elif name.startswith("decoder."):
            group_name = "decoder"
        elif name in groups and name.startswith("layers."):
            group_name = name
        else:
            unassigned.append(name)
            continue
        groups[group_name].append((name, parameter))
    if unassigned:
        raise ValueError(f"unassigned trainable parameters: {sorted(unassigned)}")
    empty = [name for name, values in groups.items() if not values]
    if empty:
        raise ValueError(f"required parameter groups are empty: {empty}")
    return groups


def gradient_group_stats(
    groups: Mapping[str, Sequence[tuple[str, nn.Parameter]]],
) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    for group_name, named_parameters in groups.items():
        parameters = list(named_parameters)
        if not parameters:
            raise ValueError(f"parameter group {group_name} must not be empty")
        total_count = sum(parameter.numel() for _, parameter in parameters)
        present = [parameter.grad for _, parameter in parameters if parameter.grad is not None]
        if not present:
            result[group_name] = {
                "l2_norm": None,
                "max_abs": None,
                "exact_zero_fraction": None,
                "nonfinite_fraction": None,
            }
            continue
        zero_count = sum(
            parameter.numel()
            for _, parameter in parameters
            if parameter.grad is None
        )
        nonfinite_count = 0
        squared_sum = 0.0
        max_abs = 0.0
        for gradient in present:
            value = gradient.detach()
            finite = torch.isfinite(value)
            nonfinite_count += int((~finite).sum().item())
            zero_count += int((value == 0).sum().item())
            if bool(finite.all().item()):
                squared_sum += float(value.double().square().sum().item())
                max_abs = max(max_abs, float(value.abs().max().item()))
        nonfinite_fraction = nonfinite_count / total_count
        result[group_name] = {
            "l2_norm": None if nonfinite_count else math.sqrt(squared_sum),
            "max_abs": None if nonfinite_count else max_abs,
            "exact_zero_fraction": zero_count / total_count,
            "nonfinite_fraction": nonfinite_fraction,
        }
    return result


def clone_group_parameters(
    groups: Mapping[str, Sequence[tuple[str, nn.Parameter]]],
) -> dict[str, list[Tensor]]:
    return {
        group_name: [parameter.detach().clone() for _, parameter in values]
        for group_name, values in groups.items()
    }


def parameter_delta_stats(
    groups: Mapping[str, Sequence[tuple[str, nn.Parameter]]],
    before: Mapping[str, Sequence[Tensor]],
) -> dict[str, dict[str, float]]:
    if set(groups) != set(before):
        raise ValueError("parameter delta group keys differ")
    result: dict[str, dict[str, float]] = {}
    for group_name, named_parameters in groups.items():
        parameters = list(named_parameters)
        previous = list(before[group_name])
        if len(parameters) != len(previous):
            raise ValueError(f"parameter delta group length differs for {group_name}")
        delta_squared = 0.0
        pre_squared = 0.0
        for (name, parameter), old in zip(parameters, previous, strict=True):
            if not isinstance(old, Tensor) or old.shape != parameter.shape:
                raise ValueError(f"parameter delta shape differs for {name}")
            old_on_device = old.detach().to(
                device=parameter.device,
                dtype=parameter.dtype,
            )
            delta_squared += float(
                (parameter.detach() - old_on_device).double().square().sum().item()
            )
            pre_squared += float(old_on_device.double().square().sum().item())
        delta_l2 = math.sqrt(delta_squared)
        pre_update_l2 = math.sqrt(pre_squared)
        result[group_name] = {
            "delta_l2": delta_l2,
            "pre_update_parameter_l2": pre_update_l2,
            "relative_update": delta_l2 / max(pre_update_l2, 1e-12),
        }
    return result


def codebook_weight_stats(
    model: nn.Module,
    post_kmeans_state: Mapping[str, Tensor],
) -> dict[str, dict[str, float]]:
    parameters = dict(model.named_parameters())
    result: dict[str, dict[str, float]] = {}
    for layer_index in range(3):
        name = f"layers.{layer_index}.embedding.weight"
        if name not in parameters or name not in post_kmeans_state:
            raise ValueError(f"missing codebook baseline for {name}")
        value = parameters[name].detach()
        baseline = post_kmeans_state[name].detach().to(
            device=value.device,
            dtype=value.dtype,
        )
        if value.shape != baseline.shape:
            raise ValueError(f"codebook baseline shape differs for {name}")
        result[name] = {
            "weight_variance": float(value.float().var(unbiased=False).item()),
            "weight_l2_norm": float(value.double().norm().item()),
            "movement_l2": float((value - baseline).double().norm().item()),
        }
    return result


def scheduled_snapshot_triggers(
    *,
    optimizer_step: int,
    completed_epoch: int | None,
    step_schedule: frozenset[int],
    epoch_schedule: frozenset[int],
) -> tuple[str, ...]:
    if type(optimizer_step) is not int or optimizer_step < 0:
        raise ValueError("optimizer_step must be a nonnegative int")
    if completed_epoch is not None and (
        type(completed_epoch) is not int or completed_epoch < 0
    ):
        raise ValueError("completed_epoch must be a nonnegative int or None")
    if type(step_schedule) is not frozenset or any(
        type(value) is not int or value < 0 for value in step_schedule
    ):
        raise ValueError("step_schedule must be a frozenset of nonnegative ints")
    if type(epoch_schedule) is not frozenset or any(
        type(value) is not int or value <= 0 for value in epoch_schedule
    ):
        raise ValueError("epoch_schedule must be a frozenset of positive ints")
    triggers: list[str] = []
    if optimizer_step in step_schedule:
        triggers.append(f"optimizer_step:{optimizer_step}")
    if completed_epoch is not None and completed_epoch in epoch_schedule:
        triggers.append(f"epoch:{completed_epoch}")
    return tuple(triggers)


def classify_collapse(
    epoch_hard_collapse: Mapping[int, bool],
    *,
    failure_status: FailureStatus,
) -> dict[str, Any]:
    if not isinstance(epoch_hard_collapse, Mapping) or any(
        type(epoch) is not int
        or epoch <= 0
        or type(value) is not bool
        for epoch, value in epoch_hard_collapse.items()
    ):
        raise ValueError("epoch_hard_collapse must map positive ints to bools")
    if not isinstance(failure_status, FailureStatus):
        raise TypeError("failure_status must be a FailureStatus")
    ever = any(epoch_hard_collapse.values())
    if failure_status is not FailureStatus.COMPLETE or 500 not in epoch_hard_collapse:
        return {
            "ever_hard_collapse": ever,
            "final_hard_collapse": None,
            "sustained_hard_collapse": None,
            "collapse_class": CollapseClass.NOT_EVALUABLE.value,
            "failure_status": failure_status.value,
        }
    final = epoch_hard_collapse[500]
    sustained = all(epoch_hard_collapse.get(epoch, False) for epoch in (100, 250, 500))
    if sustained:
        collapse_class = CollapseClass.SUSTAINED
    elif final:
        collapse_class = CollapseClass.FINAL_ONLY
    elif ever:
        collapse_class = CollapseClass.TRANSIENT_RECOVERED
    else:
        collapse_class = CollapseClass.NEVER
    return {
        "ever_hard_collapse": ever,
        "final_hard_collapse": final,
        "sustained_hard_collapse": sustained,
        "collapse_class": collapse_class.value,
        "failure_status": failure_status.value,
    }


def capture_read_only_corpus_snapshot(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    canonical_loader: DataLoader,
    diagnostic_loader_generator: torch.Generator,
    post_kmeans_state: Mapping[str, Tensor],
    optimizer_step: int,
    completed_epoch: int | None,
    triggers: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(canonical_loader, DataLoader):
        raise TypeError("canonical_loader must be a DataLoader")
    if not isinstance(diagnostic_loader_generator, torch.Generator):
        raise TypeError("diagnostic_loader_generator must be a torch.Generator")
    if isinstance(triggers, (str, bytes)) or not isinstance(triggers, Sequence):
        raise TypeError("triggers must be a sequence of strings")
    trigger_list = list(triggers)
    if any(type(value) is not str for value in trigger_list):
        raise TypeError("triggers must contain only strings")
    if not hasattr(model, "layers") or any(
        not bool(getattr(layer, "kmeans_initted", False))
        for layer in model.layers
    ):
        raise ValueError(
            "read-only corpus diagnostics require a post-kmeans model"
        )

    was_training = model.training
    kmeans_states = tuple(layer.kmeans_initted for layer in model.layers)
    model_state_before = clone_cpu_state_dict(model)
    model_hash_before = hash_state_dict(model_state_before)
    optimizer_state_before = copy.deepcopy(optimizer.state_dict())
    optimizer_hash_before = hash_nested_state(optimizer_state_before)
    rng_before = capture_rng_state()
    rng_hash_before = hash_rng_state(rng_before)
    generator_state = (
        diagnostic_loader_generator.get_state().detach().cpu().clone()
    )
    generator_hash_before = hash_tensor(generator_state)

    all_inputs: list[Tensor] = []
    all_encoded: list[Tensor] = []
    all_semantic_ids: list[Tensor] = []
    residuals_by_layer: list[list[Tensor]] | None = None
    collection_error: BaseException | None = None
    collection_traceback: Any = None
    mutation_checks: dict[str, bool] = {}
    restoration_error: RuntimeError | None = None
    try:
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            for batch in canonical_loader:
                inputs = _extract_item_tensor(batch)
                device_inputs = inputs.to(device)
                encoded = model.encode(device_inputs)
                output = model.get_semantic_ids(device_inputs)
                all_inputs.append(inputs.detach().cpu())
                all_encoded.append(encoded.detach().cpu())
                all_semantic_ids.append(output.sem_ids.detach().cpu())
                residual_layers = list(output.residuals.unbind(dim=-1))
                if residuals_by_layer is None:
                    residuals_by_layer = [list() for _ in residual_layers]
                if len(residual_layers) != len(residuals_by_layer):
                    raise ValueError(
                        "RQ-VAE residual layer count changed across batches"
                    )
                for values, residual in zip(
                    residuals_by_layer,
                    residual_layers,
                    strict=True,
                ):
                    values.append(residual.detach().cpu())
    except BaseException as error:
        collection_error = error
        collection_traceback = error.__traceback__
    finally:
        observed_model_hash = hash_state_dict(clone_cpu_state_dict(model))
        observed_optimizer_hash = hash_nested_state(optimizer.state_dict())
        observed_kmeans_states = tuple(
            layer.kmeans_initted for layer in model.layers
        )
        mutation_checks = {
            "model_hash": observed_model_hash == model_hash_before,
            "optimizer_state_hash": (
                observed_optimizer_hash == optimizer_hash_before
            ),
            "kmeans_state": observed_kmeans_states == kmeans_states,
        }
        if not mutation_checks["model_hash"]:
            model.load_state_dict(model_state_before, strict=True)
        if not mutation_checks["optimizer_state_hash"]:
            optimizer.load_state_dict(copy.deepcopy(optimizer_state_before))
        for layer, state in zip(model.layers, kmeans_states, strict=True):
            layer.kmeans_initted = state
        diagnostic_loader_generator.set_state(generator_state)
        restore_rng_state(rng_before)
        model.train(was_training)

        restoration_checks = {
            "model_hash": (
                hash_state_dict(clone_cpu_state_dict(model))
                == model_hash_before
            ),
            "optimizer_state_hash": (
                hash_nested_state(optimizer.state_dict())
                == optimizer_hash_before
            ),
            "rng_hash": (
                hash_rng_state(capture_rng_state()) == rng_hash_before
            ),
            "diagnostic_loader_generator_hash": (
                hash_tensor(diagnostic_loader_generator.get_state())
                == generator_hash_before
            ),
            "model_mode": model.training is was_training,
            "kmeans_state": (
                tuple(layer.kmeans_initted for layer in model.layers)
                == kmeans_states
            ),
        }
        if not all(restoration_checks.values()):
            failed = sorted(
                name
                for name, passed in restoration_checks.items()
                if not passed
            )
            restoration_error = RuntimeError(
                "failed to restore read-only corpus diagnostic state: "
                f"{failed}"
            )

    if collection_error is not None:
        if restoration_error is not None:
            collection_error.add_note(str(restoration_error))
        raise collection_error.with_traceback(collection_traceback)
    if restoration_error is not None:
        raise restoration_error
    if not all(mutation_checks.values()):
        failed = sorted(
            name for name, passed in mutation_checks.items() if not passed
        )
        raise RuntimeError(
            f"read-only corpus diagnostics mutated state: {failed}"
        )
    if not all_inputs or residuals_by_layer is None:
        raise ValueError("canonical_loader must yield at least one batch")

    inputs = torch.cat(all_inputs, dim=0)
    encoded = torch.cat(all_encoded, dim=0)
    semantic_ids = torch.cat(all_semantic_ids, dim=0)
    residuals = [torch.cat(values, dim=0) for values in residuals_by_layer]
    return {
        "optimizer_step": optimizer_step,
        "completed_epoch": completed_epoch,
        "triggers": trigger_list,
        "item_count": int(inputs.shape[0]),
        "input_distribution": input_distribution_stats(inputs),
        "encoder_distribution": encoder_distribution_stats(encoded),
        "residual_distributions": residual_distribution_stats(residuals),
        "assignment_diagnostics": assignment_diagnostics(semantic_ids),
        "codebook_weight_stats": codebook_weight_stats(
            model,
            post_kmeans_state,
        ),
        "non_mutation_checks": {
            "passed": True,
            "model_hash": model_hash_before,
            "optimizer_state_hash": optimizer_hash_before,
            "rng_hash": rng_hash_before,
            "diagnostic_loader_generator_hash": generator_hash_before,
        },
    }
_CHECKPOINT_ARTIFACT_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "model_state",
        "optimizer_state",
        "rng_state",
        "model_state_hash",
        "optimizer_state_hash",
        "rng_state_hash",
        "training",
        "optimizer",
        "compatibility",
        "complete_config_hash",
        "epoch_plan_hashes",
        "snapshot_history",
    }
)
_CHECKPOINT_TRAINING_FIELDS = frozenset({"completed_epoch", "optimizer_step"})
_CHECKPOINT_OPTIMIZER_FIELDS = frozenset(
    {"treatment", "treatment_hash", "metadata", "metadata_hash"}
)
_CHECKPOINT_COMPATIBILITY_FIELDS = frozenset(
    {
        "optimizer_treatment_hash",
        "common_initialization_hash",
        "invariant_config_hash",
        "dataset_sha256",
        "seed",
        "epoch_plan_rolling_hash",
    }
)
_CHECKPOINT_SNAPSHOT_FIELDS = frozenset(
    {"path", "record_count", "rolling_hash"}
)


def _require_nonnegative_int(field_name: str, value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a nonnegative int")
    return value


def _require_sha256(field_name: str, value: object) -> str:
    if type(value) is not str or len(value) != 64:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex string")
    if value != value.lower() or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex string")
    return value


def _clone_optimizer_state(value: object) -> object:
    if isinstance(value, Tensor):
        return value.detach().to(device="cpu").clone()
    if isinstance(value, Mapping):
        return {
            copy.deepcopy(key): _clone_optimizer_state(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_clone_optimizer_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_optimizer_state(item) for item in value)
    return copy.deepcopy(value)


def _optimizer_parameters(
    optimizer: torch.optim.Optimizer,
) -> list[nn.Parameter]:
    return [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]


def _validate_optimizer_owns_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    model_parameters = list(model.parameters())
    optimizer_parameters = _optimizer_parameters(optimizer)
    if len({id(value) for value in optimizer_parameters}) != len(
        optimizer_parameters
    ):
        raise ValueError("diagnostic optimizer contains duplicate parameters")
    if [id(value) for value in optimizer_parameters] != [
        id(value) for value in model_parameters
    ]:
        raise ValueError(
            "diagnostic optimizer parameters do not exactly match model parameters"
        )


def _validate_optimizer_matches_spec(
    optimizer: torch.optim.Optimizer,
    spec: OptimizerSpec,
) -> dict[str, Any]:
    canonical_spec = _canonical_optimizer_spec(spec)
    expected_type: type[torch.optim.Optimizer]
    if canonical_spec.name == "adagrad":
        expected_type = torch.optim.Adagrad
    else:
        expected_type = torch.optim.AdamW
    if type(optimizer) is not expected_type:
        raise ValueError("optimizer metadata class does not match treatment")
    if len(optimizer.param_groups) != 1:
        raise ValueError("diagnostic optimizer must contain exactly one param group")

    expected_values: dict[str, object] = {
        "lr": canonical_spec.learning_rate,
        "weight_decay": canonical_spec.weight_decay,
        "eps": canonical_spec.eps,
    }
    if canonical_spec.name == "adagrad":
        expected_values["initial_accumulator_value"] = (
            canonical_spec.initial_accumulator_value
        )
    else:
        expected_values["betas"] = canonical_spec.betas
    for field_name, expected in expected_values.items():
        if optimizer.defaults.get(field_name) != expected:
            raise ValueError(
                f"optimizer metadata defaults {field_name} does not match treatment"
            )
        if optimizer.param_groups[0].get(field_name) != expected:
            raise ValueError(
                f"optimizer metadata param group {field_name} "
                "does not match treatment"
            )
    return optimizer_metadata(optimizer)


def _snapshot_jsonl_metadata(path: Path) -> tuple[int, str]:
    snapshot_path = Path(path)
    data = snapshot_path.read_bytes()
    if data and not data.endswith(b"\n"):
        raise ValueError(
            "snapshot history final record must include its terminating newline"
        )
    records = data.splitlines(keepends=True)
    for index, record in enumerate(records, start=1):
        record_bytes = record[:-1]
        if not record_bytes.strip():
            raise ValueError(f"snapshot history record {index} must not be blank")
        try:
            record_text = record_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(
                f"snapshot history record {index} must be valid UTF-8"
            ) from error
        try:
            json.loads(record_text)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"snapshot history record {index} must contain valid JSON"
            ) from error
    return len(records), hashlib.sha256(data).hexdigest()


def _validate_snapshot_history(
    path: Path,
    *,
    expected_record_count: int,
    expected_rolling_hash: str,
) -> None:
    record_count = _require_nonnegative_int(
        "snapshot record count",
        expected_record_count,
    )
    rolling_hash = _require_sha256(
        "snapshot rolling hash",
        expected_rolling_hash,
    )
    actual_record_count, actual_rolling_hash = _snapshot_jsonl_metadata(path)
    if actual_record_count != record_count:
        raise ValueError("snapshot history record count mismatch")
    if actual_rolling_hash != rolling_hash:
        raise ValueError("snapshot history rolling hash mismatch")


def _fsync_parent_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    file_descriptor = os.open(Path(path).parent, flags)
    try:
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)


def _exclusive_torch_publish(path: Path, payload: object) -> None:
    destination = Path(path)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    published = False
    try:
        torch.save(payload, temporary_path)
        with temporary_path.open("rb") as temporary_file:
            os.fsync(temporary_file.fileno())
        os.link(temporary_path, destination)
        published = True
        _fsync_parent_directory(destination)
    except BaseException:
        if published:
            try:
                destination.unlink(missing_ok=True)
                _fsync_parent_directory(destination)
            except OSError:
                pass
        raise
    finally:
        temporary_path.unlink(missing_ok=True)


def save_diagnostic_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    optimizer_spec: OptimizerSpec,
    completed_epoch: int,
    optimizer_step: int,
    rng_state: RngState,
    common_initialization_hash: str,
    invariant_config_hash: str,
    complete_config_hash: str,
    dataset_sha256: str,
    seed: int,
    epoch_hashes: Sequence[str],
    epoch_plan_rolling_hash: str,
    snapshot_jsonl_path: Path,
    snapshot_record_count: int,
    snapshot_rolling_hash: str,
) -> None:
    completed_epoch = _require_nonnegative_int(
        "completed_epoch",
        completed_epoch,
    )
    optimizer_step = _require_nonnegative_int("optimizer_step", optimizer_step)
    if type(seed) is not int:
        raise ValueError("seed must be an int")
    common_initialization_hash = _require_sha256(
        "common initialization hash",
        common_initialization_hash,
    )
    invariant_config_hash = _require_sha256(
        "invariant config hash",
        invariant_config_hash,
    )
    complete_config_hash = _require_sha256(
        "complete config hash",
        complete_config_hash,
    )
    dataset_sha256 = _require_sha256("dataset SHA-256", dataset_sha256)
    epoch_plan_rolling_hash = _require_sha256(
        "epoch plan rolling hash",
        epoch_plan_rolling_hash,
    )
    normalized_epoch_hashes = _normalize_epoch_hashes(epoch_hashes)
    for epoch_hash in normalized_epoch_hashes:
        _require_sha256("epoch plan hash", epoch_hash)
    calculated_epoch_plan_hash = hash_json(normalized_epoch_hashes)
    if calculated_epoch_plan_hash != epoch_plan_rolling_hash:
        raise ValueError("epoch plan rolling hash mismatch")

    snapshot_path = Path(snapshot_jsonl_path)
    _validate_snapshot_history(
        snapshot_path,
        expected_record_count=snapshot_record_count,
        expected_rolling_hash=snapshot_rolling_hash,
    )
    _validate_optimizer_owns_model(model, optimizer)
    live_optimizer_metadata = _validate_optimizer_matches_spec(
        optimizer,
        optimizer_spec,
    )

    model_state = clone_cpu_state_dict(model)
    optimizer_state = _clone_optimizer_state(optimizer.state_dict())
    if not isinstance(optimizer_state, dict):
        raise TypeError("optimizer state must be a dict")
    cloned_rng_state = _clone_rng_state(rng_state)
    treatment = optimizer_treatment_payload(optimizer_spec)
    treatment_hash = optimizer_treatment_hash(optimizer_spec)
    compatibility = {
        "optimizer_treatment_hash": treatment_hash,
        "common_initialization_hash": common_initialization_hash,
        "invariant_config_hash": invariant_config_hash,
        "dataset_sha256": dataset_sha256,
        "seed": seed,
        "epoch_plan_rolling_hash": calculated_epoch_plan_hash,
    }
    artifact = {
        "schema_version": 1,
        "artifact_kind": "rqvae_diagnostic_checkpoint",
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "rng_state": cloned_rng_state,
        "model_state_hash": hash_state_dict(model_state),
        "optimizer_state_hash": hash_nested_state(optimizer_state),
        "rng_state_hash": hash_rng_state(cloned_rng_state),
        "training": {
            "completed_epoch": completed_epoch,
            "optimizer_step": optimizer_step,
        },
        "optimizer": {
            "treatment": treatment,
            "treatment_hash": treatment_hash,
            "metadata": live_optimizer_metadata,
            "metadata_hash": hash_json(live_optimizer_metadata),
        },
        "compatibility": compatibility,
        "complete_config_hash": complete_config_hash,
        "epoch_plan_hashes": normalized_epoch_hashes,
        "snapshot_history": {
            "path": str(snapshot_path),
            "record_count": snapshot_record_count,
            "rolling_hash": snapshot_rolling_hash,
        },
    }
    _exclusive_torch_publish(Path(path), artifact)


def _validate_checkpoint_model_target(
    model: nn.Module,
    model_state: Mapping[str, Tensor],
) -> None:
    current_state = clone_cpu_state_dict(model)
    _validate_state_layouts(current_state, model_state)
    layers = getattr(model, "layers", None)
    if layers is None or any(
        not hasattr(layer, "kmeans_initted") for layer in layers
    ):
        raise ValueError("model does not expose RQ-VAE kmeans flags")


def _validate_optimizer_state_structure(
    optimizer: torch.optim.Optimizer,
    optimizer_state: Mapping[str, object],
    spec: OptimizerSpec,
) -> None:
    if set(optimizer_state) != {"state", "param_groups"}:
        raise ValueError("optimizer state has an invalid schema")
    if not isinstance(optimizer_state["state"], Mapping):
        raise ValueError("optimizer state entries must be a mapping")
    if type(optimizer_state["param_groups"]) is not list:
        raise ValueError("optimizer state param_groups must be a list")
    stored_groups = optimizer_state["param_groups"]
    if len(stored_groups) != len(optimizer.param_groups):
        raise ValueError("optimizer state param group count mismatch")
    for stored_group, live_group in zip(
        stored_groups,
        optimizer.state_dict()["param_groups"],
        strict=True,
    ):
        if not isinstance(stored_group, Mapping):
            raise ValueError("optimizer state param group must be a mapping")
        if set(stored_group) != set(live_group):
            raise ValueError("optimizer state param group schema mismatch")
        if type(stored_group["params"]) is not list:
            raise ValueError("optimizer state params must be a list")
        if len(stored_group["params"]) != len(live_group["params"]):
            raise ValueError("optimizer state parameter count mismatch")
        if {
            key: value for key, value in stored_group.items() if key != "params"
        } != {
            key: value for key, value in live_group.items() if key != "params"
        }:
            raise ValueError("optimizer state param group metadata mismatch")

    dummy_parameters = nn.ParameterList(
        [
            nn.Parameter(
                torch.empty(
                    parameter.shape,
                    dtype=parameter.dtype,
                    device="cpu",
                )
            )
            for parameter in _optimizer_parameters(optimizer)
        ]
    )
    probe = build_diagnostic_optimizer(dummy_parameters, spec)
    try:
        probe.load_state_dict(copy.deepcopy(dict(optimizer_state)))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("optimizer state is incompatible with live optimizer") from error




def _clone_runtime_value(value: object) -> object:
    if isinstance(value, Tensor):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {key: _clone_runtime_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_runtime_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_runtime_value(item) for item in value)
    return copy.deepcopy(value)


def _capture_optimizer_runtime_state(
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    return {
        "state": [
            (parameter, _clone_runtime_value(value))
            for parameter, value in optimizer.state.items()
        ],
        "param_groups": [
            {
                key: list(value) if key == "params" else copy.deepcopy(value)
                for key, value in group.items()
            }
            for group in optimizer.param_groups
        ],
        "defaults": copy.deepcopy(optimizer.defaults),
    }


def _restore_optimizer_runtime_state(
    optimizer: torch.optim.Optimizer,
    snapshot: Mapping[str, Any],
) -> None:
    optimizer.state = defaultdict(
        dict,
        {
            parameter: _clone_runtime_value(value)
            for parameter, value in snapshot["state"]
        },
    )
    optimizer.param_groups = [
        {
            key: list(value) if key == "params" else copy.deepcopy(value)
            for key, value in group.items()
        }
        for group in snapshot["param_groups"]
    ]
    optimizer.defaults = copy.deepcopy(snapshot["defaults"])


def _restore_model_state_direct(
    model: nn.Module,
    snapshot: Mapping[str, Tensor],
) -> None:
    current_state = model.state_dict()
    _validate_state_layouts(current_state, snapshot)
    with torch.no_grad():
        for key, target in current_state.items():
            target.copy_(snapshot[key].to(device=target.device))


def _restore_rng_state_direct(state: RngState) -> None:
    random.setstate(copy.deepcopy(state["python"]))
    np.random.set_state(_clone_numpy_rng_state(state["numpy"]))
    torch.set_rng_state(state["torch_cpu"].detach().to(device="cpu").clone())
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(
            [value.detach().to(device="cpu").clone() for value in state["torch_cuda"]]
        )

def load_diagnostic_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    optimizer_spec: OptimizerSpec,
    expected_common_initialization_hash: str,
    expected_invariant_config_hash: str,
    expected_dataset_sha256: str,
    expected_seed: int,
    expected_epoch_plan_rolling_hash: str,
    expected_snapshot_jsonl_path: Path,
) -> dict[str, int]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("diagnostic checkpoint must be a mapping")
    _require_exact_fields(
        checkpoint,
        _CHECKPOINT_ARTIFACT_FIELDS,
        "diagnostic checkpoint",
    )
    _require_equal("schema version", checkpoint["schema_version"], 1)
    _require_equal(
        "artifact kind",
        checkpoint["artifact_kind"],
        "rqvae_diagnostic_checkpoint",
    )

    expected_treatment_hash = optimizer_treatment_hash(optimizer_spec)
    compatibility = checkpoint["compatibility"]
    if not isinstance(compatibility, Mapping):
        raise ValueError("checkpoint compatibility must be a mapping")
    _require_exact_fields(
        compatibility,
        _CHECKPOINT_COMPATIBILITY_FIELDS,
        "checkpoint compatibility",
    )
    required = {
        "optimizer_treatment_hash": expected_treatment_hash,
        "common_initialization_hash": expected_common_initialization_hash,
        "invariant_config_hash": expected_invariant_config_hash,
        "dataset_sha256": expected_dataset_sha256,
        "seed": expected_seed,
        "epoch_plan_rolling_hash": expected_epoch_plan_rolling_hash,
    }
    for key, expected in required.items():
        if compatibility[key] != expected:
            raise ValueError(f"incompatible diagnostic checkpoint {key}")
    for field_name in (
        "optimizer_treatment_hash",
        "common_initialization_hash",
        "invariant_config_hash",
        "dataset_sha256",
        "epoch_plan_rolling_hash",
    ):
        _require_sha256(field_name, compatibility[field_name])
    if type(compatibility["seed"]) is not int:
        raise ValueError("checkpoint compatibility seed must be an int")

    training = checkpoint["training"]
    if not isinstance(training, Mapping):
        raise ValueError("checkpoint training metadata must be a mapping")
    _require_exact_fields(
        training,
        _CHECKPOINT_TRAINING_FIELDS,
        "checkpoint training metadata",
    )
    completed_epoch = _require_nonnegative_int(
        "completed_epoch",
        training["completed_epoch"],
    )
    optimizer_step = _require_nonnegative_int(
        "optimizer_step",
        training["optimizer_step"],
    )
    _require_sha256("complete config hash", checkpoint["complete_config_hash"])

    if type(checkpoint["epoch_plan_hashes"]) is not list:
        raise TypeError("checkpoint epoch_plan_hashes must be a list")
    normalized_epoch_hashes = _normalize_epoch_hashes(
        checkpoint["epoch_plan_hashes"]
    )
    for epoch_hash in normalized_epoch_hashes:
        _require_sha256("epoch plan hash", epoch_hash)
    calculated_epoch_plan_hash = hash_json(normalized_epoch_hashes)
    if calculated_epoch_plan_hash != compatibility["epoch_plan_rolling_hash"]:
        raise ValueError("checkpoint epoch plan rolling hash mismatch")

    model_state = _clone_cpu_state_mapping(checkpoint["model_state"])
    calculated_model_hash = hash_state_dict(model_state)
    _require_equal(
        "model state hash",
        checkpoint["model_state_hash"],
        calculated_model_hash,
    )
    _validate_checkpoint_model_target(model, model_state)

    if not isinstance(checkpoint["optimizer_state"], Mapping):
        raise ValueError("optimizer state must be a mapping")
    optimizer_state = _clone_optimizer_state(checkpoint["optimizer_state"])
    if not isinstance(optimizer_state, dict):
        raise TypeError("optimizer state must be a dict")
    calculated_optimizer_hash = hash_nested_state(optimizer_state)
    _require_equal(
        "optimizer state hash",
        checkpoint["optimizer_state_hash"],
        calculated_optimizer_hash,
    )

    calculated_rng_hash = hash_nested_state(checkpoint["rng_state"])
    _require_equal(
        "RNG state hash",
        checkpoint["rng_state_hash"],
        calculated_rng_hash,
    )
    try:
        rng_state = _clone_rng_state(checkpoint["rng_state"])
    except RuntimeError as error:
        raise ValueError("checkpoint RNG state is invalid") from error

    optimizer_payload = checkpoint["optimizer"]
    if not isinstance(optimizer_payload, Mapping):
        raise ValueError("checkpoint optimizer metadata must be a mapping")
    _require_exact_fields(
        optimizer_payload,
        _CHECKPOINT_OPTIMIZER_FIELDS,
        "checkpoint optimizer metadata",
    )
    expected_treatment = optimizer_treatment_payload(optimizer_spec)
    _require_equal(
        "optimizer treatment",
        optimizer_payload["treatment"],
        expected_treatment,
    )
    _require_equal(
        "optimizer treatment hash",
        optimizer_payload["treatment_hash"],
        expected_treatment_hash,
    )
    if not isinstance(optimizer_payload["metadata"], Mapping):
        raise ValueError("stored optimizer metadata must be a mapping")
    stored_optimizer_metadata = dict(optimizer_payload["metadata"])
    stored_optimizer_metadata_hash = hash_json(stored_optimizer_metadata)
    _require_equal(
        "optimizer metadata hash",
        optimizer_payload["metadata_hash"],
        stored_optimizer_metadata_hash,
    )
    _validate_optimizer_owns_model(model, optimizer)
    live_optimizer_metadata = _validate_optimizer_matches_spec(
        optimizer,
        optimizer_spec,
    )
    _require_equal(
        "optimizer metadata",
        stored_optimizer_metadata,
        live_optimizer_metadata,
    )
    _validate_optimizer_state_structure(
        optimizer,
        optimizer_state,
        optimizer_spec,
    )

    snapshot_history = checkpoint["snapshot_history"]
    if not isinstance(snapshot_history, Mapping):
        raise ValueError("snapshot history must be a mapping")
    _require_exact_fields(
        snapshot_history,
        _CHECKPOINT_SNAPSHOT_FIELDS,
        "snapshot history",
    )
    expected_snapshot_path = Path(expected_snapshot_jsonl_path)
    if snapshot_history["path"] != str(expected_snapshot_path):
        raise ValueError("snapshot history path mismatch")
    _validate_snapshot_history(
        expected_snapshot_path,
        expected_record_count=snapshot_history["record_count"],
        expected_rolling_hash=snapshot_history["rolling_hash"],
    )

    live_model_state = clone_cpu_state_dict(model)
    live_optimizer_state = _capture_optimizer_runtime_state(optimizer)
    live_rng_state = capture_rng_state()
    live_kmeans_flags = [layer.kmeans_initted for layer in model.layers]
    try:
        optimizer.load_state_dict(optimizer_state)
        model.load_state_dict(model_state, strict=True)
        for layer in model.layers:
            layer.kmeans_initted = True
        restore_rng_state(rng_state)
    except BaseException:
        _restore_model_state_direct(model, live_model_state)
        _restore_optimizer_runtime_state(optimizer, live_optimizer_state)
        for layer, flag in zip(model.layers, live_kmeans_flags, strict=True):
            layer.kmeans_initted = flag
        _restore_rng_state_direct(live_rng_state)
        raise
    return {
        "start_epoch": completed_epoch + 1,
        "optimizer_step": optimizer_step,
    }
