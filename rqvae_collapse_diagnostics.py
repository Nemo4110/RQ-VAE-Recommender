from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import struct
import tempfile
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import TypedDict

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.utils.data import Dataset


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
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("snapshot history must be valid UTF-8") from error
    if data and not data.endswith(b"\n"):
        raise ValueError(
            "snapshot history final record must include its terminating newline"
        )
    return data.count(b"\n"), hashlib.sha256(data).hexdigest()


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


def _exclusive_torch_publish(path: Path, payload: object) -> None:
    destination = Path(path)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        os.link(temporary_path, destination)
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

    optimizer.load_state_dict(optimizer_state)
    model.load_state_dict(model_state, strict=True)
    for layer in model.layers:
        layer.kmeans_initted = True
    restore_rng_state(rng_state)
    return {
        "start_epoch": completed_epoch + 1,
        "optimizer_step": optimizer_step,
    }
