import inspect
import os
import stat
import random
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import FrozenInstanceError
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from types import MethodType
from types import SimpleNamespace
from typing import Any
from typing import get_type_hints

import numpy as np
import pytest
import torch
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset

import rqvae_collapse_diagnostics as diagnostics
from modules.quantize import QuantizeForwardMode
from modules.rqvae import RqVae
from rqvae_collapse_diagnostics import (
    OptimizerSpec,
    build_diagnostic_optimizer,
    canonical_json,
    complete_config_hash,
    hash_json,
    initialization_config,
    initialization_config_hash,
    invariant_config,
    invariant_config_hash,
    optimizer_metadata,
    optimizer_metadata_hash,
    optimizer_treatment_hash,
    optimizer_treatment_payload,
)


def diagnostic_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "seed": 20260701,
        "epochs": 500,
        "dataset_sha256": "abc",
        "dataset_item_count": 12101,
        "batch_size": 1024,
        "vae_input_dim": 768,
        "vae_hidden_dims": [512, 256, 128],
        "vae_embed_dim": 32,
        "vae_codebook_size": 256,
        "vae_n_layers": 3,
        "commitment_weight": 0.25,
        "optimizer_name": "adagrad",
        "learning_rate": 0.4,
        "weight_decay": 0.0,
        "optimizer_eps": 1e-10,
        "initial_accumulator_value": 0.0,
        "adam_betas": None,
        "arm_label": "adagrad04",
        "save_dir_root": "/data/a",
        "config_path": "/configs/a.gin",
        "resume_checkpoint": None,
    }
    config.update(overrides)
    return config


def test_optimizer_specs_are_frozen_and_have_exact_factory_values() -> None:
    default = OptimizerSpec.adagrad_default()
    accumulator = OptimizerSpec.adagrad_accum01_eps1e7()
    adamw = OptimizerSpec.adamw_author()

    assert asdict(default) == {
        "name": "adagrad",
        "learning_rate": 0.4,
        "weight_decay": 0.0,
        "eps": 1e-10,
        "initial_accumulator_value": 0.0,
        "betas": None,
    }
    assert asdict(accumulator) == {
        "name": "adagrad",
        "learning_rate": 0.4,
        "weight_decay": 0.0,
        "eps": 1e-7,
        "initial_accumulator_value": 0.1,
        "betas": None,
    }
    assert asdict(adamw) == {
        "name": "adamw",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "eps": 1e-8,
        "initial_accumulator_value": None,
        "betas": (0.9, 0.999),
    }
    with pytest.raises(FrozenInstanceError):
        default.learning_rate = 0.1  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("learning_rate", 0),
        ("weight_decay", 0),
        ("eps", 0),
        ("initial_accumulator_value", 0),
        ("weight_decay", False),
        ("betas", [0.9, 0.999]),
        ("betas", (0.9, False)),
    ],
)
def test_raw_optimizer_specs_reject_non_exact_runtime_types(
    field: str,
    value: object,
) -> None:
    fields = asdict(OptimizerSpec.adagrad_default())
    if field == "betas":
        fields = asdict(OptimizerSpec.adamw_author())
    fields[field] = value
    spec = OptimizerSpec(**fields)

    with pytest.raises(ValueError, match=field):
        build_diagnostic_optimizer(nn.Linear(3, 2), spec)
    with pytest.raises(ValueError, match=field):
        optimizer_treatment_hash(spec)


@pytest.mark.parametrize(
    ("config_fields", "expected"),
    [
        (
            {
                "optimizer_name": "adagrad",
                "learning_rate": 0.4,
                "weight_decay": 0,
                "optimizer_eps": 1e-10,
                "initial_accumulator_value": 0,
                "adam_betas": None,
            },
            OptimizerSpec.adagrad_default(),
        ),
        (
            {
                "optimizer_name": "adagrad",
                "learning_rate": 0.4,
                "weight_decay": 0,
                "optimizer_eps": 1e-7,
                "initial_accumulator_value": 0.1,
                "adam_betas": None,
            },
            OptimizerSpec.adagrad_accum01_eps1e7(),
        ),
        (
            {
                "optimizer_name": "adamw",
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "optimizer_eps": 1e-8,
                "initial_accumulator_value": None,
                "adam_betas": [0.9, 0.999],
            },
            OptimizerSpec.adamw_author(),
        ),
    ],
)
def test_from_config_fields_normalizes_gin_values_to_canonical_specs(
    config_fields: dict[str, object],
    expected: OptimizerSpec,
) -> None:
    spec = OptimizerSpec.from_config_fields(**config_fields)

    assert spec == expected
    assert asdict(spec) == asdict(expected)
    assert optimizer_treatment_payload(spec) == optimizer_treatment_payload(expected)
    assert optimizer_treatment_hash(spec) == optimizer_treatment_hash(expected)


@pytest.mark.parametrize(
    "config_fields",
    [
        {
            "optimizer_name": "adagrad",
            "learning_rate": 0.4,
            "weight_decay": False,
            "optimizer_eps": 1e-10,
            "initial_accumulator_value": 0,
            "adam_betas": None,
        },
        {
            "optimizer_name": "adamw",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "optimizer_eps": 1e-8,
            "initial_accumulator_value": None,
            "adam_betas": "0.9,0.999",
        },
    ],
)
def test_from_config_fields_rejects_invalid_gin_types(
    config_fields: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="optimizer"):
        OptimizerSpec.from_config_fields(**config_fields)


@pytest.mark.parametrize(
    ("spec", "optimizer_class", "expected_defaults"),
    [
        (
            OptimizerSpec.adagrad_default(),
            torch.optim.Adagrad,
            {
                "lr": 0.4,
                "weight_decay": 0.0,
                "eps": 1e-10,
                "initial_accumulator_value": 0.0,
            },
        ),
        (
            OptimizerSpec.adagrad_accum01_eps1e7(),
            torch.optim.Adagrad,
            {
                "lr": 0.4,
                "weight_decay": 0.0,
                "eps": 1e-7,
                "initial_accumulator_value": 0.1,
            },
        ),
        (
            OptimizerSpec.adamw_author(),
            torch.optim.AdamW,
            {
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "eps": 1e-8,
                "betas": (0.9, 0.999),
            },
        ),
    ],
)
def test_supported_optimizer_treatments_have_exact_live_defaults(
    spec: OptimizerSpec,
    optimizer_class: type[torch.optim.Optimizer],
    expected_defaults: dict[str, object],
) -> None:
    optimizer = build_diagnostic_optimizer(nn.Linear(3, 2), spec)

    assert isinstance(optimizer, optimizer_class)
    for key, expected in expected_defaults.items():
        if isinstance(expected, float):
            assert optimizer.defaults[key] == pytest.approx(expected)
        else:
            assert optimizer.defaults[key] == expected


@pytest.mark.parametrize(
    "spec",
    [
        OptimizerSpec("adagrad", 0.3, 0.0, 1e-10, 0.0, None),
        OptimizerSpec("adagrad", 0.4, 0.0, 1e-10, 0.1, None),
        OptimizerSpec("adamw", 1e-3, 0.0, 1e-8, None, (0.9, 0.999)),
    ],
)
def test_optimizer_treatment_drift_is_rejected(spec: OptimizerSpec) -> None:
    with pytest.raises(ValueError, match="unsupported diagnostic optimizer"):
        build_diagnostic_optimizer(nn.Linear(3, 2), spec)


def test_optimizer_treatment_payloads_and_hashes_are_distinct() -> None:
    specs = [
        OptimizerSpec.adagrad_default(),
        OptimizerSpec.adagrad_accum01_eps1e7(),
        OptimizerSpec.adamw_author(),
    ]

    payloads = [optimizer_treatment_payload(spec) for spec in specs]
    hashes = [optimizer_treatment_hash(spec) for spec in specs]

    assert payloads == [asdict(spec) for spec in specs]
    assert hashes == [hash_json(payload) for payload in payloads]
    assert len(set(hashes)) == 3


def test_optimizer_metadata_records_live_defaults_and_has_stable_hash() -> None:
    adagrad = build_diagnostic_optimizer(
        nn.Linear(3, 2),
        OptimizerSpec.adagrad_default(),
    )
    adamw = build_diagnostic_optimizer(nn.Linear(3, 2), OptimizerSpec.adamw_author())

    adagrad_metadata = optimizer_metadata(adagrad)
    adamw_metadata = optimizer_metadata(adamw)

    assert adagrad_metadata["class"] == "Adagrad"
    assert adagrad_metadata["defaults"]["initial_accumulator_value"] == 0.0
    assert adagrad_metadata["param_groups"][0]["lr"] == pytest.approx(0.4)
    assert adamw_metadata["class"] == "AdamW"
    assert adamw_metadata["defaults"]["betas"] == (0.9, 0.999)
    assert adamw_metadata["param_groups"][0]["weight_decay"] == pytest.approx(1e-4)
    assert optimizer_metadata_hash(adagrad) == hash_json(adagrad_metadata)
    assert optimizer_metadata_hash(adamw) == hash_json(adamw_metadata)
    assert optimizer_metadata_hash(adagrad) != optimizer_metadata_hash(adamw)


def test_equivalent_live_optimizers_have_identical_metadata_hashes() -> None:
    first = build_diagnostic_optimizer(
        nn.Linear(3, 2),
        OptimizerSpec.adagrad_default(),
    )
    second = build_diagnostic_optimizer(
        nn.Linear(3, 2),
        OptimizerSpec.adagrad_default(),
    )

    assert optimizer_metadata(first) == optimizer_metadata(second)
    assert optimizer_metadata_hash(first) == optimizer_metadata_hash(second)


def test_canonical_json_is_sorted_compact_and_hashes_its_utf8_bytes() -> None:
    payload = {"z": 2, "a": (1, "x")}

    canonical = canonical_json(payload)

    assert canonical == '{"a":[1,"x"],"z":2}'
    assert hash_json(payload) == sha256(canonical.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_json_rejects_nonfinite_floats(value: float) -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        canonical_json({"value": value})


def test_config_field_collections_are_immutable() -> None:
    assert isinstance(diagnostics.INITIALIZATION_CONFIG_FIELDS, frozenset)
    assert isinstance(diagnostics._INVARIANT_EXCLUDED_FIELDS, frozenset)


def test_same_stage_arms_share_invariant_bytes_hash_and_initialization() -> None:
    base = diagnostic_config()
    other = diagnostic_config(
        optimizer_name="adamw",
        learning_rate=1e-3,
        weight_decay=1e-4,
        optimizer_eps=1e-8,
        initial_accumulator_value=None,
        adam_betas=[0.9, 0.999],
        arm_label="adamw",
        save_dir_root="/data/b",
        config_path="/configs/b.gin",
        resume_checkpoint="/data/b/checkpoint.pt",
    )

    base_invariant = invariant_config(base)
    other_invariant = invariant_config(other)

    assert complete_config_hash(base) != complete_config_hash(other)
    assert base_invariant == other_invariant
    assert canonical_json(base_invariant).encode("utf-8") == canonical_json(
        other_invariant
    ).encode("utf-8")
    assert invariant_config_hash(base) == invariant_config_hash(other)
    assert initialization_config(base) == initialization_config(other)
    assert initialization_config_hash(base) == initialization_config_hash(other)


def test_invariant_config_excludes_only_flattened_arm_fields() -> None:
    config = diagnostic_config(extra_research_field={"keep": True})

    result = invariant_config(config)

    excluded = {
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
    assert excluded.isdisjoint(result)
    assert result["epochs"] == 500
    assert result["extra_research_field"] == {"keep": True}


def test_initialization_config_contains_exact_required_fields() -> None:
    result = initialization_config(diagnostic_config(unrelated="ignored"))

    assert list(result) == [
        "batch_size",
        "commitment_weight",
        "dataset_item_count",
        "dataset_sha256",
        "seed",
        "vae_codebook_size",
        "vae_embed_dim",
        "vae_hidden_dims",
        "vae_input_dim",
        "vae_n_layers",
    ]


@pytest.mark.parametrize(
    "missing_field",
    [
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
    ],
)
def test_initialization_config_rejects_each_missing_field(
    missing_field: str,
) -> None:
    config = diagnostic_config()
    del config[missing_field]

    with pytest.raises(ValueError, match=missing_field):
        initialization_config(config)


def build_tiny_rqvae() -> RqVae:
    return RqVae(
        input_dim=8,
        embed_dim=4,
        hidden_dims=[6],
        codebook_size=256,
        codebook_kmeans_init=True,
        codebook_mode=QuantizeForwardMode.STE,
        n_layers=3,
        commitment_weight=0.25,
        n_cat_features=0,
    )


def test_tensor_and_nested_state_hashes_are_typed_and_deterministic() -> None:
    contiguous = torch.arange(12, dtype=torch.int64).reshape(3, 4)
    same_values_noncontiguous = contiguous.T.contiguous().T

    expected_tensor_digest = sha256()
    expected_tensor_digest.update(str(contiguous.dtype).encode("ascii"))
    expected_tensor_digest.update(str(tuple(contiguous.shape)).encode("ascii"))
    expected_tensor_digest.update(contiguous.numpy().tobytes())
    assert diagnostics.hash_tensor(contiguous) == expected_tensor_digest.hexdigest()
    assert diagnostics.hash_tensor(contiguous) == diagnostics.hash_tensor(
        same_values_noncontiguous
    )
    assert diagnostics.hash_tensor(contiguous) != diagnostics.hash_tensor(
        contiguous.reshape(2, 6)
    )
    assert diagnostics.hash_tensor(contiguous) != diagnostics.hash_tensor(
        contiguous.to(torch.float64)
    )

    first = {
        "z": [Path("artifact.pt"), np.arange(6, dtype=np.int16).reshape(2, 3)],
        "a": (None, True, 7, 1.25, "value", contiguous),
    }
    second = {"a": first["a"], "z": first["z"]}
    assert diagnostics.hash_nested_state(first) == diagnostics.hash_nested_state(
        second
    )
    assert diagnostics.hash_nested_state(first) != diagnostics.hash_nested_state(
        {"a": list(first["a"]), "z": first["z"]}
    )


def test_seed_capture_restore_and_rng_hash_cover_exact_rng_schema() -> None:
    diagnostics.seed_all(20260701)
    state = diagnostics.capture_rng_state()

    assert set(state) == {"python", "numpy", "torch_cpu", "torch_cuda"}
    assert isinstance(state["numpy"], tuple)
    assert isinstance(state["torch_cpu"], torch.Tensor)
    assert isinstance(state["torch_cuda"], list)
    assert all(isinstance(value, torch.Tensor) for value in state["torch_cuda"])

    expected = (random.random(), float(np.random.random()), torch.rand(4))
    diagnostics.restore_rng_state(state)
    observed = (random.random(), float(np.random.random()), torch.rand(4))

    assert observed[:2] == expected[:2]
    assert torch.equal(observed[2], expected[2])
    assert diagnostics.hash_rng_state(state) == diagnostics.hash_nested_state(state)


def test_load_items_by_index_preserves_requested_order() -> None:
    item_tensor = torch.arange(80, dtype=torch.float32).reshape(10, 8)
    dataset = TensorDataset(item_tensor)
    indices = torch.tensor([7, 1, 9, 1], dtype=torch.long)

    assert torch.equal(
        diagnostics.load_items_by_index(dataset, indices),
        item_tensor[indices],
    )


def test_clone_and_hash_state_dict_are_cpu_snapshots() -> None:
    model = nn.Linear(3, 2)
    snapshot = diagnostics.clone_cpu_state_dict(model)
    snapshot_hash = diagnostics.hash_state_dict(snapshot)

    assert snapshot_hash == diagnostics.hash_nested_state(
        dict(sorted(snapshot.items()))
    )
    assert all(value.device.type == "cpu" for value in snapshot.values())
    with torch.no_grad():
        model.weight.add_(1)
    assert snapshot_hash == diagnostics.hash_state_dict(snapshot)
    assert snapshot_hash != diagnostics.hash_state_dict(
        diagnostics.clone_cpu_state_dict(model)
    )


def test_state_dict_hash_frames_keys_against_structural_collisions() -> None:
    first_tensor = torch.tensor([1])
    second_tensor = torch.tensor([2])
    first_state = {"a": first_tensor, "b": second_tensor}
    colliding_unframed_key = (
        "a" + diagnostics.hash_tensor(first_tensor) + "b"
    )
    structurally_different_state = {
        colliding_unframed_key: second_tensor,
    }

    assert diagnostics.hash_state_dict(first_state) != diagnostics.hash_state_dict(
        structurally_different_state
    )


def test_epoch_permutations_and_500_epoch_plan_have_fixed_hashes() -> None:
    epoch_one = diagnostics.epoch_permutation(12101, 20260701, 1)
    epoch_two = diagnostics.epoch_permutation(12101, 20260701, 2)
    reference = torch.randperm(
        12101,
        generator=torch.Generator().manual_seed(20260701),
    )

    assert torch.equal(epoch_one, reference)
    assert torch.equal(
        diagnostics.epoch_permutation(12101, 20260701, 1),
        reference,
    )
    assert not torch.equal(epoch_one, epoch_two)
    assert epoch_one[:10].tolist() == [
        5789,
        7976,
        1954,
        61,
        4956,
        6279,
        8367,
        9758,
        10931,
        11451,
    ]
    assert diagnostics.hash_tensor(epoch_one) == (
        "8907d569cce451f348f62fac9bc20acab7197642b88d65afc80c91c0e031eb5d"
    )
    assert diagnostics.hash_tensor(epoch_two) == (
        "80221b13579551f7563d8c84ab9680bca94be2b592419988f8096f43b559d50b"
    )

    epoch_hashes = diagnostics.epoch_plan_hashes(12101, 20260701, 500)
    assert len(epoch_hashes) == 500
    assert epoch_hashes[:2] == [
        "8907d569cce451f348f62fac9bc20acab7197642b88d65afc80c91c0e031eb5d",
        "80221b13579551f7563d8c84ab9680bca94be2b592419988f8096f43b559d50b",
    ]
    assert diagnostics.rolling_epoch_plan_hash(12101, 20260701, 500) == (
        "658100b51fb7833b8574188dec3c2901bf0fc7387ae4b58401a6f7202a70a9ac"
    )


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: diagnostics.epoch_permutation(0, 7, 1), "item_count"),
        (lambda: diagnostics.epoch_permutation(10, 7, 0), "epoch"),
        (lambda: diagnostics.epoch_plan_hashes(10, 7, 0), "epochs"),
        (lambda: diagnostics.rolling_epoch_plan_hash(10, 7, 0), "epochs"),
    ],
)
def test_epoch_plan_rejects_nonpositive_dimensions(call, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()


def test_common_initialization_public_signatures_are_exact() -> None:
    assert get_type_hints(diagnostics.load_items_by_index) == {
        "dataset": Dataset,
        "indices": Tensor,
        "return": Tensor,
    }
    assert get_type_hints(diagnostics.capture_rng_state) == {
        "return": diagnostics.RngState,
    }
    assert get_type_hints(diagnostics.restore_rng_state) == {
        "state": diagnostics.RngState,
        "return": type(None),
    }
    assert get_type_hints(diagnostics.hash_rng_state) == {
        "state": diagnostics.RngState,
        "return": str,
    }
    assert get_type_hints(diagnostics.save_common_initialization) == {
        "path": Path,
        "initial_model_state": Mapping[str, Tensor],
        "post_kmeans_model_state": Mapping[str, Tensor],
        "rng_before_model_initialization": diagnostics.RngState,
        "rng_at_training_start": diagnostics.RngState,
        "initial_batch_indices": Tensor,
        "initial_batch_inputs": Tensor,
        "initial_batch_used_code_counts": Sequence[int],
        "dataset_sha256": str,
        "dataset_item_count": int,
        "seed": int,
        "initialization_config_payload": Mapping[str, Any],
        "epoch_hashes": Sequence[str],
        "epoch_plan_rolling_hash": str,
        "return": dict[str, Any],
    }
    assert get_type_hints(diagnostics.load_common_initialization) == {
        "path": Path,
        "model": nn.Module,
        "expected_dataset_sha256": str,
        "expected_dataset_item_count": int,
        "expected_seed": int,
        "expected_initialization_config_hash": str,
        "expected_epoch_plan_rolling_hash": str,
        "return": dict[str, Any],
    }
    assert list(
        inspect.signature(diagnostics.save_common_initialization).parameters
    ) == [
        "path",
        "initial_model_state",
        "post_kmeans_model_state",
        "rng_before_model_initialization",
        "rng_at_training_start",
        "initial_batch_indices",
        "initial_batch_inputs",
        "initial_batch_used_code_counts",
        "dataset_sha256",
        "dataset_item_count",
        "seed",
        "initialization_config_payload",
        "epoch_hashes",
        "epoch_plan_rolling_hash",
    ]


def test_common_initialization_round_trip_validation_and_exclusive_publish(
    tmp_path: Path,
) -> None:
    seed = 20260701
    dataset_sha256 = "a" * 64
    dataset_item_count = 1024
    initialization_payload = initialization_config(
        diagnostic_config(
            dataset_sha256=dataset_sha256,
            dataset_item_count=dataset_item_count,
            vae_input_dim=8,
            vae_hidden_dims=[6],
            vae_embed_dim=4,
        )
    )

    diagnostics.seed_all(seed)
    rng_before_model_initialization = diagnostics.capture_rng_state()
    model = build_tiny_rqvae()
    initial_model_state = diagnostics.clone_cpu_state_dict(model)
    initial_batch_indices = diagnostics.epoch_permutation(
        dataset_item_count,
        seed,
        1,
    )[:1024]
    initial_batch_inputs = (
        torch.arange(1024 * 8, dtype=torch.float32).reshape(1024, 8) / 1024
    )
    semantic_ids = model.get_semantic_ids(initial_batch_inputs).sem_ids
    assert all(layer.kmeans_initted for layer in model.layers)
    initial_batch_used_code_counts = [
        int(semantic_ids[:, index].unique().numel())
        for index in range(semantic_ids.shape[1])
    ]
    post_kmeans_model_state = diagnostics.clone_cpu_state_dict(model)
    rng_at_training_start = diagnostics.capture_rng_state()
    epoch_hashes = diagnostics.epoch_plan_hashes(dataset_item_count, seed, 500)
    epoch_plan_rolling_hash = diagnostics.rolling_epoch_plan_hash(
        dataset_item_count,
        seed,
        500,
    )
    path = tmp_path / "rqvae_common_initialization.pt"
    save_kwargs = {
        "initial_model_state": initial_model_state,
        "post_kmeans_model_state": post_kmeans_model_state,
        "rng_before_model_initialization": rng_before_model_initialization,
        "rng_at_training_start": rng_at_training_start,
        "initial_batch_indices": initial_batch_indices,
        "initial_batch_inputs": initial_batch_inputs,
        "initial_batch_used_code_counts": initial_batch_used_code_counts,
        "dataset_sha256": dataset_sha256,
        "dataset_item_count": dataset_item_count,
        "seed": seed,
        "initialization_config_payload": initialization_payload,
        "epoch_hashes": epoch_hashes,
        "epoch_plan_rolling_hash": epoch_plan_rolling_hash,
    }

    artifact = diagnostics.save_common_initialization(path, **save_kwargs)

    assert path.is_file()
    assert set(artifact) == {
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
    assert artifact["schema_version"] == 1
    assert artifact["artifact_kind"] == "rqvae_common_initialization"
    assert artifact["dataset"] == {
        "name": "amazon-beauty",
        "item_count": dataset_item_count,
        "sha256": dataset_sha256,
    }
    assert artifact["initialization_config"] == initialization_payload
    assert artifact["initialization_config_hash"] == hash_json(
        initialization_payload
    )
    assert artifact["initial_model_hash"] == diagnostics.hash_state_dict(
        initial_model_state
    )
    assert artifact["post_kmeans_model_hash"] == diagnostics.hash_state_dict(
        post_kmeans_model_state
    )
    expected_codebook_hash = hash_json(
        [
            diagnostics.hash_tensor(value)
            for key, value in sorted(post_kmeans_model_state.items())
            if key.startswith("layers.") and key.endswith(".embedding.weight")
        ]
    )
    assert artifact["post_kmeans_codebook_hash"] == expected_codebook_hash
    assert artifact["rng_before_model_initialization_hash"] == (
        diagnostics.hash_rng_state(rng_before_model_initialization)
    )
    assert artifact["rng_at_training_start_hash"] == diagnostics.hash_rng_state(
        rng_at_training_start
    )
    assert set(artifact["initial_batch"]) == {
        "indices",
        "indices_hash",
        "input_tensor_hash",
        "used_code_counts",
    }
    assert torch.equal(
        artifact["initial_batch"]["indices"],
        initial_batch_indices,
    )
    assert artifact["initial_batch"]["indices_hash"] == diagnostics.hash_tensor(
        initial_batch_indices
    )
    assert artifact["initial_batch"]["input_tensor_hash"] == diagnostics.hash_tensor(
        initial_batch_inputs
    )
    assert artifact["initial_batch"]["used_code_counts"] == (
        initial_batch_used_code_counts
    )
    assert artifact["epoch_plan_hashes"] == epoch_hashes
    assert len(artifact["epoch_plan_hashes"]) == 500
    assert artifact["epoch_plan_rolling_hash"] == epoch_plan_rolling_hash

    expected_compatibility = {
        "dataset_sha256": dataset_sha256,
        "dataset_item_count": dataset_item_count,
        "seed": seed,
        "initialization_config_hash": artifact["initialization_config_hash"],
        "initial_model_hash": artifact["initial_model_hash"],
        "post_kmeans_model_hash": artifact["post_kmeans_model_hash"],
        "post_kmeans_codebook_hash": artifact["post_kmeans_codebook_hash"],
        "initial_batch_indices_hash": artifact["initial_batch"]["indices_hash"],
        "initial_batch_input_hash": artifact["initial_batch"]["input_tensor_hash"],
        "rng_at_training_start_hash": artifact["rng_at_training_start_hash"],
        "epoch_plan_rolling_hash": epoch_plan_rolling_hash,
    }
    assert artifact["compatibility"] == expected_compatibility
    assert artifact["common_initialization_hash"] == hash_json(
        expected_compatibility
    )

    normalized_sequence_path = tmp_path / "normalized_sequences.pt"
    normalized_sequence_artifact = diagnostics.save_common_initialization(
        normalized_sequence_path,
        **{
            **save_kwargs,
            "initial_batch_used_code_counts": tuple(
                initial_batch_used_code_counts
            ),
            "epoch_hashes": tuple(epoch_hashes),
        },
    )
    assert type(
        normalized_sequence_artifact["initial_batch"]["used_code_counts"]
    ) is list
    assert type(normalized_sequence_artifact["epoch_plan_hashes"]) is list
    normalized_sequence_path.unlink()

    invalid_config_payloads = [
        [],
        {
            key: value
            for key, value in initialization_payload.items()
            if key != "seed"
        },
        {**initialization_payload, "unexpected": True},
        {**initialization_payload, "dataset_sha256": "b" * 64},
        {**initialization_payload, "dataset_item_count": 2048},
        {**initialization_payload, "seed": seed + 1},
    ]
    for index, invalid_payload in enumerate(invalid_config_payloads):
        with pytest.raises(
            (TypeError, ValueError),
            match="initialization config",
        ):
            diagnostics.save_common_initialization(
                tmp_path / f"invalid_config_{index}.pt",
                **{
                    **save_kwargs,
                    "initialization_config_payload": invalid_payload,
                },
            )

    invalid_batches = [
        {
            "initial_batch_indices": initial_batch_indices.tolist(),
        },
        {
            "initial_batch_indices": initial_batch_indices.to(torch.int32),
        },
        {
            "initial_batch_indices": initial_batch_indices[:1023],
            "initial_batch_inputs": initial_batch_inputs[:1023],
        },
        {
            "initial_batch_indices": initial_batch_indices.roll(1),
        },
        {
            "initial_batch_inputs": initial_batch_inputs[:1023],
        },
    ]
    for index, overrides in enumerate(invalid_batches):
        with pytest.raises(
            (TypeError, ValueError),
            match="initial batch",
        ):
            diagnostics.save_common_initialization(
                tmp_path / f"invalid_batch_{index}.pt",
                **{**save_kwargs, **overrides},
            )

    forged_epoch_hashes = list(epoch_hashes)
    forged_epoch_hashes[0] = "0" * 64
    forged_rolling_hash = hash_json(forged_epoch_hashes)
    forged_save_kwargs = {
        **save_kwargs,
        "epoch_hashes": forged_epoch_hashes,
        "epoch_plan_rolling_hash": forged_rolling_hash,
    }
    with pytest.raises(ValueError, match="deterministic epoch plan"):
        diagnostics.save_common_initialization(
            tmp_path / "forged_save.pt",
            **forged_save_kwargs,
        )

    published_bytes = path.read_bytes()
    with pytest.raises(FileExistsError):
        diagnostics.save_common_initialization(path, **save_kwargs)
    assert path.read_bytes() == published_bytes
    assert list(tmp_path.iterdir()) == [path]

    diagnostics.restore_rng_state(rng_before_model_initialization)
    fresh_model = build_tiny_rqvae()
    assert diagnostics.hash_state_dict(
        diagnostics.clone_cpu_state_dict(fresh_model)
    ) == artifact["initial_model_hash"]
    random.random()
    np.random.random()
    torch.rand(3)

    loaded = diagnostics.load_common_initialization(
        path,
        model=fresh_model,
        expected_dataset_sha256=dataset_sha256,
        expected_dataset_item_count=dataset_item_count,
        expected_seed=seed,
        expected_initialization_config_hash=artifact[
            "initialization_config_hash"
        ],
        expected_epoch_plan_rolling_hash=epoch_plan_rolling_hash,
    )

    assert loaded["common_initialization_hash"] == artifact[
        "common_initialization_hash"
    ]
    assert diagnostics.hash_state_dict(
        diagnostics.clone_cpu_state_dict(fresh_model)
    ) == artifact["post_kmeans_model_hash"]
    assert all(layer.kmeans_initted for layer in fresh_model.layers)
    assert diagnostics.hash_rng_state(diagnostics.capture_rng_state()) == artifact[
        "rng_at_training_start_hash"
    ]

    def assert_stored_artifact_rejected(
        candidate: dict[str, Any],
        *,
        filename: str,
        error_type: type[Exception],
        match: str,
        expected_config_hash: str,
    ) -> None:
        candidate_path = tmp_path / filename
        torch.save(candidate, candidate_path)
        diagnostics.restore_rng_state(rng_before_model_initialization)
        candidate_model = build_tiny_rqvae()
        candidate_model_hash = diagnostics.hash_state_dict(
            diagnostics.clone_cpu_state_dict(candidate_model)
        )
        candidate_rng_hash = diagnostics.hash_rng_state(
            diagnostics.capture_rng_state()
        )
        with pytest.raises(error_type, match=match):
            diagnostics.load_common_initialization(
                candidate_path,
                model=candidate_model,
                expected_dataset_sha256=dataset_sha256,
                expected_dataset_item_count=dataset_item_count,
                expected_seed=seed,
                expected_initialization_config_hash=expected_config_hash,
                expected_epoch_plan_rolling_hash=epoch_plan_rolling_hash,
            )
        assert diagnostics.hash_state_dict(
            diagnostics.clone_cpu_state_dict(candidate_model)
        ) == candidate_model_hash
        assert diagnostics.hash_rng_state(
            diagnostics.capture_rng_state()
        ) == candidate_rng_hash

    tuple_epoch_artifact = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    tuple_epoch_artifact["epoch_plan_hashes"] = tuple(epoch_hashes)
    assert_stored_artifact_rejected(
        tuple_epoch_artifact,
        filename="tuple_epoch_hashes.pt",
        error_type=TypeError,
        match="epoch_plan_hashes.*list",
        expected_config_hash=artifact["initialization_config_hash"],
    )

    tensor_usage_artifact = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    tensor_usage_artifact["initial_batch"]["used_code_counts"] = torch.tensor(
        initial_batch_used_code_counts
    )
    assert_stored_artifact_rejected(
        tensor_usage_artifact,
        filename="tensor_usage.pt",
        error_type=TypeError,
        match="used_code_counts.*list",
        expected_config_hash=artifact["initialization_config_hash"],
    )

    int32_indices_artifact = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    int32_indices = initial_batch_indices.to(torch.int32)
    int32_indices_hash = diagnostics.hash_tensor(int32_indices)
    int32_indices_artifact["initial_batch"]["indices"] = int32_indices
    int32_indices_artifact["initial_batch"]["indices_hash"] = int32_indices_hash
    int32_indices_artifact["compatibility"][
        "initial_batch_indices_hash"
    ] = int32_indices_hash
    int32_indices_artifact["common_initialization_hash"] = hash_json(
        int32_indices_artifact["compatibility"]
    )
    assert_stored_artifact_rejected(
        int32_indices_artifact,
        filename="int32_indices.pt",
        error_type=ValueError,
        match="initial batch.*long",
        expected_config_hash=artifact["initialization_config_hash"],
    )

    wrong_indices_artifact = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    wrong_indices = initial_batch_indices.roll(1)
    wrong_indices_hash = diagnostics.hash_tensor(wrong_indices)
    wrong_indices_artifact["initial_batch"]["indices"] = wrong_indices
    wrong_indices_artifact["initial_batch"]["indices_hash"] = wrong_indices_hash
    wrong_indices_artifact["compatibility"][
        "initial_batch_indices_hash"
    ] = wrong_indices_hash
    wrong_indices_artifact["common_initialization_hash"] = hash_json(
        wrong_indices_artifact["compatibility"]
    )
    assert_stored_artifact_rejected(
        wrong_indices_artifact,
        filename="wrong_indices.pt",
        error_type=ValueError,
        match="initial batch.*epoch permutation",
        expected_config_hash=artifact["initialization_config_hash"],
    )

    mismatched_config_artifact = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    mismatched_config_artifact["initialization_config"]["seed"] = seed + 1
    mismatched_config_hash = hash_json(
        mismatched_config_artifact["initialization_config"]
    )
    mismatched_config_artifact[
        "initialization_config_hash"
    ] = mismatched_config_hash
    mismatched_config_artifact["compatibility"][
        "initialization_config_hash"
    ] = mismatched_config_hash
    mismatched_config_artifact["common_initialization_hash"] = hash_json(
        mismatched_config_artifact["compatibility"]
    )
    assert_stored_artifact_rejected(
        mismatched_config_artifact,
        filename="mismatched_config.pt",
        error_type=ValueError,
        match="initialization config.*seed",
        expected_config_hash=mismatched_config_hash,
    )

    forged = torch.load(path, map_location="cpu", weights_only=False)
    forged["epoch_plan_hashes"] = forged_epoch_hashes
    forged["epoch_plan_rolling_hash"] = forged_rolling_hash
    forged["compatibility"]["epoch_plan_rolling_hash"] = forged_rolling_hash
    forged["common_initialization_hash"] = hash_json(forged["compatibility"])
    forged_path = tmp_path / "forged_load.pt"
    torch.save(forged, forged_path)
    diagnostics.restore_rng_state(rng_before_model_initialization)
    forged_model = build_tiny_rqvae()
    forged_model_hash = diagnostics.hash_state_dict(
        diagnostics.clone_cpu_state_dict(forged_model)
    )
    forged_rng_hash = diagnostics.hash_rng_state(
        diagnostics.capture_rng_state()
    )
    with pytest.raises(ValueError, match="deterministic epoch plan"):
        diagnostics.load_common_initialization(
            forged_path,
            model=forged_model,
            expected_dataset_sha256=dataset_sha256,
            expected_dataset_item_count=dataset_item_count,
            expected_seed=seed,
            expected_initialization_config_hash=artifact[
                "initialization_config_hash"
            ],
            expected_epoch_plan_rolling_hash=forged_rolling_hash,
        )
    assert diagnostics.hash_state_dict(
        diagnostics.clone_cpu_state_dict(forged_model)
    ) == forged_model_hash
    assert diagnostics.hash_rng_state(
        diagnostics.capture_rng_state()
    ) == forged_rng_hash

    tampered = torch.load(path, map_location="cpu", weights_only=False)
    first_state_key = next(iter(tampered["post_kmeans_model_state"]))
    tampered["post_kmeans_model_state"][first_state_key].add_(1)
    tampered_path = tmp_path / "tampered.pt"
    torch.save(tampered, tampered_path)

    diagnostics.restore_rng_state(rng_before_model_initialization)
    untouched_model = build_tiny_rqvae()
    untouched_model_hash = diagnostics.hash_state_dict(
        diagnostics.clone_cpu_state_dict(untouched_model)
    )
    untouched_rng_hash = diagnostics.hash_rng_state(
        diagnostics.capture_rng_state()
    )
    with pytest.raises(ValueError, match="post-kmeans model hash"):
        diagnostics.load_common_initialization(
            tampered_path,
            model=untouched_model,
            expected_dataset_sha256=dataset_sha256,
            expected_dataset_item_count=dataset_item_count,
            expected_seed=seed,
            expected_initialization_config_hash=artifact[
                "initialization_config_hash"
            ],
            expected_epoch_plan_rolling_hash=epoch_plan_rolling_hash,
        )
    assert diagnostics.hash_state_dict(
        diagnostics.clone_cpu_state_dict(untouched_model)
    ) == untouched_model_hash
    assert diagnostics.hash_rng_state(
        diagnostics.capture_rng_state()
    ) == untouched_rng_hash



def _tiny_diagnostic_rqvae() -> RqVae:
    model = RqVae(
        input_dim=4, embed_dim=2, hidden_dims=[3], codebook_size=256,
        codebook_kmeans_init=False, codebook_normalize=False,
        codebook_sim_vq=False, codebook_mode=QuantizeForwardMode.STE,
        n_layers=3, commitment_weight=0.25, n_cat_features=0,
    )
    for layer in model.layers:
        layer.kmeans_initted = True
    return model


def test_distribution_stats_use_population_variance_and_fixed_quantiles() -> None:
    inputs = torch.tensor([[0.0, 0.0], [2.0, 4.0]])
    assert diagnostics.input_distribution_stats(inputs) == pytest.approx({
        "global_variance": 2.75, "mean_feature_variance": 2.5,
        "norm_mean": (20.0**0.5) / 2, "norm_std": (20.0**0.5) / 2,
        "norm_min": 0.0, "norm_max": 20.0**0.5,
    })
    encoded = torch.tensor([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
    stats = diagnostics.encoder_distribution_stats(encoded)
    assert stats["global_variance"] == pytest.approx(encoded.var(unbiased=False).item())
    assert [stats[f"norm_p{x}"] for x in ("00", "50", "95", "99", "100")] == pytest.approx([0, 5, 9.5, 9.9, 10])
    residual = diagnostics.residual_distribution_stats([encoded])[0]
    assert residual == pytest.approx({
        "variance": encoded.var(unbiased=False).item(),
        "norm_mean": 5.0, "norm_p50": 5.0, "norm_p95": 9.5,
    })


def test_assignment_diagnostics_cover_collapse_balance_nearest_rank_and_overflow() -> None:
    collapsed = diagnostics.assignment_diagnostics(torch.zeros((100, 3), dtype=torch.long))
    assert collapsed["used_counts"] == [1, 1, 1]
    assert [x["normalized_entropy"] for x in collapsed["layers"]] == [0, 0, 0]
    assert [x["top1_mass"] for x in collapsed["layers"]] == [1, 1, 1]
    assert (
        collapsed["unique_three_token_count"],
        collapsed["unique_four_token_count"],
        collapsed["max_bucket_size"],
        collapsed["bucket_size_p50"],
        collapsed["bucket_size_p95"],
        collapsed["bucket_size_p99"],
        collapsed["collision_gate_passed"],
        collapsed["hard_collapse"],
    ) == (1, 100, 100, 100, 100, 100, True, True)

    codes = torch.arange(256)
    balanced = diagnostics.assignment_diagnostics(torch.stack([codes] * 3, dim=1))
    assert balanced["used_counts"] == [256, 256, 256]
    assert [x["normalized_entropy"] for x in balanced["layers"]] == pytest.approx([1] * 3)
    assert [x["top1_mass"] for x in balanced["layers"]] == pytest.approx([1 / 256] * 3)
    assert (balanced["unique_three_token_count"], balanced["max_bucket_size"], balanced["hard_collapse"]) == (256, 1, False)
    assert balanced["paper_gate_passed"] is False

    rows = [[code] * 3 for code, size in enumerate([1, 2, 3, 4, 20]) for _ in range(size)]
    ranked = diagnostics.assignment_diagnostics(torch.tensor(rows))
    assert [ranked[f"bucket_size_p{x}"] for x in (50, 95, 99)] == [3, 20, 20]

    overflow = diagnostics.assignment_diagnostics(torch.zeros((257, 3), dtype=torch.long))
    assert overflow["unique_four_token_count"] == 0
    assert overflow["collision_gate_passed"] is False
    assert overflow["fixed_collision_cardinality"] == 256


@pytest.mark.parametrize(("epochs", "status", "expected"), [
    ({100: False, 250: False, 500: False}, "complete", "never"),
    ({25: True, 100: False, 250: False, 500: False}, "complete", "transient_recovered"),
    ({100: False, 250: False, 500: True}, "complete", "final_only"),
    ({100: True, 250: True, 500: True}, "complete", "sustained"),
    ({100: False, 250: False}, "complete", "not_evaluable"),
    ({100: True, 250: True, 500: True}, "deadline_stop", "not_evaluable"),
])
def test_classify_collapse_fixed_semantics(epochs: dict[int, bool], status: str, expected: str) -> None:
    value = diagnostics.classify_collapse(epochs, failure_status=diagnostics.FailureStatus(status))
    assert value["collapse_class"] == expected
    if expected == "not_evaluable":
        assert value["final_hard_collapse"] is None
        assert value["sustained_hard_collapse"] is None


def test_triggers_are_step_first() -> None:
    assert diagnostics.scheduled_snapshot_triggers(
        optimizer_step=12, completed_epoch=1,
        step_schedule=frozenset({0, 12}), epoch_schedule=frozenset({1, 2}),
    ) == ("optimizer_step:12", "epoch:1")


def test_group_gradient_and_delta_diagnostics_only_measure_expected_update() -> None:
    model = _tiny_diagnostic_rqvae()
    groups = diagnostics.parameter_groups(model)
    assert list(groups) == ["encoder", "decoder"] + [f"layers.{i}.embedding.weight" for i in range(3)]
    names = [name for group in groups.values() for name, _ in group]
    assert sorted(names) == sorted(name for name, p in model.named_parameters() if p.requires_grad)
    assert all(all(value is None for value in stats.values()) for stats in diagnostics.gradient_group_stats(groups).values())

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model_before = diagnostics.hash_state_dict(diagnostics.clone_cpu_state_dict(model))
    optimizer_before = diagnostics.hash_nested_state(optimizer.state_dict())
    sum(p.square().sum() for p in model.parameters()).backward()
    gradients = diagnostics.gradient_group_stats(groups)
    assert all(x["nonfinite_fraction"] == 0 and x["l2_norm"] is not None for x in gradients.values())
    assert diagnostics.hash_state_dict(diagnostics.clone_cpu_state_dict(model)) == model_before
    assert diagnostics.hash_nested_state(optimizer.state_dict()) == optimizer_before

    before = diagnostics.clone_group_parameters(groups)
    optimizer.step()
    model_after = diagnostics.hash_state_dict(diagnostics.clone_cpu_state_dict(model))
    optimizer_after = diagnostics.hash_nested_state(optimizer.state_dict())
    deltas = diagnostics.parameter_delta_stats(groups, before)
    assert all(x["delta_l2"] > 0 and x["relative_update"] > 0 for x in deltas.values())
    assert diagnostics.hash_state_dict(diagnostics.clone_cpu_state_dict(model)) == model_after != model_before
    assert diagnostics.hash_nested_state(optimizer.state_dict()) == optimizer_after != optimizer_before

    for _, parameter in groups["encoder"]:
        parameter.grad = torch.zeros_like(parameter)
    groups["encoder"][0][1].grad.reshape(-1)[0] = float("inf")
    nonfinite = diagnostics.gradient_group_stats(groups)["encoder"]
    assert nonfinite["l2_norm"] is None and nonfinite["max_abs"] is None
    assert nonfinite["nonfinite_fraction"] > 0


def test_parameter_groups_reject_unassigned_trainable_parameters() -> None:
    model = _tiny_diagnostic_rqvae()
    model.extra = nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError, match="unassigned trainable"):
        diagnostics.parameter_groups(model)


def test_codebook_stats_compare_baseline_without_mutation() -> None:
    model = _tiny_diagnostic_rqvae()
    baseline = diagnostics.clone_cpu_state_dict(model)
    baseline_hash = diagnostics.hash_state_dict(baseline)
    with torch.no_grad():
        model.layers[1].embedding.weight.add_(0.5)
    model_hash = diagnostics.hash_state_dict(diagnostics.clone_cpu_state_dict(model))
    stats = diagnostics.codebook_weight_stats(model, baseline)
    assert list(stats) == [f"layers.{i}.embedding.weight" for i in range(3)]
    assert stats["layers.0.embedding.weight"]["movement_l2"] == 0
    assert stats["layers.1.embedding.weight"]["movement_l2"] > 0
    assert diagnostics.hash_state_dict(baseline) == baseline_hash
    assert diagnostics.hash_state_dict(diagnostics.clone_cpu_state_dict(model)) == model_hash


def test_read_only_snapshot_restores_model_optimizer_rng_loader_and_epoch_plan() -> None:
    diagnostics.seed_all(20260701)
    model = _tiny_diagnostic_rqvae()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 31
    generator = torch.Generator().manual_seed(1234)
    loader = DataLoader(TensorDataset(inputs), batch_size=3, shuffle=False, num_workers=0, generator=generator)
    baseline = diagnostics.clone_cpu_state_dict(model)
    before = (
        diagnostics.hash_state_dict(baseline),
        diagnostics.hash_nested_state(optimizer.state_dict()),
        diagnostics.hash_rng_state(diagnostics.capture_rng_state()),
        diagnostics.hash_tensor(generator.get_state()),
        diagnostics.rolling_epoch_plan_hash(8, 20260701, 5),
    )
    snapshot = diagnostics.capture_read_only_corpus_snapshot(
        model=model, optimizer=optimizer, canonical_loader=loader,
        diagnostic_loader_generator=generator, post_kmeans_state=baseline,
        optimizer_step=0, completed_epoch=None, triggers=("optimizer_step:0",),
    )
    assert snapshot["optimizer_step"] == 0 and snapshot["completed_epoch"] is None
    assert snapshot["triggers"] == ["optimizer_step:0"] and snapshot["item_count"] == 8
    assert set(snapshot) >= {"input_distribution", "encoder_distribution", "residual_distributions", "assignment_diagnostics", "codebook_weight_stats"}
    after = (
        diagnostics.hash_state_dict(diagnostics.clone_cpu_state_dict(model)),
        diagnostics.hash_nested_state(optimizer.state_dict()),
        diagnostics.hash_rng_state(diagnostics.capture_rng_state()),
        diagnostics.hash_tensor(generator.get_state()),
        diagnostics.rolling_epoch_plan_hash(8, 20260701, 5),
    )
    assert model.training is True
    assert after == before



def _snapshot_fixture() -> tuple[
    RqVae,
    torch.optim.Optimizer,
    DataLoader,
    torch.Generator,
    dict[str, Tensor],
]:
    diagnostics.seed_all(20260701)
    model = _tiny_diagnostic_rqvae()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(1234)
    loader = DataLoader(
        TensorDataset(torch.arange(32, dtype=torch.float32).reshape(8, 4) / 31),
        batch_size=3,
        shuffle=False,
        num_workers=0,
        generator=generator,
    )
    return (
        model,
        optimizer,
        loader,
        generator,
        diagnostics.clone_cpu_state_dict(model),
    )


def _snapshot_observable_state(
    model: RqVae,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
) -> tuple[str, str, str, str, bool, tuple[bool, ...]]:
    return (
        diagnostics.hash_state_dict(diagnostics.clone_cpu_state_dict(model)),
        diagnostics.hash_nested_state(optimizer.state_dict()),
        diagnostics.hash_rng_state(diagnostics.capture_rng_state()),
        diagnostics.hash_tensor(generator.get_state()),
        model.training,
        tuple(layer.kmeans_initted for layer in model.layers),
    )


def _install_mutating_semantic_ids(
    model: RqVae,
    optimizer: torch.optim.Optimizer,
    *,
    raise_after_mutation: bool,
) -> None:
    original = model.get_semantic_ids

    def mutating_get_semantic_ids(
        self: RqVae,
        inputs: Tensor,
        gumbel_t: float = 0.001,
    ) -> object:
        with torch.no_grad():
            next(self.parameters()).add_(1.0)
        first_parameter = next(self.parameters())
        optimizer.param_groups[0]["lr"] = 0.123
        optimizer.state[first_parameter]["diagnostic_pollution"] = torch.ones(1)
        random.random()
        np.random.random()
        torch.rand(1)
        self.layers[0].kmeans_initted = False
        if raise_after_mutation:
            raise RuntimeError("synthetic forward failure")
        return original(inputs, gumbel_t=gumbel_t)

    model.get_semantic_ids = MethodType(mutating_get_semantic_ids, model)


def test_snapshot_restores_all_state_before_reraising_forward_exception() -> None:
    model, optimizer, loader, generator, baseline = _snapshot_fixture()
    before = _snapshot_observable_state(model, optimizer, generator)
    _install_mutating_semantic_ids(
        model,
        optimizer,
        raise_after_mutation=True,
    )

    with pytest.raises(RuntimeError, match="synthetic forward failure"):
        diagnostics.capture_read_only_corpus_snapshot(
            model=model,
            optimizer=optimizer,
            canonical_loader=loader,
            diagnostic_loader_generator=generator,
            post_kmeans_state=baseline,
            optimizer_step=0,
            completed_epoch=None,
            triggers=("optimizer_step:0",),
        )

    assert _snapshot_observable_state(model, optimizer, generator) == before


def test_snapshot_restores_detected_mutation_before_raising_invariant() -> None:
    model, optimizer, loader, generator, baseline = _snapshot_fixture()
    before = _snapshot_observable_state(model, optimizer, generator)
    _install_mutating_semantic_ids(
        model,
        optimizer,
        raise_after_mutation=False,
    )

    with pytest.raises(RuntimeError, match="read-only corpus diagnostics mutated state"):
        diagnostics.capture_read_only_corpus_snapshot(
            model=model,
            optimizer=optimizer,
            canonical_loader=loader,
            diagnostic_loader_generator=generator,
            post_kmeans_state=baseline,
            optimizer_step=0,
            completed_epoch=None,
            triggers=("optimizer_step:0",),
        )

    assert _snapshot_observable_state(model, optimizer, generator) == before


def test_snapshot_rejects_non_post_kmeans_model_without_mutation() -> None:
    model, optimizer, loader, generator, baseline = _snapshot_fixture()
    model.layers[1].kmeans_initted = False
    before = _snapshot_observable_state(model, optimizer, generator)

    with pytest.raises(ValueError, match="post-kmeans"):
        diagnostics.capture_read_only_corpus_snapshot(
            model=model,
            optimizer=optimizer,
            canonical_loader=loader,
            diagnostic_loader_generator=generator,
            post_kmeans_state=baseline,
            optimizer_step=0,
            completed_epoch=None,
            triggers=("optimizer_step:0",),
        )

    assert _snapshot_observable_state(model, optimizer, generator) == before
def _real_tiny_checkpoint_fixture(
    tmp_path: Path,
    spec: OptimizerSpec,
) -> dict[str, Any]:
    diagnostics.seed_all(20260701)
    model = build_tiny_rqvae()
    for layer in model.layers:
        layer.kmeans_initted = True
    optimizer = build_diagnostic_optimizer(model, spec)
    inputs = torch.arange(64 * 8, dtype=torch.float32).reshape(64, 8) / 128
    optimizer.zero_grad(set_to_none=True)
    output = model(SimpleNamespace(x=inputs), gumbel_t=0.2)
    output.loss.backward()
    optimizer.step()

    rng_state = diagnostics.capture_rng_state()
    snapshot_path = tmp_path / f'{spec.name}-snapshots.jsonl'
    snapshot_bytes = (
        b'{"optimizer_step":1,"trigger":"optimizer_step:1"}\n'
        b'{"completed_epoch":1,"note":"exact bytes"}\n'
    )
    snapshot_path.write_bytes(snapshot_bytes)
    epoch_hashes = diagnostics.epoch_plan_hashes(1024, 20260701, 500)
    epoch_plan_rolling_hash = hash_json(epoch_hashes)
    checkpoint_path = tmp_path / f'{spec.name}-checkpoint.pt'
    save_kwargs = {
        'model': model,
        'optimizer': optimizer,
        'optimizer_spec': spec,
        'completed_epoch': 7,
        'optimizer_step': 13,
        'rng_state': rng_state,
        'common_initialization_hash': '1' * 64,
        'invariant_config_hash': '2' * 64,
        'complete_config_hash': '3' * 64,
        'dataset_sha256': '4' * 64,
        'seed': 20260701,
        'epoch_hashes': epoch_hashes,
        'epoch_plan_rolling_hash': epoch_plan_rolling_hash,
        'snapshot_jsonl_path': snapshot_path,
        'snapshot_record_count': 2,
        'snapshot_rolling_hash': sha256(snapshot_bytes).hexdigest(),
    }
    diagnostics.save_diagnostic_checkpoint(checkpoint_path, **save_kwargs)
    return {
        'checkpoint_path': checkpoint_path,
        'model': model,
        'optimizer': optimizer,
        'optimizer_spec': spec,
        'rng_state': rng_state,
        'snapshot_path': snapshot_path,
        'snapshot_bytes': snapshot_bytes,
        'epoch_hashes': epoch_hashes,
        'epoch_plan_rolling_hash': epoch_plan_rolling_hash,
        'save_kwargs': save_kwargs,
    }


def _checkpoint_load_kwargs(fixture: Mapping[str, Any]) -> dict[str, Any]:
    return {
        'optimizer_spec': fixture['optimizer_spec'],
        'expected_common_initialization_hash': '1' * 64,
        'expected_invariant_config_hash': '2' * 64,
        'expected_dataset_sha256': '4' * 64,
        'expected_seed': 20260701,
        'expected_epoch_plan_rolling_hash': fixture[
            'epoch_plan_rolling_hash'
        ],
        'expected_snapshot_jsonl_path': fixture['snapshot_path'],
    }


def _fresh_checkpoint_target(
    spec: OptimizerSpec,
) -> tuple[RqVae, torch.optim.Optimizer]:
    model = build_tiny_rqvae()
    optimizer = build_diagnostic_optimizer(model, spec)
    return model, optimizer


def _assert_rejected_without_runtime_mutation(
    fixture: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    model: RqVae,
    optimizer: torch.optim.Optimizer,
    expected_exception: type[Exception] = ValueError,
    match: str,
    load_overrides: Mapping[str, Any] | None = None,
) -> None:
    model_hash = diagnostics.hash_state_dict(
        diagnostics.clone_cpu_state_dict(model)
    )
    optimizer_hash = diagnostics.hash_nested_state(optimizer.state_dict())
    rng_hash = diagnostics.hash_rng_state(diagnostics.capture_rng_state())
    kmeans_flags = [layer.kmeans_initted for layer in model.layers]
    kwargs = _checkpoint_load_kwargs(fixture)
    if load_overrides:
        kwargs.update(load_overrides)

    with pytest.raises(expected_exception, match=match):
        diagnostics.load_diagnostic_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            **kwargs,
        )

    assert diagnostics.hash_state_dict(
        diagnostics.clone_cpu_state_dict(model)
    ) == model_hash
    assert diagnostics.hash_nested_state(optimizer.state_dict()) == optimizer_hash
    assert diagnostics.hash_rng_state(diagnostics.capture_rng_state()) == rng_hash
    assert [layer.kmeans_initted for layer in model.layers] == kmeans_flags


def test_diagnostic_checkpoint_public_signatures_are_exact() -> None:
    assert get_type_hints(diagnostics.save_diagnostic_checkpoint) == {
        'path': Path,
        'model': nn.Module,
        'optimizer': torch.optim.Optimizer,
        'optimizer_spec': OptimizerSpec,
        'completed_epoch': int,
        'optimizer_step': int,
        'rng_state': diagnostics.RngState,
        'common_initialization_hash': str,
        'invariant_config_hash': str,
        'complete_config_hash': str,
        'dataset_sha256': str,
        'seed': int,
        'epoch_hashes': Sequence[str],
        'epoch_plan_rolling_hash': str,
        'snapshot_jsonl_path': Path,
        'snapshot_record_count': int,
        'snapshot_rolling_hash': str,
        'return': type(None),
    }
    assert get_type_hints(diagnostics.load_diagnostic_checkpoint) == {
        'path': Path,
        'model': nn.Module,
        'optimizer': torch.optim.Optimizer,
        'optimizer_spec': OptimizerSpec,
        'expected_common_initialization_hash': str,
        'expected_invariant_config_hash': str,
        'expected_dataset_sha256': str,
        'expected_seed': int,
        'expected_epoch_plan_rolling_hash': str,
        'expected_snapshot_jsonl_path': Path,
        'return': dict[str, int],
    }
    assert list(
        inspect.signature(diagnostics.save_diagnostic_checkpoint).parameters
    ) == [
        'path',
        'model',
        'optimizer',
        'optimizer_spec',
        'completed_epoch',
        'optimizer_step',
        'rng_state',
        'common_initialization_hash',
        'invariant_config_hash',
        'complete_config_hash',
        'dataset_sha256',
        'seed',
        'epoch_hashes',
        'epoch_plan_rolling_hash',
        'snapshot_jsonl_path',
        'snapshot_record_count',
        'snapshot_rolling_hash',
    ]
    assert list(
        inspect.signature(diagnostics.load_diagnostic_checkpoint).parameters
    ) == [
        'path',
        'model',
        'optimizer',
        'optimizer_spec',
        'expected_common_initialization_hash',
        'expected_invariant_config_hash',
        'expected_dataset_sha256',
        'expected_seed',
        'expected_epoch_plan_rolling_hash',
        'expected_snapshot_jsonl_path',
    ]


@pytest.mark.parametrize(
    'spec',
    [OptimizerSpec.adagrad_default(), OptimizerSpec.adamw_author()],
    ids=['adagrad', 'adamw'],
)
def test_real_tiny_checkpoint_round_trip_restores_all_state_and_schema(
    tmp_path: Path,
    spec: OptimizerSpec,
) -> None:
    fixture = _real_tiny_checkpoint_fixture(tmp_path, spec)
    checkpoint_path = fixture['checkpoint_path']
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    assert set(checkpoint) == {
        'schema_version',
        'artifact_kind',
        'model_state',
        'optimizer_state',
        'rng_state',
        'model_state_hash',
        'optimizer_state_hash',
        'rng_state_hash',
        'training',
        'optimizer',
        'compatibility',
        'complete_config_hash',
        'epoch_plan_hashes',
        'snapshot_history',
    }
    assert checkpoint['schema_version'] == 1
    assert checkpoint['artifact_kind'] == 'rqvae_diagnostic_checkpoint'
    assert checkpoint['training'] == {
        'completed_epoch': 7,
        'optimizer_step': 13,
    }
    assert checkpoint['optimizer'] == {
        'treatment': optimizer_treatment_payload(spec),
        'treatment_hash': optimizer_treatment_hash(spec),
        'metadata': optimizer_metadata(fixture['optimizer']),
        'metadata_hash': optimizer_metadata_hash(fixture['optimizer']),
    }
    assert checkpoint['compatibility'] == {
        'optimizer_treatment_hash': optimizer_treatment_hash(spec),
        'common_initialization_hash': '1' * 64,
        'invariant_config_hash': '2' * 64,
        'dataset_sha256': '4' * 64,
        'seed': 20260701,
        'epoch_plan_rolling_hash': fixture['epoch_plan_rolling_hash'],
    }
    assert checkpoint['complete_config_hash'] == '3' * 64
    assert checkpoint['epoch_plan_hashes'] == fixture['epoch_hashes']
    assert checkpoint['snapshot_history'] == {
        'path': str(fixture['snapshot_path']),
        'record_count': 2,
        'rolling_hash': sha256(fixture['snapshot_bytes']).hexdigest(),
    }
    assert checkpoint['model_state_hash'] == diagnostics.hash_state_dict(
        checkpoint['model_state']
    )
    assert checkpoint['optimizer_state_hash'] == diagnostics.hash_nested_state(
        checkpoint['optimizer_state']
    )
    assert checkpoint['rng_state_hash'] == diagnostics.hash_rng_state(
        checkpoint['rng_state']
    )

    target_model, target_optimizer = _fresh_checkpoint_target(spec)
    random.random()
    np.random.random()
    torch.rand(5)
    resumed = diagnostics.load_diagnostic_checkpoint(
        checkpoint_path,
        model=target_model,
        optimizer=target_optimizer,
        **_checkpoint_load_kwargs(fixture),
    )

    assert resumed == {'start_epoch': 8, 'optimizer_step': 13}
    assert diagnostics.hash_state_dict(
        diagnostics.clone_cpu_state_dict(target_model)
    ) == checkpoint['model_state_hash']
    assert diagnostics.hash_nested_state(
        target_optimizer.state_dict()
    ) == checkpoint['optimizer_state_hash']
    assert diagnostics.hash_rng_state(
        diagnostics.capture_rng_state()
    ) == checkpoint['rng_state_hash']
    assert all(layer.kmeans_initted for layer in target_model.layers)


def test_checkpoint_publish_is_exclusive_and_cleans_temporary_file(
    tmp_path: Path,
) -> None:
    fixture = _real_tiny_checkpoint_fixture(
        tmp_path,
        OptimizerSpec.adagrad_default(),
    )
    checkpoint_path = fixture['checkpoint_path']
    published_bytes = checkpoint_path.read_bytes()

    with pytest.raises(FileExistsError):
        diagnostics.save_diagnostic_checkpoint(
            checkpoint_path,
            **fixture['save_kwargs'],
        )

    assert checkpoint_path.read_bytes() == published_bytes
    assert list(tmp_path.glob(f'.{checkpoint_path.name}.*.tmp')) == []


def test_cross_treatment_checkpoint_is_rejected_before_mutation(
    tmp_path: Path,
) -> None:
    fixture = _real_tiny_checkpoint_fixture(
        tmp_path,
        OptimizerSpec.adagrad_default(),
    )
    model, optimizer = _fresh_checkpoint_target(OptimizerSpec.adamw_author())

    _assert_rejected_without_runtime_mutation(
        fixture,
        checkpoint_path=fixture['checkpoint_path'],
        model=model,
        optimizer=optimizer,
        match='optimizer_treatment_hash',
        load_overrides={'optimizer_spec': OptimizerSpec.adamw_author()},
    )


@pytest.mark.parametrize(
    ('field', 'load_overrides'),
    [
        (
            'common_initialization_hash',
            {'expected_common_initialization_hash': 'a' * 64},
        ),
        (
            'invariant_config_hash',
            {'expected_invariant_config_hash': 'b' * 64},
        ),
        ('dataset_sha256', {'expected_dataset_sha256': 'c' * 64}),
        ('seed', {'expected_seed': 7}),
        (
            'epoch_plan_rolling_hash',
            {'expected_epoch_plan_rolling_hash': 'd' * 64},
        ),
    ],
)
def test_checkpoint_compatibility_mismatch_rejects_before_mutation(
    tmp_path: Path,
    field: str,
    load_overrides: dict[str, Any],
) -> None:
    fixture = _real_tiny_checkpoint_fixture(
        tmp_path,
        OptimizerSpec.adagrad_default(),
    )
    model, optimizer = _fresh_checkpoint_target(fixture['optimizer_spec'])

    _assert_rejected_without_runtime_mutation(
        fixture,
        checkpoint_path=fixture['checkpoint_path'],
        model=model,
        optimizer=optimizer,
        match=field,
        load_overrides=load_overrides,
    )


@pytest.mark.parametrize(
    'tamper_kind',
    ['model_state', 'optimizer_state', 'rng_state', 'optimizer_metadata'],
)
def test_checkpoint_tampered_state_or_metadata_rejects_before_mutation(
    tmp_path: Path,
    tamper_kind: str,
) -> None:
    fixture = _real_tiny_checkpoint_fixture(
        tmp_path,
        OptimizerSpec.adagrad_default(),
    )
    checkpoint = torch.load(
        fixture['checkpoint_path'],
        map_location='cpu',
        weights_only=False,
    )
    if tamper_kind == 'model_state':
        first_key = next(iter(checkpoint['model_state']))
        checkpoint['model_state'][first_key].add_(1)
        match = 'model state hash'
    elif tamper_kind == 'optimizer_state':
        first_state = next(iter(checkpoint['optimizer_state']['state'].values()))
        first_tensor = next(
            value for value in first_state.values() if isinstance(value, Tensor)
        )
        first_tensor.add_(1)
        match = 'optimizer state hash'
    elif tamper_kind == 'rng_state':
        checkpoint['rng_state']['torch_cpu'] = checkpoint['rng_state'][
            'torch_cpu'
        ].roll(1)
        match = 'RNG state hash'
    else:
        checkpoint['optimizer']['metadata']['defaults']['lr'] = 0.25
        checkpoint['optimizer']['metadata_hash'] = hash_json(
            checkpoint['optimizer']['metadata']
        )
        match = 'optimizer metadata'
    tampered_path = tmp_path / f'tampered-{tamper_kind}.pt'
    torch.save(checkpoint, tampered_path)
    model, optimizer = _fresh_checkpoint_target(fixture['optimizer_spec'])

    _assert_rejected_without_runtime_mutation(
        fixture,
        checkpoint_path=tampered_path,
        model=model,
        optimizer=optimizer,
        match=match,
    )


@pytest.mark.parametrize('tamper_kind', ['model_layout', 'optimizer_layout'])
def test_checkpoint_rejects_rehashed_incompatible_state_before_load_calls(
    tmp_path: Path,
    tamper_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _real_tiny_checkpoint_fixture(
        tmp_path,
        OptimizerSpec.adagrad_default(),
    )
    checkpoint = torch.load(
        fixture['checkpoint_path'],
        map_location='cpu',
        weights_only=False,
    )
    if tamper_kind == 'model_layout':
        first_key = next(iter(checkpoint['model_state']))
        checkpoint['model_state'][first_key] = checkpoint['model_state'][
            first_key
        ].reshape(-1)[:-1]
        checkpoint['model_state_hash'] = diagnostics.hash_state_dict(
            checkpoint['model_state']
        )
        match = 'model state shape'
    else:
        checkpoint['optimizer_state']['param_groups'][0]['params'].pop()
        checkpoint['optimizer_state_hash'] = diagnostics.hash_nested_state(
            checkpoint['optimizer_state']
        )
        match = 'optimizer state'
    tampered_path = tmp_path / f'rehashed-{tamper_kind}.pt'
    torch.save(checkpoint, tampered_path)
    model, optimizer = _fresh_checkpoint_target(fixture['optimizer_spec'])
    calls = {'model': 0, 'optimizer': 0}
    original_model_load = model.load_state_dict
    original_optimizer_load = optimizer.load_state_dict

    def tracked_model_load(*args, **kwargs):
        calls['model'] += 1
        return original_model_load(*args, **kwargs)

    def tracked_optimizer_load(*args, **kwargs):
        calls['optimizer'] += 1
        return original_optimizer_load(*args, **kwargs)

    monkeypatch.setattr(model, 'load_state_dict', tracked_model_load)
    monkeypatch.setattr(optimizer, 'load_state_dict', tracked_optimizer_load)

    _assert_rejected_without_runtime_mutation(
        fixture,
        checkpoint_path=tampered_path,
        model=model,
        optimizer=optimizer,
        match=match,
    )
    assert calls == {'model': 0, 'optimizer': 0}


@pytest.mark.parametrize(
    ('history_mutation', 'match'),
    [
        ('append_tail', 'record count'),
        ('change_record', 'rolling hash'),
        ('remove_final_newline', 'terminating newline'),
    ],
)
def test_checkpoint_requires_exact_newline_inclusive_jsonl_history(
    tmp_path: Path,
    history_mutation: str,
    match: str,
) -> None:
    fixture = _real_tiny_checkpoint_fixture(
        tmp_path,
        OptimizerSpec.adagrad_default(),
    )
    if history_mutation == 'append_tail':
        fixture['snapshot_path'].write_bytes(
            fixture['snapshot_bytes'] + b'{"unexpected":"tail"}\n'
        )
    elif history_mutation == 'change_record':
        fixture['snapshot_path'].write_bytes(
            fixture['snapshot_bytes'].replace(b'exact bytes', b'other bytes')
        )
    else:
        fixture['snapshot_path'].write_bytes(fixture['snapshot_bytes'][:-1])
    model, optimizer = _fresh_checkpoint_target(fixture['optimizer_spec'])

    _assert_rejected_without_runtime_mutation(
        fixture,
        checkpoint_path=fixture['checkpoint_path'],
        model=model,
        optimizer=optimizer,
        match=match,
    )


def test_checkpoint_rejects_different_jsonl_path_before_mutation(
    tmp_path: Path,
) -> None:
    fixture = _real_tiny_checkpoint_fixture(
        tmp_path,
        OptimizerSpec.adagrad_default(),
    )
    other_path = tmp_path / 'other.jsonl'
    other_path.write_bytes(fixture['snapshot_bytes'])
    model, optimizer = _fresh_checkpoint_target(fixture['optimizer_spec'])

    _assert_rejected_without_runtime_mutation(
        fixture,
        checkpoint_path=fixture['checkpoint_path'],
        model=model,
        optimizer=optimizer,
        match='snapshot history path',
        load_overrides={'expected_snapshot_jsonl_path': other_path},
    )


@pytest.mark.parametrize(
    ('override', 'match'),
    [
        ({'snapshot_record_count': 1}, 'record count'),
        ({'snapshot_rolling_hash': 'f' * 64}, 'rolling hash'),
    ],
)
def test_checkpoint_save_rejects_inconsistent_jsonl_metadata(
    tmp_path: Path,
    override: dict[str, Any],
    match: str,
) -> None:
    diagnostics.seed_all(20260701)
    model = build_tiny_rqvae()
    for layer in model.layers:
        layer.kmeans_initted = True
    spec = OptimizerSpec.adagrad_default()
    optimizer = build_diagnostic_optimizer(model, spec)
    snapshot_path = tmp_path / 'snapshots.jsonl'
    snapshot_bytes = b'{"record":1}\n{"record":2}\n'
    snapshot_path.write_bytes(snapshot_bytes)
    epoch_hashes = diagnostics.epoch_plan_hashes(1024, 20260701, 500)
    checkpoint_path = tmp_path / 'invalid-checkpoint.pt'
    kwargs = {
        'model': model,
        'optimizer': optimizer,
        'optimizer_spec': spec,
        'completed_epoch': 0,
        'optimizer_step': 0,
        'rng_state': diagnostics.capture_rng_state(),
        'common_initialization_hash': '1' * 64,
        'invariant_config_hash': '2' * 64,
        'complete_config_hash': '3' * 64,
        'dataset_sha256': '4' * 64,
        'seed': 20260701,
        'epoch_hashes': epoch_hashes,
        'epoch_plan_rolling_hash': hash_json(epoch_hashes),
        'snapshot_jsonl_path': snapshot_path,
        'snapshot_record_count': 2,
        'snapshot_rolling_hash': sha256(snapshot_bytes).hexdigest(),
        **override,
    }

    with pytest.raises(ValueError, match=match):
        diagnostics.save_diagnostic_checkpoint(checkpoint_path, **kwargs)
    assert not checkpoint_path.exists()



@pytest.mark.parametrize(
    ("snapshot_bytes", "match"),
    [
        (b'{"record":1}\n\n', "blank"),
        (b'{"record":1}\nnot-json\n', "valid JSON"),
    ],
)
def test_checkpoint_save_rejects_self_consistent_invalid_jsonl_records(
    tmp_path: Path,
    snapshot_bytes: bytes,
    match: str,
) -> None:
    model = build_tiny_rqvae()
    for layer in model.layers:
        layer.kmeans_initted = True
    spec = OptimizerSpec.adagrad_default()
    optimizer = build_diagnostic_optimizer(model, spec)
    snapshot_path = tmp_path / "invalid-records.jsonl"
    snapshot_path.write_bytes(snapshot_bytes)
    epoch_hashes = diagnostics.epoch_plan_hashes(1024, 20260701, 500)
    checkpoint_path = tmp_path / "invalid-records-checkpoint.pt"

    with pytest.raises(ValueError, match=match):
        diagnostics.save_diagnostic_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            optimizer_spec=spec,
            completed_epoch=0,
            optimizer_step=0,
            rng_state=diagnostics.capture_rng_state(),
            common_initialization_hash="1" * 64,
            invariant_config_hash="2" * 64,
            complete_config_hash="3" * 64,
            dataset_sha256="4" * 64,
            seed=20260701,
            epoch_hashes=epoch_hashes,
            epoch_plan_rolling_hash=hash_json(epoch_hashes),
            snapshot_jsonl_path=snapshot_path,
            snapshot_record_count=snapshot_bytes.count(b"\n"),
            snapshot_rolling_hash=sha256(snapshot_bytes).hexdigest(),
        )
    assert not checkpoint_path.exists()


def test_checkpoint_load_rejects_self_consistent_malformed_jsonl_before_mutation(
    tmp_path: Path,
) -> None:
    fixture = _real_tiny_checkpoint_fixture(
        tmp_path,
        OptimizerSpec.adagrad_default(),
    )
    malformed_bytes = b'{"record":1}\nnot-json\n'
    fixture["snapshot_path"].write_bytes(malformed_bytes)
    checkpoint = torch.load(
        fixture["checkpoint_path"],
        map_location="cpu",
        weights_only=False,
    )
    checkpoint["snapshot_history"]["record_count"] = 2
    checkpoint["snapshot_history"]["rolling_hash"] = sha256(
        malformed_bytes
    ).hexdigest()
    tampered_path = tmp_path / "self-consistent-malformed-checkpoint.pt"
    torch.save(checkpoint, tampered_path)
    model, optimizer = _fresh_checkpoint_target(fixture["optimizer_spec"])

    _assert_rejected_without_runtime_mutation(
        fixture,
        checkpoint_path=tampered_path,
        model=model,
        optimizer=optimizer,
        match="valid JSON",
    )



def test_exclusive_checkpoint_publish_fsyncs_file_then_parent_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "durable-checkpoint.pt"
    events: list[str] = []
    real_save = diagnostics.torch.save
    real_link = diagnostics.os.link
    real_fsync = diagnostics.os.fsync

    def tracked_save(payload: object, path: Path) -> None:
        events.append("save")
        real_save(payload, path)

    def tracked_fsync(file_descriptor: int) -> None:
        mode = os.fstat(file_descriptor).st_mode
        events.append("fsync_directory" if stat.S_ISDIR(mode) else "fsync_file")
        real_fsync(file_descriptor)

    def tracked_link(source: Path, target: Path) -> None:
        events.append("link")
        real_link(source, target)

    monkeypatch.setattr(diagnostics.torch, "save", tracked_save)
    monkeypatch.setattr(diagnostics.os, "fsync", tracked_fsync)
    monkeypatch.setattr(diagnostics.os, "link", tracked_link)

    diagnostics._exclusive_torch_publish(destination, {"value": 1})

    assert events == ["save", "fsync_file", "link", "fsync_directory"]
    assert destination.is_file()
    assert list(tmp_path.glob(f".{destination.name}.*.tmp")) == []


@pytest.mark.parametrize("failure_stage", ["file_fsync", "directory_fsync"])
def test_exclusive_checkpoint_publish_cleans_all_outputs_on_fsync_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    destination = tmp_path / f"failed-{failure_stage}.pt"
    real_fsync = diagnostics.os.fsync
    call_count = 0

    def failing_fsync(file_descriptor: int) -> None:
        nonlocal call_count
        call_count += 1
        if failure_stage == "file_fsync" and call_count == 1:
            raise OSError("file fsync failure")
        if failure_stage == "directory_fsync" and call_count == 2:
            raise OSError("directory fsync failure")
        real_fsync(file_descriptor)

    monkeypatch.setattr(diagnostics.os, "fsync", failing_fsync)

    with pytest.raises(OSError, match="fsync failure"):
        diagnostics._exclusive_torch_publish(destination, {"value": 1})

    assert not destination.exists()
    assert list(tmp_path.glob(f".{destination.name}.*.tmp")) == []



@pytest.mark.parametrize(
    "failure_stage",
    ["optimizer_load", "model_post_hook", "rng_restore"],
)
def test_checkpoint_load_rolls_back_every_runtime_state_after_mutation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    fixture = _real_tiny_checkpoint_fixture(
        tmp_path,
        OptimizerSpec.adagrad_default(),
    )
    model, optimizer = _fresh_checkpoint_target(fixture["optimizer_spec"])
    original_model_state = diagnostics.clone_cpu_state_dict(model)
    original_model_hash = diagnostics.hash_state_dict(original_model_state)
    original_optimizer_state = diagnostics._clone_optimizer_state(
        optimizer.state_dict()
    )
    original_optimizer_hash = diagnostics.hash_nested_state(
        original_optimizer_state
    )
    original_rng_state = diagnostics.capture_rng_state()
    original_rng_hash = diagnostics.hash_rng_state(original_rng_state)
    original_kmeans_flags = [layer.kmeans_initted for layer in model.layers]
    hook_handle = None
    calls = 0

    if failure_stage == "optimizer_load":
        original_load = optimizer.load_state_dict

        def fail_after_optimizer_mutation(state: Mapping[str, Any]) -> None:
            nonlocal calls
            original_load(state)
            calls += 1
            if calls == 1:
                raise RuntimeError("optimizer load failure after mutation")

        monkeypatch.setattr(
            optimizer,
            "load_state_dict",
            fail_after_optimizer_mutation,
        )
    elif failure_stage == "model_post_hook":

        def fail_first_model_post_hook(
            module: nn.Module,
            incompatible_keys: object,
        ) -> None:
            del incompatible_keys
            nonlocal calls
            calls += 1
            if calls == 1:
                for layer in module.layers:
                    layer.kmeans_initted = True
                raise RuntimeError("model post-hook failure after mutation")

        hook_handle = model.register_load_state_dict_post_hook(
            fail_first_model_post_hook
        )
    else:
        original_restore = diagnostics.restore_rng_state

        def fail_after_rng_mutation(state: diagnostics.RngState) -> None:
            nonlocal calls
            original_restore(state)
            calls += 1
            if calls == 1:
                raise RuntimeError("RNG restore failure after mutation")

        monkeypatch.setattr(
            diagnostics,
            "restore_rng_state",
            fail_after_rng_mutation,
        )

    with pytest.raises(RuntimeError, match="failure after mutation"):
        diagnostics.load_diagnostic_checkpoint(
            fixture["checkpoint_path"],
            model=model,
            optimizer=optimizer,
            **_checkpoint_load_kwargs(fixture),
        )

    if hook_handle is not None:
        hook_handle.remove()
    assert calls == 1
    assert diagnostics.hash_state_dict(
        diagnostics.clone_cpu_state_dict(model)
    ) == original_model_hash
    assert diagnostics.hash_nested_state(
        optimizer.state_dict()
    ) == original_optimizer_hash
    assert diagnostics.hash_rng_state(
        diagnostics.capture_rng_state()
    ) == original_rng_hash
    assert [layer.kmeans_initted for layer in model.layers] == (
        original_kmeans_flags
    )
