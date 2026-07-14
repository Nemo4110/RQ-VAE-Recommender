from dataclasses import FrozenInstanceError
from dataclasses import asdict
from hashlib import sha256

import pytest
import torch
from torch import nn

import rqvae_collapse_diagnostics as diagnostics
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
