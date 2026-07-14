from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

import train_rqvae_diagnostic as trainer
from rqvae_collapse_diagnostics import FailureStatus


def valid_kwargs(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "run_mode": "bounded",
        "epochs": 500,
        "max_items": None,
        "batch_size": 1024,
        "initialization_batch_size": 1024,
        "num_workers": 0,
        "drop_last": False,
        "amp": False,
        "dataset_split": "beauty",
        "dataset_item_count": 12101,
        "checkpoint_epochs": (100, 250, 500),
        "snapshot_optimizer_steps": (0, 1, 2, 3, 12),
        "snapshot_epochs": (1, 2, 5, 10, 25, 50, 100, 250, 500),
        "snapshot_batch_size": 1024,
        "vae_input_dim": 768,
        "vae_hidden_dims": (512, 256, 128),
        "vae_embed_dim": 32,
        "vae_codebook_size": 256,
        "vae_n_layers": 3,
        "commitment_weight": 0.25,
        "gumbel_temperature": 0.2,
    }
    values.update(overrides)
    return values


@pytest.mark.parametrize(
    ("mode", "epochs", "max_items", "checkpoints"),
    [
        ("smoke", 2, 2048, (2,)),
        ("bounded", 500, None, (100, 250, 500)),
    ],
)
def test_run_matrix_accepts_only_fixed_modes(
    mode: str,
    epochs: int,
    max_items: int | None,
    checkpoints: tuple[int, ...],
) -> None:
    trainer.validate_diagnostic_run(
        **valid_kwargs(
            run_mode=mode,
            epochs=epochs,
            max_items=max_items,
            checkpoint_epochs=checkpoints,
        )
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"run_mode": "full"},
        {"epochs": 499},
        {"max_items": 2048},
        {"batch_size": 640},
        {"num_workers": 1},
        {"drop_last": True},
        {"amp": True},
        {"checkpoint_epochs": (500,)},
        {"snapshot_optimizer_steps": (0, 1)},
        {"snapshot_epochs": (1, 500)},
    ],
)
def test_run_matrix_rejects_protocol_drift(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        trainer.validate_diagnostic_run(**valid_kwargs(**overrides))


def test_schedule_coalesces_step_12_and_epoch_1_into_13_states() -> None:
    states = trainer.planned_snapshot_states(
        epochs=500,
        training_item_count=12101,
        batch_size=1024,
        step_schedule=(0, 1, 2, 3, 12),
        epoch_schedule=(1, 2, 5, 10, 25, 50, 100, 250, 500),
    )

    assert len(states) == 13
    assert states[4] == {
        "optimizer_step": 12,
        "completed_epoch": 1,
        "triggers": ["optimizer_step:12", "epoch:1"],
    }


def test_checkpoint_paths_are_exact_and_have_no_final_alias(tmp_path: Path) -> None:
    assert trainer.checkpoint_path_for(
        tmp_path, completed_epoch=100, optimizer_step=1200
    ).name == "checkpoint_epoch_00100_step_0001200.pt"
    assert not hasattr(trainer, "publish_final_checkpoint")


def test_emergency_failure_is_exclusive_and_allows_preoptimizer_null_hashes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "emergency.json"
    payload = trainer.write_emergency_failure(
        path,
        failure_status=FailureStatus.INVARIANT_MISMATCH,
        failure_message="common init mismatch",
        boundary="pre_optimizer_initialization",
        completed_epoch=0,
        optimizer_step=0,
        attempted_optimizer_step=0,
        losses=None,
        gradient_stats=None,
        model_hash=None,
        optimizer_state_hash=None,
        rng_hash="a" * 64,
        common_initialization_hash=None,
        invariant_config_hash=None,
    )

    assert json.loads(path.read_text()) == payload
    assert payload["model_hash"] is None
    assert payload["optimizer_state_hash"] is None
    with pytest.raises(FileExistsError):
        trainer.write_emergency_failure(
            path,
            failure_status=FailureStatus.RUNTIME_ERROR,
            failure_message="again",
            boundary="startup",
            completed_epoch=0,
            optimizer_step=0,
            attempted_optimizer_step=0,
            losses=None,
            gradient_stats=None,
            model_hash=None,
            optimizer_state_hash=None,
            rng_hash="b" * 64,
            common_initialization_hash=None,
            invariant_config_hash=None,
        )


def test_summary_schema_and_evidence_are_fixed() -> None:
    required = {
        "dataset",
        "dataset_split",
        "dataset_item_count",
        "training_item_count",
        "model",
        "seed",
        "config",
        "complete_config_hash",
        "invariant_config_hash",
        "initialization_config_hash",
        "common_initialization_hash",
        "epoch_plan_rolling_hash",
        "optimizer",
        "completed_epoch",
        "optimizer_step",
        "losses",
        "collapse",
        "failure_status",
        "failure_message",
        "timings",
        "device",
        "external",
        "artifacts",
        "evidence_level",
        "final_assignment",
        "local_promising",
        "paper_gate_passed",
    }
    assert required <= trainer.SUMMARY_FIELDS
    assert trainer.evidence_level("smoke") == "diagnostic_smoke"
    assert trainer.evidence_level("bounded") == "diagnostic_bounded"


def test_deadline_parser_requires_aware_utc() -> None:
    deadline = trainer.parse_utc_deadline("2026-07-14T19:40:00Z")
    assert deadline.isoformat() == "2026-07-14T19:40:00+00:00"
    with pytest.raises(ValueError):
        trainer.parse_utc_deadline("2026-07-14T19:40:00")


def test_cli_prepare_dispatch_never_calls_training(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(trainer, "parse_config", lambda: calls.append("parse"))
    monkeypatch.setattr(
        trainer,
        "prepare_common_initialization_from_gin",
        lambda: calls.append("prepare") or {},
    )
    monkeypatch.setattr(trainer, "train", lambda: calls.append("train") or {})

    trainer.main(["bounded.gin", "--prepare-common-initialization"])

    assert calls == ["parse", "prepare"]


def test_snapshot_record_bytes_are_canonical_and_fresh_file_is_exclusive(
    tmp_path: Path,
) -> None:
    path = tmp_path / "snapshots.jsonl"
    history = trainer.SnapshotHistory.create(path)
    history.append({"z": 2, "a": 1})
    assert path.read_bytes() == b'{"a":1,"z":2}\n'
    assert history.record_count == 1
    with pytest.raises(FileExistsError):
        trainer.SnapshotHistory.create(path)


def test_resume_history_rejects_any_extra_tail(tmp_path: Path) -> None:
    path = tmp_path / "snapshots.jsonl"
    history = trainer.SnapshotHistory.create(path)
    history.append({"optimizer_step": 0})
    count, rolling_hash = history.record_count, history.rolling_hash
    with path.open("ab") as stream:
        stream.write(b'{"optimizer_step":1}\n')

    with pytest.raises(ValueError, match="snapshot history"):
        trainer.SnapshotHistory.resume(
            path,
            expected_record_count=count,
            expected_rolling_hash=rolling_hash,
        )


class _SizedDataset:
    def __len__(self) -> int:
        return 12101


def _patch_summary_failure_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(trainer.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(trainer.torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(trainer.torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(trainer.torch.cuda, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(trainer, "_device_metadata", lambda device: {"type": "cuda"})
    monkeypatch.setattr(trainer, "_external_metadata", lambda: {"commit": "test"})


def test_failed_summary_records_loaded_training_dataset_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_summary_failure_runtime(monkeypatch)
    monkeypatch.setattr(trainer, "_load_dataset", lambda *args: _SizedDataset())
    monkeypatch.setattr(
        trainer,
        "build_paper_model",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("model build failed")),
    )
    save_root = tmp_path / "loaded-failure"

    with pytest.raises(RuntimeError, match="model build failed"):
        trainer.train(
            save_dir_root=str(save_root),
            common_initialization_path=str(tmp_path / "common.pt"),
            run_stop_utc="2099-01-01T00:00:00Z",
        )

    summary = json.loads((save_root / "summary.json").read_text())
    assert summary["dataset_item_count"] == 12101
    assert summary["training_item_count"] == 2048


def test_failed_summary_before_dataset_load_keeps_training_size_null(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_summary_failure_runtime(monkeypatch)
    monkeypatch.setattr(
        trainer,
        "_load_dataset",
        lambda *args: (_ for _ in ()).throw(RuntimeError("dataset load failed")),
    )
    save_root = tmp_path / "preload-failure"

    with pytest.raises(RuntimeError, match="dataset load failed"):
        trainer.train(
            save_dir_root=str(save_root),
            common_initialization_path=str(tmp_path / "common.pt"),
            run_stop_utc="2099-01-01T00:00:00Z",
        )

    summary = json.loads((save_root / "summary.json").read_text())
    assert summary["dataset_item_count"] == 12101
    assert summary["training_item_count"] is None



def test_summary_does_not_probe_nested_locals_for_dataset_state() -> None:
    source = inspect.getsource(trainer.train)
    assert '"training_dataset" in locals()' not in source
    assert '"training_item_count": training_item_count' in source



def test_nonfinite_emergency_losses_are_sanitized_without_reclassification(
    tmp_path: Path,
) -> None:
    path = tmp_path / "nonfinite-emergency.json"
    payload = trainer.write_emergency_failure(
        path,
        failure_status=FailureStatus.NUMERICAL_FAILURE,
        failure_message="nonfinite loss before backward",
        boundary="pre_backward_pre_update",
        completed_epoch=0,
        optimizer_step=0,
        attempted_optimizer_step=1,
        losses={
            "loss": float("nan"),
            "reconstruction_loss": float("inf"),
            "rqvae_loss": float("-inf"),
        },
        gradient_stats=None,
        model_hash="1" * 64,
        optimizer_state_hash="2" * 64,
        rng_hash="3" * 64,
        common_initialization_hash="4" * 64,
        invariant_config_hash="5" * 64,
    )

    stored = json.loads(path.read_text())
    assert stored == payload
    assert stored["failure_status"] == "numerical_failure"
    assert stored["failure_message"] == "nonfinite loss before backward"
    assert stored["losses"] == {
        "loss": None,
        "reconstruction_loss": None,
        "rqvae_loss": None,
    }


def test_fully_qualified_gin_config_resolves_module_train_for_prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "bounded.gin"
    config.write_text(
        "\n".join(
            [
                'train_rqvae_diagnostic.train.run_mode = "bounded"',
                "train_rqvae_diagnostic.train.epochs = 500",
                "train_rqvae_diagnostic.train.max_items = None",
                "train_rqvae_diagnostic.train.checkpoint_epochs = (100, 250, 500)",
                'train_rqvae_diagnostic.train.common_initialization_path = "/tmp/common.pt"',
            ]
        )
        + "\n"
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        trainer,
        "_prepare_common_initialization",
        lambda **kwargs: captured.update(kwargs) or {"prepared": True},
    )
    trainer.gin.clear_config()
    try:
        result = trainer.main([str(config), "--prepare-common-initialization"])
    finally:
        trainer.gin.clear_config()

    assert result == {"prepared": True}
    assert captured["run_mode"] == "bounded"
    assert captured["epochs"] == 500
    assert captured["max_items"] is None
    assert captured["checkpoint_epochs"] == (100, 250, 500)


def test_validated_resume_checkpoint_is_reused_at_same_deadline_position(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint_epoch_00002_step_0000004.pt"
    trainer.torch.save(
        {"training": {"completed_epoch": 2, "optimizer_step": 4}},
        path,
    )

    assert trainer.reusable_checkpoint_at_position(
        path,
        validated_resume_checkpoint=path,
        completed_epoch=2,
        optimizer_step=4,
    ) is path
    with pytest.raises(FileExistsError):
        trainer.reusable_checkpoint_at_position(
            path,
            validated_resume_checkpoint=None,
            completed_epoch=2,
            optimizer_step=4,
        )


def test_oom_emergency_uses_only_cached_cpu_safe_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("OOM emergency touched live CUDA state")

    monkeypatch.setattr(trainer, "clone_cpu_state_dict", forbidden)
    monkeypatch.setattr(trainer, "hash_nested_state", forbidden)
    monkeypatch.setattr(trainer, "capture_rng_state", forbidden)
    path = tmp_path / "oom-emergency.json"

    payload = trainer.write_oom_emergency_failure(
        path,
        failure_message="CUDA out of memory",
        completed_epoch=3,
        optimizer_step=36,
        attempted_optimizer_step=37,
        losses={"loss": 1.0},
        cached_rng_hash="a" * 64,
        common_initialization_hash="b" * 64,
        invariant_config_hash="c" * 64,
    )

    assert payload["failure_status"] == "oom"
    assert payload["boundary"] == "cpu_only_oom"
    assert payload["model_hash"] is None
    assert payload["optimizer_state_hash"] is None
    assert payload["rng_hash"] == "a" * 64
    assert json.loads(path.read_text()) == payload



def test_epoch_100_cadence_checkpoint_is_reused_before_epoch_101_deadline(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint_epoch_00100_step_0001200.pt"
    trainer.torch.save(
        {"training": {"completed_epoch": 100, "optimizer_step": 1200}},
        path,
    )

    assert trainer.reusable_checkpoint_at_position(
        path,
        validated_resume_checkpoint=None,
        current_process_checkpoint=path,
        completed_epoch=100,
        optimizer_step=1200,
    ) is path
