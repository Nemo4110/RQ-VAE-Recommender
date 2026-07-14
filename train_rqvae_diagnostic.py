from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import gin
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from data.processed import ItemData, RecDataset
from data.utils import batch_to
from modules.utils import parse_config
from rqvae_collapse_diagnostics import (
    FailureStatus,
    OptimizerSpec,
    build_diagnostic_optimizer,
    canonical_json,
    capture_read_only_corpus_snapshot,
    capture_rng_state,
    classify_collapse,
    clone_cpu_state_dict,
    clone_group_parameters,
    complete_config_hash,
    epoch_permutation,
    epoch_plan_hashes,
    gradient_group_stats,
    hash_nested_state,
    hash_rng_state,
    hash_state_dict,
    initialization_config,
    initialization_config_hash,
    invariant_config_hash,
    load_common_initialization,
    load_diagnostic_checkpoint,
    load_items_by_index,
    optimizer_metadata,
    optimizer_metadata_hash,
    optimizer_treatment_hash,
    optimizer_treatment_payload,
    parameter_delta_stats,
    parameter_groups,
    rolling_epoch_plan_hash,
    save_common_initialization,
    save_diagnostic_checkpoint,
    scheduled_snapshot_triggers,
    seed_all,
)
from train_rqvae_paper import build_paper_model


_STEP_SCHEDULE = (0, 1, 2, 3, 12)
_EPOCH_SCHEDULE = (1, 2, 5, 10, 25, 50, 100, 250, 500)
_RUN_MATRIX: dict[str, tuple[int, int | None, tuple[int, ...]]] = {
    "smoke": (2, 2048, (2,)),
    "bounded": (500, None, (100, 250, 500)),
}

SUMMARY_FIELDS = frozenset(
    {
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
)


def validate_diagnostic_run(
    *,
    run_mode: str,
    epochs: int,
    max_items: int | None,
    batch_size: int,
    initialization_batch_size: int,
    num_workers: int,
    drop_last: bool,
    amp: bool,
    dataset_split: str,
    dataset_item_count: int,
    checkpoint_epochs: Sequence[int],
    snapshot_optimizer_steps: Sequence[int],
    snapshot_epochs: Sequence[int],
    snapshot_batch_size: int,
    vae_input_dim: int,
    vae_hidden_dims: Sequence[int],
    vae_embed_dim: int,
    vae_codebook_size: int,
    vae_n_layers: int,
    commitment_weight: float,
    gumbel_temperature: float,
) -> None:
    if run_mode not in _RUN_MATRIX:
        raise ValueError(f"unsupported diagnostic run mode: {run_mode}")
    expected_epochs, expected_items, expected_checkpoints = _RUN_MATRIX[run_mode]
    if epochs != expected_epochs or max_items != expected_items:
        raise ValueError(
            f"{run_mode} requires epochs={expected_epochs} and max_items={expected_items}"
        )
    exact = {
        "batch_size": (batch_size, 1024),
        "initialization_batch_size": (initialization_batch_size, 1024),
        "num_workers": (num_workers, 0),
        "drop_last": (drop_last, False),
        "amp": (amp, False),
        "dataset_split": (dataset_split, "beauty"),
        "dataset_item_count": (dataset_item_count, 12101),
        "snapshot_batch_size": (snapshot_batch_size, 1024),
        "vae_input_dim": (vae_input_dim, 768),
        "vae_embed_dim": (vae_embed_dim, 32),
        "vae_codebook_size": (vae_codebook_size, 256),
        "vae_n_layers": (vae_n_layers, 3),
        "commitment_weight": (commitment_weight, 0.25),
        "gumbel_temperature": (gumbel_temperature, 0.2),
    }
    for field, (actual, expected) in exact.items():
        if actual != expected or type(actual) is not type(expected):
            raise ValueError(f"diagnostic {field} must equal {expected!r}")
    if list(vae_hidden_dims) != [512, 256, 128]:
        raise ValueError("diagnostic hidden dimensions must equal [512, 256, 128]")
    if tuple(checkpoint_epochs) != expected_checkpoints:
        raise ValueError(f"{run_mode} checkpoint epochs must equal {expected_checkpoints}")
    if tuple(snapshot_optimizer_steps) != _STEP_SCHEDULE:
        raise ValueError(f"snapshot optimizer steps must equal {_STEP_SCHEDULE}")
    if tuple(snapshot_epochs) != _EPOCH_SCHEDULE:
        raise ValueError(f"snapshot epochs must equal {_EPOCH_SCHEDULE}")


def checkpoint_path_for(
    save_root: Path, *, completed_epoch: int, optimizer_step: int
) -> Path:
    return Path(save_root) / (
        f"checkpoint_epoch_{completed_epoch:05d}_step_{optimizer_step:07d}.pt"
    )


def evidence_level(run_mode: str) -> str:
    if run_mode == "smoke":
        return "diagnostic_smoke"
    if run_mode == "bounded":
        return "diagnostic_bounded"
    raise ValueError(f"unsupported diagnostic run mode: {run_mode}")


def parse_utc_deadline(value: str) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise ValueError("run_stop_utc must use an explicit UTC Z suffix")
    parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("run_stop_utc must be UTC")
    return parsed


def planned_snapshot_states(
    *,
    epochs: int,
    training_item_count: int,
    batch_size: int,
    step_schedule: Sequence[int],
    epoch_schedule: Sequence[int],
) -> list[dict[str, Any]]:
    steps_per_epoch = math.ceil(training_item_count / batch_size)
    states: dict[tuple[int, int | None], list[str]] = {}
    for step in step_schedule:
        if step == 0:
            states[(0, None)] = ["optimizer_step:0"]
            continue
        completed_epoch = step // steps_per_epoch if step % steps_per_epoch == 0 else None
        states[(step, completed_epoch)] = [f"optimizer_step:{step}"]
    for epoch in epoch_schedule:
        if epoch > epochs:
            continue
        step = epoch * steps_per_epoch
        key = (step, epoch)
        states.setdefault(key, [])
        step_trigger = f"optimizer_step:{step}"
        if step in step_schedule and step_trigger not in states[key]:
            states[key].append(step_trigger)
        states[key].append(f"epoch:{epoch}")
        if step in step_schedule:
            states.pop((step, None), None)
    return [
        {"optimizer_step": step, "completed_epoch": epoch, "triggers": triggers}
        for (step, epoch), triggers in sorted(states.items(), key=lambda item: item[0][0])
    ]


def _fsync_parent(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(Path(path).parent, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    data = (canonical_json(dict(payload)) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_path, destination)
        published = True
        _fsync_parent(destination)
    except BaseException:
        if published:
            try:
                destination.unlink(missing_ok=True)
                _fsync_parent(destination)
            except OSError:
                pass
        raise
    finally:
        temporary_path.unlink(missing_ok=True)


def _sanitize_failure_json(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _sanitize_failure_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_failure_json(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_failure_json(item) for item in value]
    return value


def write_emergency_failure(
    path: Path,
    *,
    failure_status: FailureStatus,
    failure_message: str,
    boundary: str,
    completed_epoch: int,
    optimizer_step: int,
    attempted_optimizer_step: int,
    losses: Mapping[str, float] | None,
    gradient_stats: Mapping[str, Any] | None,
    model_hash: str | None,
    optimizer_state_hash: str | None,
    rng_hash: str,
    common_initialization_hash: str | None,
    invariant_config_hash: str | None,
) -> dict[str, Any]:
    if not isinstance(failure_status, FailureStatus):
        raise TypeError("failure_status must be a FailureStatus")
    payload = {
        "schema_version": 1,
        "artifact_kind": "rqvae_diagnostic_emergency_failure",
        "failure_status": failure_status.value,
        "failure_message": failure_message,
        "boundary": boundary,
        "completed_epoch": completed_epoch,
        "optimizer_step": optimizer_step,
        "attempted_optimizer_step": attempted_optimizer_step,
        "losses": None if losses is None else _sanitize_failure_json(losses),
        "gradient_stats": (
            None if gradient_stats is None else _sanitize_failure_json(gradient_stats)
        ),
        "model_hash": model_hash,
        "optimizer_state_hash": optimizer_state_hash,
        "rng_hash": rng_hash,
        "common_initialization_hash": common_initialization_hash,
        "invariant_config_hash": invariant_config_hash,
    }
    _exclusive_json(Path(path), payload)
    return payload


def write_oom_emergency_failure(
    path: Path,
    *,
    failure_message: str,
    completed_epoch: int,
    optimizer_step: int,
    attempted_optimizer_step: int,
    losses: Mapping[str, float] | None,
    cached_rng_hash: str,
    common_initialization_hash: str | None,
    invariant_config_hash: str | None,
) -> dict[str, Any]:
    return write_emergency_failure(
        path,
        failure_status=FailureStatus.OOM,
        failure_message=failure_message,
        boundary="cpu_only_oom",
        completed_epoch=completed_epoch,
        optimizer_step=optimizer_step,
        attempted_optimizer_step=attempted_optimizer_step,
        losses=losses,
        gradient_stats=None,
        model_hash=None,
        optimizer_state_hash=None,
        rng_hash=cached_rng_hash,
        common_initialization_hash=common_initialization_hash,
        invariant_config_hash=invariant_config_hash,
    )


def reusable_checkpoint_at_position(
    path: Path,
    *,
    validated_resume_checkpoint: Path | None,
    completed_epoch: int,
    optimizer_step: int,
    current_process_checkpoint: Path | None = None,
) -> Path:
    checkpoint_path = path
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    reusable_sources = (
        validated_resume_checkpoint,
        current_process_checkpoint,
    )
    if not any(
        candidate is not None
        and candidate.resolve() == checkpoint_path.resolve()
        for candidate in reusable_sources
    ):
        raise FileExistsError(
            f"refusing to reuse unvalidated immutable checkpoint {checkpoint_path}"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    training = checkpoint.get("training") if isinstance(checkpoint, Mapping) else None
    expected = {
        "completed_epoch": completed_epoch,
        "optimizer_step": optimizer_step,
    }
    if not isinstance(training, Mapping) or dict(training) != expected:
        raise ValueError("existing validated checkpoint training position mismatch")
    return checkpoint_path


class SnapshotHistory:
    def __init__(self, path: Path, record_count: int, rolling_hash: str) -> None:
        self.path = Path(path)
        self.record_count = record_count
        self.rolling_hash = rolling_hash

    @staticmethod
    def _metadata(path: Path) -> tuple[int, str]:
        data = Path(path).read_bytes()
        if data and not data.endswith(b"\n"):
            raise ValueError("snapshot history final record lacks newline")
        records = data.splitlines(keepends=True)
        for record in records:
            json.loads(record.decode("utf-8"))
        return len(records), hashlib.sha256(data).hexdigest()

    @classmethod
    def create(cls, path: Path) -> "SnapshotHistory":
        path = Path(path)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_parent(path)
        return cls(path, 0, hashlib.sha256(b"").hexdigest())

    @classmethod
    def resume(
        cls,
        path: Path,
        *,
        expected_record_count: int,
        expected_rolling_hash: str,
    ) -> "SnapshotHistory":
        count, digest = cls._metadata(Path(path))
        if count != expected_record_count or digest != expected_rolling_hash:
            raise ValueError("snapshot history does not match checkpoint")
        return cls(Path(path), count, digest)

    def append(self, payload: Mapping[str, Any]) -> None:
        record = (canonical_json(dict(payload)) + "\n").encode("utf-8")
        original_size = self.path.stat().st_size
        descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND)
        try:
            written = 0
            while written < len(record):
                count = os.write(descriptor, record[written:])
                if count <= 0:
                    raise OSError("snapshot append made no progress")
                written += count
            os.fsync(descriptor)
        except BaseException as error:
            try:
                os.ftruncate(descriptor, original_size)
                os.fsync(descriptor)
            except BaseException as rollback_error:
                error.add_note(f"snapshot append rollback failed: {rollback_error}")
            raise
        finally:
            os.close(descriptor)
        record_count, rolling_hash = self._metadata(self.path)
        self.record_count = record_count
        self.rolling_hash = rolling_hash


def _external_metadata() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
    return {
        "url": "https://github.com/EdoardoBotta/RQ-VAE-Recommender",
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "checkout_path": str(root),
        "implementation": "locally_modified_third_party",
    }


def _device_metadata(device: torch.device) -> dict[str, Any]:
    index = device.index if device.index is not None else torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(index)
    return {
        "type": "cuda",
        "index": index,
        "name": torch.cuda.get_device_name(index),
        "capability": [major, minor],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def _training_config(locals_payload: Mapping[str, Any]) -> dict[str, Any]:
    names = inspect.signature(train).parameters
    return {
        name: (
            list(locals_payload[name])
            if isinstance(locals_payload[name], tuple)
            else locals_payload[name]
        )
        for name in names
    }


def _load_dataset(dataset_folder: str, dataset_split: str) -> ItemData:
    return ItemData(
        root=dataset_folder,
        dataset=RecDataset.AMAZON,
        force_process=False,
        train_test_split="all",
        split=dataset_split,
    )


def _snapshot_hard_collapse(path: Path) -> dict[int, bool]:
    result: dict[int, bool] = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        epoch = record.get("completed_epoch")
        if epoch is not None:
            result[int(epoch)] = bool(
                record["assignment_diagnostics"]["hard_collapse"]
            )
    return result


def _checkpoint_history(path: Path) -> tuple[int, str]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    history = checkpoint["snapshot_history"]
    return int(history["record_count"]), str(history["rolling_hash"])


def _failure_status_for(error: BaseException) -> FailureStatus:
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return FailureStatus.OOM
    if isinstance(error, FloatingPointError):
        return FailureStatus.NUMERICAL_FAILURE
    if isinstance(error, FileExistsError):
        return FailureStatus.OUTPUT_EXISTS
    if isinstance(error, (ValueError, FileNotFoundError)):
        return FailureStatus.INVARIANT_MISMATCH
    if isinstance(error, KeyboardInterrupt):
        return FailureStatus.INTERRUPTED
    return FailureStatus.RUNTIME_ERROR


def _write_early_terminal_artifacts(
    *,
    error: BaseException,
    save_root: Path,
    snapshot_path: Path,
    summary_path: Path,
    emergency_path: Path,
    run_mode: str,
    local_values: Mapping[str, Any],
    started: float,
    history: SnapshotHistory | None,
    config: Mapping[str, Any] | None,
    complete_hash: str | None,
    invariant_hash: str | None,
    initialization_hash: str | None,
    epoch_rolling_hash: str | None,
) -> None:
    status = _failure_status_for(error)
    message = str(error)
    unavailable_rng_hash = hashlib.sha256(
        b"rng unavailable during diagnostic initialization"
    ).hexdigest()
    write_emergency_failure(
        emergency_path,
        failure_status=status,
        failure_message=message,
        boundary="initialization_transaction",
        completed_epoch=0,
        optimizer_step=0,
        attempted_optimizer_step=0,
        losses=None,
        gradient_stats=None,
        model_hash=None,
        optimizer_state_hash=None,
        rng_hash=unavailable_rng_hash,
        common_initialization_hash=None,
        invariant_config_hash=invariant_hash,
    )
    collapse = classify_collapse({}, failure_status=status)
    summary = {
        "dataset": "amazon-beauty",
        "dataset_split": local_values["dataset_split"],
        "dataset_item_count": local_values["dataset_item_count"],
        "training_item_count": None,
        "model": {
            "name": "RQ-VAE",
            "input_dim": local_values["vae_input_dim"],
            "hidden_dims": list(local_values["vae_hidden_dims"]),
            "embed_dim": local_values["vae_embed_dim"],
            "codebook_size": local_values["vae_codebook_size"],
            "n_layers": local_values["vae_n_layers"],
            "commitment_weight": local_values["commitment_weight"],
            "gumbel_temperature": local_values["gumbel_temperature"],
        },
        "seed": local_values["seed"],
        "config": None if config is None else dict(config),
        "complete_config_hash": complete_hash,
        "invariant_config_hash": invariant_hash,
        "initialization_config_hash": initialization_hash,
        "common_initialization_hash": None,
        "epoch_plan_rolling_hash": epoch_rolling_hash,
        "optimizer": None,
        "completed_epoch": 0,
        "optimizer_step": 0,
        "losses": {},
        "collapse": collapse,
        "failure_status": status.value,
        "failure_message": message,
        "timings": {
            "elapsed_seconds": time.perf_counter() - started,
            "startup_seconds": None,
            "steady_step_seconds": None,
            "full_corpus_snapshot_seconds": None,
            "peak_allocated_vram_bytes": None,
            "peak_reserved_vram_bytes": None,
        },
        "device": {"type": "cuda", "metadata_available": False},
        "external": {
            "url": "https://github.com/EdoardoBotta/RQ-VAE-Recommender",
            "commit": None,
            "branch": None,
            "checkout_path": str(Path(__file__).resolve().parent),
            "implementation": "locally_modified_third_party",
        },
        "artifacts": {
            "output_root": str(save_root),
            "config_path": local_values["config_path"],
            "common_initialization_path": str(
                Path(local_values["common_initialization_path"])
                .expanduser()
                .resolve()
            ),
            "snapshot_jsonl_path": str(snapshot_path),
            "snapshot_record_count": 0 if history is None else history.record_count,
            "snapshot_rolling_hash": None if history is None else history.rolling_hash,
            "checkpoint_path": None,
            "emergency_failure_path": str(emergency_path),
            "summary_path": str(summary_path),
            "log_path": str(save_root / "train.log"),
            "final_checkpoint_path": None,
            "decoder_eligible_symlink": None,
        },
        "evidence_level": evidence_level(run_mode),
        "final_assignment": None,
        "local_promising": None,
        "paper_gate_passed": None,
    }
    _exclusive_json(summary_path, summary)


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _prepare_common_initialization(**kwargs: Any) -> dict[str, Any]:
    validate_diagnostic_run(
        **{key: kwargs[key] for key in inspect.signature(validate_diagnostic_run).parameters}
    )
    if kwargs["run_mode"] != "bounded":
        raise ValueError("common initialization must be prepared from bounded config")
    if not torch.cuda.is_available():
        raise RuntimeError("diagnostic common initialization requires CUDA")
    destination = Path(kwargs["common_initialization_path"]).expanduser().resolve()
    if not str(kwargs["common_initialization_path"]):
        raise ValueError("common_initialization_path must not be empty")
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset = _load_dataset(kwargs["dataset_folder"], kwargs["dataset_split"])
    if len(dataset) != kwargs["dataset_item_count"]:
        raise ValueError("canonical dataset item count mismatch")
    device = torch.device("cuda")
    seed_all(kwargs["seed"])
    rng_before = capture_rng_state()
    model = build_paper_model(
        input_dim=kwargs["vae_input_dim"],
        hidden_dims=list(kwargs["vae_hidden_dims"]),
        embed_dim=kwargs["vae_embed_dim"],
        codebook_size=kwargs["vae_codebook_size"],
        n_layers=kwargs["vae_n_layers"],
        commitment_weight=kwargs["commitment_weight"],
    ).to(device)
    initial_state = clone_cpu_state_dict(model)
    indices = epoch_permutation(len(dataset), kwargs["seed"], 1)[:1024]
    inputs = load_items_by_index(dataset, indices).detach().cpu()
    semantic_ids = model.get_semantic_ids(
        inputs.to(device), gumbel_t=kwargs["gumbel_temperature"]
    ).sem_ids
    used_counts = [
        int(semantic_ids[:, index].unique().numel())
        for index in range(semantic_ids.shape[1])
    ]
    post_state = clone_cpu_state_dict(model)
    training_rng = capture_rng_state()
    config = _training_config(kwargs)
    initialization_payload = initialization_config(config)
    hashes = epoch_plan_hashes(len(dataset), kwargs["seed"], 500)
    rolling_hash = rolling_epoch_plan_hash(len(dataset), kwargs["seed"], 500)
    artifact = save_common_initialization(
        destination,
        initial_model_state=initial_state,
        post_kmeans_model_state=post_state,
        rng_before_model_initialization=rng_before,
        rng_at_training_start=training_rng,
        initial_batch_indices=indices,
        initial_batch_inputs=inputs,
        initial_batch_used_code_counts=used_counts,
        dataset_sha256=kwargs["dataset_sha256"],
        dataset_item_count=len(dataset),
        seed=kwargs["seed"],
        initialization_config_payload=initialization_payload,
        epoch_hashes=hashes,
        epoch_plan_rolling_hash=rolling_hash,
    )
    model.load_state_dict(initial_state, strict=True)
    for layer in model.layers:
        layer.kmeans_initted = False
    loaded = load_common_initialization(
        destination,
        model=model,
        expected_dataset_sha256=kwargs["dataset_sha256"],
        expected_dataset_item_count=len(dataset),
        expected_seed=kwargs["seed"],
        expected_initialization_config_hash=initialization_config_hash(config),
        expected_epoch_plan_rolling_hash=rolling_hash,
    )
    if loaded["common_initialization_hash"] != artifact["common_initialization_hash"]:
        raise ValueError("common initialization reload hash mismatch")
    return {
        "path": str(destination),
        "common_initialization_hash": artifact["common_initialization_hash"],
        "initial_model_hash": artifact["initial_model_hash"],
        "post_kmeans_model_hash": artifact["post_kmeans_model_hash"],
        "epoch_plan_rolling_hash": rolling_hash,
        "used_code_counts": used_counts,
    }


def _resolved_train_kwargs() -> dict[str, Any]:
    signature = inspect.signature(train)
    resolved = {
        name: parameter.default for name, parameter in signature.parameters.items()
    }
    resolved.update(gin.get_bindings(train))
    return resolved


def prepare_common_initialization_from_gin() -> dict[str, Any]:
    return _prepare_common_initialization(**_resolved_train_kwargs())


@gin.configurable(module="train_rqvae_diagnostic")
def train(
    run_mode: Literal["smoke", "bounded"] = "smoke",
    optimizer_name: Literal["adagrad", "adamw"] = "adagrad",
    learning_rate: float = 0.4,
    weight_decay: float = 0.0,
    optimizer_eps: float = 1e-10,
    initial_accumulator_value: float | None = 0.0,
    adam_betas: tuple[float, float] | None = None,
    seed: int = 20260701,
    epochs: int = 2,
    max_items: int | None = 2048,
    batch_size: int = 1024,
    initialization_batch_size: int = 1024,
    num_workers: int = 0,
    drop_last: bool = False,
    amp: bool = False,
    dataset_folder: str = "dataset/amazon-p5-st5",
    dataset_split: str = "beauty",
    dataset_sha256: str = "2a4be981c007724a1a27fa41155c3857802fb370387968499256b58d9909ee21",
    dataset_item_count: int = 12101,
    common_initialization_path: str = "",
    save_dir_root: str = "",
    config_path: str = "",
    resume_checkpoint: str | None = None,
    checkpoint_epochs: tuple[int, ...] = (2,),
    snapshot_optimizer_steps: tuple[int, ...] = _STEP_SCHEDULE,
    snapshot_epochs: tuple[int, ...] = _EPOCH_SCHEDULE,
    snapshot_batch_size: int = 1024,
    run_stop_utc: str = "2026-07-14T19:40:00Z",
    vae_input_dim: int = 768,
    vae_hidden_dims: tuple[int, ...] = (512, 256, 128),
    vae_embed_dim: int = 32,
    vae_codebook_size: int = 256,
    vae_n_layers: int = 3,
    commitment_weight: float = 0.25,
    gumbel_temperature: float = 0.2,
) -> dict[str, Any]:
    local_values = locals().copy()
    validate_diagnostic_run(
        **{key: local_values[key] for key in inspect.signature(validate_diagnostic_run).parameters}
    )
    if not common_initialization_path or not save_dir_root:
        raise ValueError("common initialization and save root paths must not be empty")
    deadline = parse_utc_deadline(run_stop_utc)
    if not torch.cuda.is_available():
        raise RuntimeError("diagnostic RQ-VAE training requires CUDA")
    save_root = Path(save_dir_root).expanduser().resolve()
    snapshot_path = save_root / "snapshots.jsonl"
    summary_path = save_root / "summary.json"
    emergency_path = save_root / "emergency_failure.json"
    if resume_checkpoint is None:
        save_root.mkdir(parents=True, exist_ok=False)
    elif not save_root.is_dir():
        raise FileNotFoundError("resume output root does not exist")

    started = time.perf_counter()
    startup_finished: float | None = None
    snapshot_seconds: list[float] = []
    step_seconds: list[float] = []
    device = torch.device("cuda")
    history: SnapshotHistory | None = None
    config: dict[str, Any] | None = None
    complete_hash: str | None = None
    invariant_hash: str | None = None
    initialization_hash: str | None = None
    epoch_hashes: list[str] | None = None
    epoch_rolling_hash: str | None = None
    common_hash: str | None = None
    training_item_count: int | None = None
    model: torch.nn.Module | None = None
    optimizer: torch.optim.Optimizer | None = None
    spec: OptimizerSpec | None = None
    completed_epoch = 0
    optimizer_step = 0
    last_losses: dict[str, float] = {}
    last_gradient: Mapping[str, Any] | None = None
    last_delta: Mapping[str, Any] | None = None
    last_snapshot: dict[str, Any] | None = None
    last_checkpoint: Path | None = None
    validated_resume_checkpoint: Path | None = None
    last_safe_rng_hash: str | None = None
    failure_status = FailureStatus.COMPLETE
    failure_message: str | None = None
    deadline_pending = False

    try:
        if resume_checkpoint is None:
            history = SnapshotHistory.create(snapshot_path)
        else:
            if summary_path.exists() or emergency_path.exists():
                raise FileExistsError(
                    "resume output root already contains terminal artifacts"
                )
            count, digest = _checkpoint_history(Path(resume_checkpoint))
            history = SnapshotHistory.resume(
                snapshot_path,
                expected_record_count=count,
                expected_rolling_hash=digest,
            )
        config = _training_config(local_values)
        complete_hash = complete_config_hash(config)
        invariant_hash = invariant_config_hash(config)
        initialization_hash = initialization_config_hash(config)
        epoch_hashes = epoch_plan_hashes(dataset_item_count, seed, 500)
        epoch_rolling_hash = rolling_epoch_plan_hash(dataset_item_count, seed, 500)
        torch.cuda.reset_peak_memory_stats()
    except BaseException as error:
        if isinstance(error, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
        _write_early_terminal_artifacts(
            error=error,
            save_root=save_root,
            snapshot_path=snapshot_path,
            summary_path=summary_path,
            emergency_path=emergency_path,
            run_mode=run_mode,
            local_values=local_values,
            started=started,
            history=history,
            config=config,
            complete_hash=complete_hash,
            invariant_hash=invariant_hash,
            initialization_hash=initialization_hash,
            epoch_rolling_hash=epoch_rolling_hash,
        )
        raise

    def write_current_emergency(
        *,
        status: FailureStatus,
        message: str,
        boundary: str,
        attempted_optimizer_step: int,
        losses: Mapping[str, float] | None,
        gradient_stats: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        return write_emergency_failure(
            emergency_path,
            failure_status=status,
            failure_message=message,
            boundary=boundary,
            completed_epoch=completed_epoch,
            optimizer_step=optimizer_step,
            attempted_optimizer_step=attempted_optimizer_step,
            losses=losses,
            gradient_stats=gradient_stats,
            model_hash=(
                hash_state_dict(clone_cpu_state_dict(model)) if model is not None else None
            ),
            optimizer_state_hash=(
                hash_nested_state(optimizer.state_dict())
                if optimizer is not None else None
            ),
            rng_hash=hash_rng_state(capture_rng_state()),
            common_initialization_hash=common_hash,
            invariant_config_hash=invariant_hash if invariant_hash else None,
        )

    def append_snapshot(triggers: Sequence[str], epoch: int | None) -> None:
        nonlocal last_snapshot, deadline_pending
        _synchronize()
        snapshot_started = time.perf_counter()
        snapshot = capture_read_only_corpus_snapshot(
            model=model,
            optimizer=optimizer,
            canonical_loader=canonical_loader,
            diagnostic_loader_generator=diagnostic_generator,
            post_kmeans_state=common_artifact["post_kmeans_model_state"],
            optimizer_step=optimizer_step,
            completed_epoch=epoch,
            triggers=triggers,
        )
        _synchronize()
        snapshot_seconds.append(time.perf_counter() - snapshot_started)
        snapshot["gradient_stats"] = last_gradient
        snapshot["parameter_delta_stats"] = last_delta
        snapshot["dataset_sha256"] = dataset_sha256
        snapshot["complete_config_hash"] = complete_hash
        snapshot["invariant_config_hash"] = invariant_hash
        snapshot["initialization_config_hash"] = initialization_hash
        snapshot["common_initialization_hash"] = common_hash
        snapshot["epoch_plan_rolling_hash"] = epoch_rolling_hash
        snapshot["optimizer_treatment_hash"] = optimizer_treatment_hash(spec)
        snapshot["optimizer_metadata"] = optimizer_metadata(optimizer)
        snapshot["optimizer_metadata_hash"] = optimizer_metadata_hash(optimizer)
        history.append(snapshot)
        last_snapshot = snapshot
        if datetime.now(timezone.utc) >= deadline:
            deadline_pending = True

    def save_checkpoint(epoch: int) -> Path:
        path = checkpoint_path_for(
            save_root, completed_epoch=epoch, optimizer_step=optimizer_step
        )
        if path.exists():
            return reusable_checkpoint_at_position(
                path,
                validated_resume_checkpoint=validated_resume_checkpoint,
                completed_epoch=epoch,
                optimizer_step=optimizer_step,
                current_process_checkpoint=last_checkpoint,
            )
        save_diagnostic_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            optimizer_spec=spec,
            completed_epoch=epoch,
            optimizer_step=optimizer_step,
            rng_state=capture_rng_state(),
            common_initialization_hash=common_hash,
            invariant_config_hash=invariant_hash,
            complete_config_hash=complete_hash,
            dataset_sha256=dataset_sha256,
            seed=seed,
            epoch_hashes=epoch_hashes,
            epoch_plan_rolling_hash=epoch_rolling_hash,
            snapshot_jsonl_path=snapshot_path,
            snapshot_record_count=history.record_count,
            snapshot_rolling_hash=history.rolling_hash,
        )
        return path

    def make_summary(status: FailureStatus, message: str | None) -> dict[str, Any]:
        collapse_by_epoch = _snapshot_hard_collapse(snapshot_path)
        collapse = classify_collapse(collapse_by_epoch, failure_status=status)
        assignment = (
            None if last_snapshot is None else last_snapshot["assignment_diagnostics"]
        )
        elapsed = time.perf_counter() - started
        return {
            "dataset": "amazon-beauty",
            "dataset_split": dataset_split,
            "dataset_item_count": dataset_item_count,
            "training_item_count": training_item_count,
            "model": {
                "name": "RQ-VAE",
                "input_dim": vae_input_dim,
                "hidden_dims": list(vae_hidden_dims),
                "embed_dim": vae_embed_dim,
                "codebook_size": vae_codebook_size,
                "n_layers": vae_n_layers,
                "commitment_weight": commitment_weight,
                "gumbel_temperature": gumbel_temperature,
            },
            "seed": seed,
            "config": config,
            "complete_config_hash": complete_hash,
            "invariant_config_hash": invariant_hash,
            "initialization_config_hash": initialization_hash,
            "common_initialization_hash": common_hash,
            "epoch_plan_rolling_hash": epoch_rolling_hash,
            "optimizer": None if optimizer is None else {
                "treatment": optimizer_treatment_payload(spec),
                "treatment_hash": optimizer_treatment_hash(spec),
                "metadata": optimizer_metadata(optimizer),
                "metadata_hash": optimizer_metadata_hash(optimizer),
            },
            "completed_epoch": completed_epoch,
            "optimizer_step": optimizer_step,
            "losses": _sanitize_failure_json(last_losses),
            "collapse": collapse,
            "failure_status": status.value,
            "failure_message": message,
            "timings": {
                "elapsed_seconds": elapsed,
                "startup_seconds": startup_finished,
                "steady_step_seconds": (
                    sum(step_seconds[1:4]) / len(step_seconds[1:4])
                    if step_seconds[1:4] else None
                ),
                "full_corpus_snapshot_seconds": (
                    sum(snapshot_seconds) / len(snapshot_seconds)
                    if snapshot_seconds else None
                ),
                "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(),
            },
            "device": _device_metadata(device),
            "external": _external_metadata(),
            "artifacts": {
                "output_root": str(save_root),
                "config_path": config_path,
                "common_initialization_path": str(Path(common_initialization_path).expanduser().resolve()),
                "snapshot_jsonl_path": str(snapshot_path),
                "snapshot_record_count": history.record_count,
                "snapshot_rolling_hash": history.rolling_hash,
                "checkpoint_path": None if last_checkpoint is None else str(last_checkpoint),
                "emergency_failure_path": str(emergency_path) if emergency_path.exists() else None,
                "summary_path": str(summary_path),
                "log_path": str(save_root / "train.log"),
                "final_checkpoint_path": None,
                "decoder_eligible_symlink": None,
            },
            "evidence_level": evidence_level(run_mode),
            "final_assignment": assignment,
            "local_promising": None if assignment is None else assignment["local_promising"],
            "paper_gate_passed": None if assignment is None else assignment["paper_gate_passed"],
        }

    try:
        canonical_dataset = _load_dataset(dataset_folder, dataset_split)
        if len(canonical_dataset) != dataset_item_count:
            raise ValueError("canonical dataset item count mismatch")
        training_dataset: Dataset = canonical_dataset
        if max_items is not None:
            training_dataset = Subset(canonical_dataset, range(max_items))
        training_item_count = len(training_dataset)
        seed_all(seed)
        last_safe_rng_hash = hash_rng_state(capture_rng_state())
        model = build_paper_model(
            input_dim=vae_input_dim,
            hidden_dims=list(vae_hidden_dims),
            embed_dim=vae_embed_dim,
            codebook_size=vae_codebook_size,
            n_layers=vae_n_layers,
            commitment_weight=commitment_weight,
        ).to(device)
        common_artifact = load_common_initialization(
            Path(common_initialization_path).expanduser().resolve(),
            model=model,
            expected_dataset_sha256=dataset_sha256,
            expected_dataset_item_count=dataset_item_count,
            expected_seed=seed,
            expected_initialization_config_hash=initialization_hash,
            expected_epoch_plan_rolling_hash=epoch_rolling_hash,
        )
        common_hash = common_artifact["common_initialization_hash"]
        spec = OptimizerSpec.from_config_fields(
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer_eps=optimizer_eps,
            initial_accumulator_value=initial_accumulator_value,
            adam_betas=adam_betas,
        )
        optimizer = build_diagnostic_optimizer(model, spec)
        groups = parameter_groups(model)
        diagnostic_generator = torch.Generator(device="cpu").manual_seed(seed)
        canonical_loader = DataLoader(
            canonical_dataset,
            batch_size=snapshot_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            generator=diagnostic_generator,
        )
        start_epoch = 1
        if resume_checkpoint is not None:
            validated_resume_checkpoint = Path(resume_checkpoint).expanduser().resolve()
            resume = load_diagnostic_checkpoint(
                validated_resume_checkpoint,
                model=model,
                optimizer=optimizer,
                optimizer_spec=spec,
                expected_common_initialization_hash=common_hash,
                expected_invariant_config_hash=invariant_hash,
                expected_dataset_sha256=dataset_sha256,
                expected_seed=seed,
                expected_epoch_plan_rolling_hash=epoch_rolling_hash,
                expected_snapshot_jsonl_path=snapshot_path,
            )
            start_epoch = resume["start_epoch"]
            optimizer_step = resume["optimizer_step"]
            completed_epoch = start_epoch - 1
        else:
            append_snapshot(["optimizer_step:0"], None)
        startup_finished = time.perf_counter() - started

        for epoch in range(start_epoch, epochs + 1):
            if datetime.now(timezone.utc) >= deadline:
                failure_status = FailureStatus.DEADLINE_STOP
                failure_message = "run deadline reached before epoch start"
                if completed_epoch > 0:
                    last_checkpoint = save_checkpoint(completed_epoch)
                summary = make_summary(failure_status, failure_message)
                _exclusive_json(summary_path, summary)
                return summary
            permutation = epoch_permutation(len(training_dataset), seed, epoch)
            loader = DataLoader(
                Subset(training_dataset, permutation.tolist()),
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=0,
            )
            epoch_totals = {"loss": 0.0, "reconstruction_loss": 0.0, "rqvae_loss": 0.0}
            batch_count = 0
            for batch_index, batch in enumerate(loader, start=1):
                _synchronize()
                step_started = time.perf_counter()
                device_batch = batch_to(batch, device)
                optimizer.zero_grad(set_to_none=True)
                last_safe_rng_hash = hash_rng_state(capture_rng_state())
                output = model(device_batch, gumbel_t=gumbel_temperature)
                step_losses = {
                    "loss": float(output.loss.detach().cpu().item()),
                    "reconstruction_loss": float(output.reconstruction_loss.detach().cpu().item()),
                    "rqvae_loss": float(output.rqvae_loss.detach().cpu().item()),
                }
                if not torch.isfinite(output.loss):
                    write_current_emergency(
                        status=FailureStatus.NUMERICAL_FAILURE,
                        message="nonfinite loss before backward",
                        boundary="pre_backward_pre_update",
                        attempted_optimizer_step=optimizer_step + 1,
                        losses=step_losses,
                        gradient_stats=None,
                    )
                    raise FloatingPointError("nonfinite loss before backward")
                output.loss.backward()
                gradient = gradient_group_stats(groups)
                if any(
                    values["nonfinite_fraction"] is not None
                    and values["nonfinite_fraction"] > 0
                    for values in gradient.values()
                ):
                    write_current_emergency(
                        status=FailureStatus.NUMERICAL_FAILURE,
                        message="nonfinite gradient before optimizer update",
                        boundary="post_backward_pre_update",
                        attempted_optimizer_step=optimizer_step + 1,
                        losses=step_losses,
                        gradient_stats=gradient,
                    )
                    raise FloatingPointError("nonfinite gradient before optimizer update")
                before = clone_group_parameters(groups)
                optimizer.step()
                optimizer_step += 1
                delta = parameter_delta_stats(groups, before)
                _synchronize()
                step_seconds.append(time.perf_counter() - step_started)
                last_gradient, last_delta = gradient, delta
                batch_count += 1
                for key, value in step_losses.items():
                    epoch_totals[key] += value
                is_epoch_end = batch_index == len(loader)
                triggers = scheduled_snapshot_triggers(
                    optimizer_step=optimizer_step,
                    completed_epoch=epoch if is_epoch_end else None,
                    step_schedule=frozenset(snapshot_optimizer_steps),
                    epoch_schedule=frozenset(snapshot_epochs),
                )
                if triggers:
                    append_snapshot(triggers, epoch if is_epoch_end else None)
            completed_epoch = epoch
            last_losses = {key: value / batch_count for key, value in epoch_totals.items()}
            if epoch in checkpoint_epochs or deadline_pending:
                last_checkpoint = save_checkpoint(epoch)
            if deadline_pending:
                failure_status = FailureStatus.DEADLINE_STOP
                failure_message = "run deadline observed after intra-epoch snapshot"
                summary = make_summary(failure_status, failure_message)
                _exclusive_json(summary_path, summary)
                return summary
        failure_status = FailureStatus.COMPLETE
        summary = make_summary(failure_status, None)
        _exclusive_json(summary_path, summary)
        return summary
    except BaseException as error:
        if isinstance(error, torch.cuda.OutOfMemoryError):
            failure_status = FailureStatus.OOM
            torch.cuda.empty_cache()
        elif isinstance(error, FloatingPointError):
            failure_status = FailureStatus.NUMERICAL_FAILURE
        elif isinstance(error, (ValueError, FileNotFoundError)):
            failure_status = FailureStatus.INVARIANT_MISMATCH
        elif isinstance(error, KeyboardInterrupt):
            failure_status = FailureStatus.INTERRUPTED
        elif isinstance(error, FileExistsError):
            failure_status = FailureStatus.OUTPUT_EXISTS
        else:
            failure_status = FailureStatus.RUNTIME_ERROR
        failure_message = str(error)
        if not emergency_path.exists():
            try:
                if failure_status is FailureStatus.OOM:
                    if last_safe_rng_hash is None:
                        raise RuntimeError("no cached pre-boundary RNG hash for OOM")
                    write_oom_emergency_failure(
                        emergency_path,
                        failure_message=failure_message,
                        completed_epoch=completed_epoch,
                        optimizer_step=optimizer_step,
                        attempted_optimizer_step=optimizer_step + 1,
                        losses=last_losses or None,
                        cached_rng_hash=last_safe_rng_hash,
                        common_initialization_hash=common_hash,
                        invariant_config_hash=invariant_hash,
                    )
                else:
                    write_current_emergency(
                        status=failure_status,
                        message=failure_message,
                        boundary="exception",
                        attempted_optimizer_step=optimizer_step + 1,
                        losses=last_losses or None,
                        gradient_stats=last_gradient,
                    )
            except BaseException:
                pass
        if not summary_path.exists():
            try:
                _exclusive_json(summary_path, make_summary(failure_status, failure_message))
            except BaseException:
                pass
        raise


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    parser.add_argument("--prepare-common-initialization", action="store_true")
    parser.add_argument("--resume_checkpoint", default=None)
    args = parser.parse_args(argv)
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], args.config_path]
        parse_config()
    finally:
        sys.argv = original_argv
    if args.prepare_common_initialization:
        return prepare_common_initialization_from_gin()
    if args.resume_checkpoint is not None:
        gin.bind_parameter("train.resume_checkpoint", args.resume_checkpoint)
    return train()


if __name__ == "__main__":
    main()
