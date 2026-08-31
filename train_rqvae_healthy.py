"""Bounded, isolated healthy-RQ-VAE probe for the TIGER audit lane.

This deliberately does not alter ``train_rqvae.py``.  It records the short-run
anti-collapse diagnostics needed before any long Beauty RQ-VAE campaign.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import gin
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from data.processed import ItemData
from data.processed import RecDataset
from data.utils import batch_to
from modules.quantize import QuantizeForwardMode
from modules.rqvae import RqVae
from modules.rqvae_healthy import EmaCodebook
from modules.rqvae_healthy import apply_kaiming_relu_initialization
from modules.rqvae_healthy import reset_dead_codes
from modules.rqvae_healthy import summarize_rqvae_health
from modules.tiger_policy import validate_non_placeholder_tiger_content
from modules.utils import parse_config


CHECKPOINT_SCHEMA_VERSION = 1
RUN_METADATA_SCHEMA_VERSION = 1


def _run_git(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _atomic_json_dump(path: Path, value: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_path, path)


def _atomic_torch_save(path: Path, value: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary_path)
    os.replace(temporary_path, path)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def historical_epoch_permutation(item_count: int, seed: int, epoch: int) -> Tensor:
    """Reproduce the historical fix-matrix epoch permutation contract."""

    if item_count <= 0 or epoch <= 0:
        raise ValueError("item_count and epoch must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed + epoch - 1)
    return torch.randperm(item_count, generator=generator, device="cpu")


def _make_model(*, codebook_normalize: bool) -> RqVae:
    return RqVae(
        input_dim=768,
        embed_dim=32,
        hidden_dims=[512, 256, 128],
        codebook_size=256,
        codebook_kmeans_init=True,
        codebook_normalize=codebook_normalize,
        codebook_sim_vq=False,
        codebook_mode=QuantizeForwardMode.STE,
        n_layers=3,
        commitment_weight=0.25,
        n_cat_features=0,
    )


def eager_rqvae_loss(
    model: RqVae, x: Tensor, *, gumbel_temperature: float
) -> tuple[Tensor, Tensor, Tensor, list[Tensor], list[Tensor]]:
    """Run the current RQ-VAE math eagerly and expose layer-aligned EMA inputs."""

    quantized = model.get_semantic_ids(x, gumbel_t=gumbel_temperature)
    x_hat = model.decode(quantized.embeddings.sum(axis=-1))
    reconstruction_loss = model.reconstruction_loss(x_hat, x).mean()
    rqvae_loss = quantized.quantize_loss.mean()
    loss = reconstruction_loss + rqvae_loss

    # ``get_semantic_ids`` stacks a list of [batch, dim] tensors through
    # einops, yielding residuals [batch, dim, layer] and IDs [batch, layer].
    # EMA/reset therefore consume the final (quantizer-layer) axis.
    residuals = list(quantized.residuals.unbind(dim=-1))
    semantic_ids = list(quantized.sem_ids.unbind(dim=-1))
    if len(residuals) != len(model.layers) or len(semantic_ids) != len(model.layers):
        raise RuntimeError("RQ-VAE output does not match its quantizer layer count")
    return loss, reconstruction_loss, rqvae_loss, residuals, semantic_ids


@torch.no_grad()
def _corpus_assignment_counts(
    model: RqVae, dataset: ItemData, *, device: torch.device
) -> list[Tensor]:
    """Count initialized corpus assignments once for an EMA bootstrap."""

    was_training = model.training
    model.eval()
    counts = [
        torch.zeros(layer.n_embed, dtype=torch.float32, device=device)
        for layer in model.layers
    ]
    loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0)
    for batch in loader:
        device_batch = batch_to(batch, device)
        semantic_ids = model.get_semantic_ids(device_batch.x, gumbel_t=0.2).sem_ids
        for index, ids in enumerate(semantic_ids.unbind(dim=-1)):
            counts[index].add_(torch.bincount(ids, minlength=model.layers[index].n_embed))
    model.train(was_training)
    return counts


@torch.no_grad()
def _corpus_snapshot(
    model: RqVae,
    dataset: ItemData,
    *,
    device: torch.device,
    codebook_size: int,
    suffix_capacity: int,
    minimum_usage: float,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0)
    semantic_ids = []
    encoder_norms = []
    for batch in loader:
        device_batch = batch_to(batch, device)
        encoder_norms.append(model.encode(device_batch.x).norm(dim=1).cpu())
        semantic_ids.append(
            model.get_semantic_ids(device_batch.x, gumbel_t=0.2).sem_ids.cpu()
        )
    model.train(was_training)
    if not semantic_ids:
        raise RuntimeError("cannot compute a corpus health snapshot for an empty dataset")
    health = summarize_rqvae_health(
        torch.cat(semantic_ids, dim=0),
        codebook_size=codebook_size,
        suffix_capacity=suffix_capacity,
        minimum_usage=minimum_usage,
    )
    encoder_norm = torch.cat(encoder_norms)
    health.update(
        {
            "encoder_norm_p50": float(encoder_norm.median().item()),
            "encoder_norm_p95": float(torch.quantile(encoder_norm, 0.95).item()),
        }
    )
    return health


def _gradient_norm(model: RqVae, *, include_codebooks: bool) -> float:
    squared_sum = 0.0
    codebook_parameters = {id(layer.embedding.weight) for layer in model.layers}
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        if not include_codebooks and id(parameter) in codebook_parameters:
            continue
        squared_sum += float(parameter.grad.detach().pow(2).sum().item())
    return squared_sum**0.5


def _parameter_norms(model: RqVae) -> dict[str, float]:
    return {
        "encoder": float(
            sum(parameter.detach().pow(2).sum().item() for parameter in model.encoder.parameters())
            ** 0.5
        ),
        "decoder": float(
            sum(parameter.detach().pow(2).sum().item() for parameter in model.decoder.parameters())
            ** 0.5
        ),
        **{
            f"codebook_{index}": float(layer.embedding.weight.detach().norm().item())
            for index, layer in enumerate(model.layers)
        },
    }


def _subtract_norms(after: dict[str, float], before: dict[str, float]) -> dict[str, float]:
    return {key: after[key] - before[key] for key in before}


def _checkpoint_payload(
    *,
    model: RqVae,
    optimizer: torch.optim.Optimizer,
    ema: EmaCodebook | None,
    optimizer_step: int,
    config: dict[str, Any],
    purpose: str,
    selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "purpose": purpose,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "optimizer_step": optimizer_step,
        "config": config,
        "selection": selection,
        "rng_state": _rng_state(),
    }


def _run_metadata(
    *,
    config: dict[str, Any],
    dataset_folder: str,
    dataset_split: str,
    dataset_item_count: int,
    model: RqVae,
) -> dict[str, Any]:
    return {
        "schema_version": RUN_METADATA_SCHEMA_VERSION,
        "study": "TIGER healthy-RQ-VAE bounded stability probe",
        "evidence_level": "healthy_local_improvement_candidate",
        "paper_fidelity_claim": False,
        "default_train_rqvae_modified": False,
        "dataset": {
            "name": Path(dataset_folder).name,
            "root": dataset_folder,
            "split": dataset_split,
            "item_count": dataset_item_count,
        },
        "model": {
            "input_dim": model.input_dim,
            "embed_dim": model.embed_dim,
            "hidden_dims": model.hidden_dims,
            "codebook_size": model.codebook_size,
            "rqvae_n_layers": model.n_layers,
            "full_id_width": model.n_layers + 1,
            "suffix_semantics": "post_training_deterministic_collision_rank",
        },
        "config": config,
        "ema_initialization": (
            "corpus_assignment_mass"
            if config["use_ema_codebook"] and config["ema_bootstrap_from_corpus"]
            else config.get("ema_initialization", "unit_pseudocount")
            if config["use_ema_codebook"]
            else "not_enabled"
        ),
        "code_revision": _run_git(["rev-parse", "HEAD"]),
        "working_tree_dirty": bool(_run_git(["status", "--porcelain"])),
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda},
    }


@gin.configurable
def train(
    *,
    arm_name: str = "E5_combined",
    output_dir: str = "out/rqvae/tiger_beauty_healthy_E5_probe_20260830",
    dataset_folder: str = "dataset/amazon-p5",
    dataset_split: str = "beauty",
    expected_num_items: int = 12101,
    seed: int = 20260830,
    epochs: int = 1,
    max_optimizer_steps: int = 1,
    batch_size: int = 1024,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    warmup_steps: int = 50,
    use_ema_codebook: bool = True,
    ema_decay: float = 0.99,
    ema_initialization: str = "unit_pseudocount",
    ema_bootstrap_from_corpus: bool = False,
    dead_code_reset_every: int = 1,
    use_kaiming_relu_initialization: bool = True,
    codebook_normalize: bool = False,
    gumbel_temperature: float = 0.2,
    suffix_capacity: int = 256,
    minimum_usage: float = 0.80,
    snapshot_every_steps: int = 1,
    stop_on_failed_health_gate: bool = True,
    checkpoint_every_epochs: int = 0,
    epoch_ordering: str = "loader_shuffle",
    reset_donor_sampling: str = "with_replacement",
    reset_rng_policy: str = "isolated",
) -> dict[str, Any]:
    """Run a fresh bounded GPU probe; refusing existing output prevents overwrite."""

    if not torch.cuda.is_available():
        raise RuntimeError("the healthy-RQ-VAE probe requires an available CUDA GPU")
    if epochs <= 0 or max_optimizer_steps <= 0 or batch_size <= 0:
        raise ValueError("epochs, max_optimizer_steps, and batch_size must be positive")
    if expected_num_items <= 0 or suffix_capacity != 256:
        raise ValueError("the paper-cardinality probe requires positive items and suffix_capacity=256")
    if dead_code_reset_every < 0 or warmup_steps < 0:
        raise ValueError("reset and warmup intervals must be non-negative")
    if ema_initialization not in ("unit_pseudocount", "zero_mass"):
        raise ValueError("unsupported ema_initialization")
    if snapshot_every_steps <= 0:
        raise ValueError("snapshot_every_steps must be positive")
    if checkpoint_every_epochs < 0:
        raise ValueError("checkpoint_every_epochs must be non-negative")
    if epoch_ordering not in ("loader_shuffle", "historical_epoch_permutation"):
        raise ValueError("unsupported epoch_ordering")
    if reset_donor_sampling not in ("with_replacement", "without_replacement"):
        raise ValueError("unsupported reset_donor_sampling")
    if reset_rng_policy not in ("isolated", "global"):
        raise ValueError("unsupported reset_rng_policy")

    save_root = Path(output_dir)
    if save_root.exists():
        raise FileExistsError(f"refusing to overwrite existing probe output: {save_root}")
    save_root.mkdir(parents=True, exist_ok=False)

    config = {
        "arm_name": arm_name,
        "dataset_folder": dataset_folder,
        "dataset_split": dataset_split,
        "expected_num_items": expected_num_items,
        "seed": seed,
        "epochs": epochs,
        "max_optimizer_steps": max_optimizer_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
        "use_ema_codebook": use_ema_codebook,
        "ema_decay": ema_decay,
        "ema_initialization": ema_initialization,
        "ema_bootstrap_from_corpus": ema_bootstrap_from_corpus,
        "dead_code_reset_every": dead_code_reset_every,
        "use_kaiming_relu_initialization": use_kaiming_relu_initialization,
        "codebook_normalize": codebook_normalize,
        "gumbel_temperature": gumbel_temperature,
        "suffix_capacity": suffix_capacity,
        "minimum_usage": minimum_usage,
        "snapshot_every_steps": snapshot_every_steps,
        "stop_on_failed_health_gate": stop_on_failed_health_gate,
        "checkpoint_every_epochs": checkpoint_every_epochs,
        "epoch_ordering": epoch_ordering,
        "reset_donor_sampling": reset_donor_sampling,
        "reset_rng_policy": reset_rng_policy,
    }
    try:
        (save_root / "resolved_config.gin").write_text(
            gin.operative_config_str(), encoding="utf-8"
        )
    except ValueError:
        (save_root / "resolved_config.gin").write_text("", encoding="utf-8")
    _atomic_json_dump(save_root / "run_metadata.json", _run_metadata(
        config=config,
        dataset_folder=dataset_folder,
        dataset_split=dataset_split,
        dataset_item_count=expected_num_items,
        model=_make_model(codebook_normalize=codebook_normalize),
    ))

    _seed_everything(seed)
    device = torch.device("cuda")
    dataset = ItemData(
        root=dataset_folder,
        dataset=RecDataset.AMAZON,
        force_process=False,
        train_test_split="all",
        split=dataset_split,
    )
    if len(dataset) != expected_num_items:
        raise ValueError(
            f"expected {expected_num_items} items for {dataset_split}, found {len(dataset)}"
        )
    validate_non_placeholder_tiger_content(dataset.item_text)
    loader: DataLoader | None = None
    if epoch_ordering == "loader_shuffle":
        loader_generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            generator=loader_generator,
        )
        initial_batch = batch_to(next(iter(loader)), device)
    else:
        initial_indices = historical_epoch_permutation(len(dataset), seed, 1)[:batch_size]
        initial_batch = batch_to(dataset[initial_indices], device)

    model = _make_model(codebook_normalize=codebook_normalize).to(device)
    if use_kaiming_relu_initialization:
        initialized_linear_layers = apply_kaiming_relu_initialization(model)
    else:
        initialized_linear_layers = 0
    model.eval()
    with torch.no_grad():
        model.get_semantic_ids(initial_batch.x, gumbel_t=gumbel_temperature)
    if not all(layer.kmeans_initted for layer in model.layers):
        raise RuntimeError("codebook k-means initialization did not complete")
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    ema = (
        EmaCodebook(
            model.layers, ema_decay, initialization=ema_initialization
        )
        if use_ema_codebook
        else None
    )
    ema_bootstrap_summary: dict[str, list[float]] | None = None
    if ema is not None and ema_bootstrap_from_corpus:
        assignment_counts = _corpus_assignment_counts(model, dataset, device=device)
        ema.bootstrap_from_assignment_counts(model.layers, assignment_counts)
        ema_bootstrap_summary = {
            "min": [float(count.min().item()) for count in assignment_counts],
            "median": [float(count.median().item()) for count in assignment_counts],
            "max": [float(count.max().item()) for count in assignment_counts],
            "total": [float(count.sum().item()) for count in assignment_counts],
        }
        _atomic_json_dump(save_root / "ema_bootstrap.json", ema_bootstrap_summary)
    reset_generator = (
        torch.Generator(device=device).manual_seed(seed + 1)
        if reset_rng_policy == "isolated"
        else None
    )
    snapshots: list[dict[str, Any]] = []

    def take_snapshot(trigger: str, optimizer_step: int) -> dict[str, Any]:
        snapshot = _corpus_snapshot(
            model,
            dataset,
            device=device,
            codebook_size=model.codebook_size,
            suffix_capacity=suffix_capacity,
            minimum_usage=minimum_usage,
        )
        snapshot.update({"trigger": trigger, "optimizer_step": optimizer_step})
        snapshots.append(snapshot)
        _atomic_json_dump(save_root / "snapshots.json", {"snapshots": snapshots})
        return snapshot

    started = time.perf_counter()
    initial_snapshot = take_snapshot("post_kmeans", optimizer_step=0)
    latest_snapshot = initial_snapshot
    optimizer_step = 0
    first_step: dict[str, Any] | None = None
    epoch_losses: list[dict[str, float | int]] = []
    best_total_loss: dict[str, float | int] | None = None
    reset_totals = [0 for _ in model.layers]
    losses_finite = True
    health_gate_failed_step: int | None = None

    for epoch in range(1, epochs + 1):
        totals = {"loss": 0.0, "reconstruction_loss": 0.0, "rqvae_loss": 0.0}
        batches = 0
        if epoch_ordering == "historical_epoch_permutation":
            permutation = historical_epoch_permutation(len(dataset), seed, epoch)
            epoch_loader = DataLoader(
                Subset(dataset, permutation.tolist()),
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
            )
        else:
            if loader is None:
                raise RuntimeError("loader_shuffle requires a configured DataLoader")
            epoch_loader = loader
        for batch in epoch_loader:
            if optimizer_step >= max_optimizer_steps:
                break
            if warmup_steps:
                current_lr = learning_rate * min(1.0, (optimizer_step + 1) / warmup_steps)
                for group in optimizer.param_groups:
                    group["lr"] = current_lr
            device_batch = batch_to(batch, device)
            optimizer.zero_grad(set_to_none=True)
            parameter_norms_before = _parameter_norms(model)
            loss, reconstruction_loss, rqvae_loss, residuals, semantic_ids = eager_rqvae_loss(
                model, device_batch.x, gumbel_temperature=gumbel_temperature
            )
            finite = bool(
                torch.isfinite(loss).item()
                and torch.isfinite(reconstruction_loss).item()
                and torch.isfinite(rqvae_loss).item()
            )
            losses_finite &= finite
            if not finite:
                _atomic_json_dump(
                    save_root / "failure.json",
                    {"reason": "non_finite_loss", "optimizer_step": optimizer_step},
                )
                raise FloatingPointError("healthy-RQ-VAE probe encountered a non-finite loss")
            loss.backward()
            gradient_norms = {
                "all": _gradient_norm(model, include_codebooks=True),
                "encoder_decoder": _gradient_norm(model, include_codebooks=False),
            }
            if ema is not None:
                for layer in model.layers:
                    layer.embedding.weight.grad = None
            optimizer.step()
            reset_counts = [0 for _ in model.layers]
            if ema is not None:
                ema.update(model.layers, residuals, semantic_ids)
            if dead_code_reset_every and (optimizer_step + 1) % dead_code_reset_every == 0:
                reset_counts = reset_dead_codes(
                    model.layers,
                    residuals,
                    semantic_ids,
                    generator=reset_generator,
                )
                reset_totals = [
                    total + count for total, count in zip(reset_totals, reset_counts, strict=True)
                ]
            parameter_norms_after = _parameter_norms(model)
            optimizer_step += 1
            batches += 1
            totals["loss"] += float(loss.item())
            totals["reconstruction_loss"] += float(reconstruction_loss.item())
            totals["rqvae_loss"] += float(rqvae_loss.item())
            if first_step is None:
                first_step = {
                    "optimizer_step": optimizer_step,
                    "loss": float(loss.item()),
                    "reconstruction_loss": float(reconstruction_loss.item()),
                    "rqvae_loss": float(rqvae_loss.item()),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "finite": finite,
                    "gradient_norms": gradient_norms,
                    "parameter_norms_before": parameter_norms_before,
                    "parameter_norm_deltas": _subtract_norms(
                        parameter_norms_after, parameter_norms_before
                    ),
                    "dead_code_resets": reset_counts,
                }
                _atomic_json_dump(save_root / "first_step.json", first_step)
            if optimizer_step % snapshot_every_steps == 0:
                current_snapshot = take_snapshot(
                    f"after_step:{optimizer_step}", optimizer_step=optimizer_step
                )
                latest_snapshot = current_snapshot
                if stop_on_failed_health_gate and not current_snapshot["paper_gate_passed"]:
                    health_gate_failed_step = optimizer_step
                    break
        if batches:
            epoch_record = {
                "epoch": epoch,
                **{key: value / batches for key, value in totals.items()},
                "batches": batches,
            }
            epoch_losses.append(epoch_record)
            selection = {
                "epoch": epoch,
                "train_total_loss": epoch_record["loss"],
                "health_snapshot": latest_snapshot,
            }
            if best_total_loss is None or epoch_record["loss"] < best_total_loss["loss"]:
                best_total_loss = {
                    "epoch": epoch,
                    "loss": epoch_record["loss"],
                    "optimizer_step": optimizer_step,
                }
                _atomic_torch_save(
                    save_root / "checkpoint_best_total_loss.pt",
                    _checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        ema=ema,
                        optimizer_step=optimizer_step,
                        config=config,
                        purpose="bounded_healthy_rqvae_best_train_total_loss",
                        selection=selection,
                    ),
                )
                _atomic_torch_save(
                    save_root / "model_best_total_loss_state_dict.pt", model.state_dict()
                )
            if checkpoint_every_epochs and epoch % checkpoint_every_epochs == 0:
                _atomic_torch_save(
                    save_root / f"checkpoint_epoch_{epoch:05d}.pt",
                    _checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        ema=ema,
                        optimizer_step=optimizer_step,
                        config=config,
                        purpose="bounded_healthy_rqvae_epoch_checkpoint",
                        selection=selection,
                    ),
                )
        if optimizer_step >= max_optimizer_steps or health_gate_failed_step is not None:
            break

    final_snapshot = take_snapshot("final", optimizer_step=optimizer_step)
    summary = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "study": "TIGER healthy-RQ-VAE bounded stability probe",
        "evidence_level": "healthy_local_improvement_candidate",
        "arm_name": arm_name,
        "optimizer_step": optimizer_step,
        "elapsed_seconds": time.perf_counter() - started,
        "losses_finite": losses_finite,
        "initial_snapshot": initial_snapshot,
        "final_snapshot": final_snapshot,
        "first_step": first_step,
        "epoch_losses": epoch_losses,
        "best_total_loss": best_total_loss,
        "dead_code_reset_totals": reset_totals,
        "ema_bootstrap_summary": ema_bootstrap_summary,
        "health_gate_failed_step": health_gate_failed_step,
        "paper_gate_passed": bool(
            losses_finite
            and health_gate_failed_step is None
            and final_snapshot["paper_gate_passed"]
        ),
        "next_action": "multi_epoch_candidate_allowed"
        if losses_finite
        and health_gate_failed_step is None
        and final_snapshot["paper_gate_passed"]
        else "do_not_start_long_rqvae_training",
    }
    _atomic_json_dump(save_root / "summary.json", summary)
    final_checkpoint = _checkpoint_payload(
        model=model,
        optimizer=optimizer,
        ema=ema,
        optimizer_step=optimizer_step,
        config=config,
        purpose="bounded_healthy_rqvae_candidate_resume",
        selection={"summary": summary},
    )
    _atomic_torch_save(save_root / "checkpoint_final.pt", final_checkpoint)
    _atomic_torch_save(save_root / "model_final_state_dict.pt", model.state_dict())
    summary["best_model_state_dict"] = "model_best_total_loss_state_dict.pt"
    summary["final_model_state_dict"] = "model_final_state_dict.pt"
    _atomic_json_dump(save_root / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


if __name__ == "__main__":
    raise SystemExit(
        "Run `python run_rqvae_healthy.py <gin-config>` so gin binds the "
        "imported train_rqvae_healthy.train function rather than a duplicate "
        "__main__ module."
    )
