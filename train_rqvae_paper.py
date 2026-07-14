from __future__ import annotations

import json
import math
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import gin
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset

from data.processed import ItemData
from data.processed import RecDataset
from data.utils import batch_to
from modules.quantize import QuantizeForwardMode
from modules.rqvae import RqVae
from modules.utils import parse_config
from paper_rqvae import build_epoch_dataloader
from paper_rqvae import build_paper_optimizer
from paper_rqvae import compute_corpus_diagnostics
from paper_rqvae import load_training_checkpoint
from paper_rqvae import publish_final_checkpoint
from paper_rqvae import save_training_checkpoint
from paper_rqvae import steps_per_epoch


_RUN_MATRIX: dict[str, tuple[int, int | None]] = {
    "smoke": (2, 2048),
    "bounded": (100, None),
    "full": (20000, None),
    "resume_probe": (3, 2048),
}


def validate_paper_run(
    *,
    run_mode: str,
    epochs: int,
    max_items: int | None,
    batch_size: int,
    learning_rate: float,
    num_workers: int,
    amp: bool,
    resume_checkpoint: str | None,
    vae_input_dim: int,
    vae_hidden_dims: list[int],
    vae_embed_dim: int,
    vae_codebook_size: int,
    vae_n_layers: int,
    commitment_weight: float,
) -> None:
    if run_mode not in _RUN_MATRIX:
        raise ValueError(f"unsupported paper-strict run mode: {run_mode}")
    expected_epochs, expected_max_items = _RUN_MATRIX[run_mode]
    if epochs != expected_epochs or max_items != expected_max_items:
        raise ValueError(
            f"{run_mode} requires epochs={expected_epochs} and "
            f"max_items={expected_max_items}"
        )
    if run_mode == "resume_probe" and not resume_checkpoint:
        raise ValueError("resume_probe requires a resume checkpoint")
    if batch_size != 1024:
        raise ValueError("paper-strict batch size must equal 1024")
    if learning_rate != 0.4:
        raise ValueError("paper-strict learning rate must equal 0.4")
    if num_workers != 0:
        raise ValueError("paper-strict num_workers must equal 0")
    if amp:
        raise ValueError("paper-strict amp must be disabled")
    if vae_input_dim != 768:
        raise ValueError("paper-strict input dimension must equal 768")
    if list(vae_hidden_dims) != [512, 256, 128]:
        raise ValueError("paper-strict hidden dimensions must equal [512, 256, 128]")
    if vae_embed_dim != 32:
        raise ValueError("paper-strict latent dimension must equal 32")
    if vae_codebook_size != 256:
        raise ValueError("paper-strict codebook size must equal 256")
    if vae_n_layers != 3:
        raise ValueError("paper-strict RQ-VAE must have 3 layers")
    if commitment_weight != 0.25:
        raise ValueError("paper-strict commitment weight must equal 0.25")


def build_paper_model(
    *,
    input_dim: int,
    hidden_dims: list[int],
    embed_dim: int,
    codebook_size: int,
    n_layers: int,
    commitment_weight: float,
) -> RqVae:
    validate_paper_run(
        run_mode="smoke",
        epochs=2,
        max_items=2048,
        batch_size=1024,
        learning_rate=0.4,
        num_workers=0,
        amp=False,
        resume_checkpoint=None,
        vae_input_dim=input_dim,
        vae_hidden_dims=hidden_dims,
        vae_embed_dim=embed_dim,
        vae_codebook_size=codebook_size,
        vae_n_layers=n_layers,
        commitment_weight=commitment_weight,
    )
    return RqVae(
        input_dim=input_dim,
        embed_dim=embed_dim,
        hidden_dims=hidden_dims,
        codebook_size=codebook_size,
        codebook_kmeans_init=True,
        codebook_normalize=False,
        codebook_sim_vq=False,
        codebook_mode=QuantizeForwardMode.STE,
        n_layers=n_layers,
        commitment_weight=commitment_weight,
        n_cat_features=0,
    )


def checkpoint_path_for(
    save_root: Path,
    *,
    completed_epoch: int,
    optimizer_step: int,
) -> Path:
    return save_root / (
        f"checkpoint_epoch_{completed_epoch:05d}_step_{optimizer_step:07d}.pt"
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _external_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parent,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _encode_corpus_three_token_ids(
    *,
    model: RqVae,
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    was_training = model.training
    model.eval()
    semantic_ids: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            device_batch = batch_to(batch, device)
            output = model.get_semantic_ids(device_batch.x, gumbel_t=0.2)
            semantic_ids.append(output.sem_ids.detach().cpu())
    model.train(was_training)
    return torch.cat(semantic_ids, dim=0)


def _device_payload(device: torch.device) -> dict[str, Any]:
    index = device.index if device.index is not None else torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(index)
    return {
        "type": device.type,
        "index": index,
        "name": torch.cuda.get_device_name(index),
        "capability": [major, minor],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def _evidence_level(run_mode: str, paper_gate_passed: bool) -> str:
    if run_mode in {"smoke", "resume_probe"}:
        return "diagnostic_smoke"
    if run_mode == "bounded":
        return "diagnostic_bounded"
    if paper_gate_passed:
        return "third_party_reproduction"
    return "diagnostic_full_failed_gate"


@gin.configurable
def train(
    run_mode: str = "smoke",
    epochs: int = 2,
    batch_size: int = 1024,
    learning_rate: float = 0.4,
    seed: int = 20260701,
    max_items: int | None = 2048,
    checkpoint_every_epochs: int = 2,
    diagnostics_every_epochs: int = 2,
    resume_checkpoint: str | None = None,
    dataset_folder: str = "dataset/amazon-p5-st5",
    dataset_split: str = "beauty",
    save_dir_root: str = "out/rqvae-paper-strict/",
    num_workers: int = 0,
    amp: bool = False,
    vae_input_dim: int = 768,
    vae_hidden_dims: list[int] = [512, 256, 128],
    vae_embed_dim: int = 32,
    vae_codebook_size: int = 256,
    vae_n_layers: int = 3,
    commitment_weight: float = 0.25,
) -> dict[str, Any]:
    validate_paper_run(
        run_mode=run_mode,
        epochs=epochs,
        max_items=max_items,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_workers=num_workers,
        amp=amp,
        resume_checkpoint=resume_checkpoint,
        vae_input_dim=vae_input_dim,
        vae_hidden_dims=vae_hidden_dims,
        vae_embed_dim=vae_embed_dim,
        vae_codebook_size=vae_codebook_size,
        vae_n_layers=vae_n_layers,
        commitment_weight=commitment_weight,
    )
    if dataset_split != "beauty":
        raise ValueError("paper-strict dataset split must equal beauty")
    if checkpoint_every_epochs <= 0 or diagnostics_every_epochs <= 0:
        raise ValueError("checkpoint and diagnostics cadence must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("paper-strict RQ-VAE training requires CUDA")

    save_root = Path(save_dir_root).expanduser().resolve()
    save_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.Generator().manual_seed(seed)

    canonical_dataset = ItemData(
        root=dataset_folder,
        dataset=RecDataset.AMAZON,
        force_process=False,
        train_test_split="all",
        split=dataset_split,
    )
    if len(canonical_dataset) != 12101:
        raise ValueError(
            "paper-strict Beauty corpus must contain exactly 12101 items; "
            f"found {len(canonical_dataset)}"
        )
    training_dataset: Dataset = canonical_dataset
    if max_items is not None:
        training_dataset = Subset(canonical_dataset, range(max_items))

    model = build_paper_model(
        input_dim=vae_input_dim,
        hidden_dims=vae_hidden_dims,
        embed_dim=vae_embed_dim,
        codebook_size=vae_codebook_size,
        n_layers=vae_n_layers,
        commitment_weight=commitment_weight,
    ).to(device)
    optimizer = build_paper_optimizer(model, learning_rate=learning_rate)

    config_payload: dict[str, Any] = {
        "run_mode": run_mode,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "max_items": max_items,
        "checkpoint_every_epochs": checkpoint_every_epochs,
        "diagnostics_every_epochs": diagnostics_every_epochs,
        "resume_checkpoint": resume_checkpoint,
        "dataset_folder": str(Path(dataset_folder).expanduser().resolve()),
        "dataset_split": dataset_split,
        "save_dir_root": str(save_root),
        "num_workers": num_workers,
        "amp": amp,
        "vae_input_dim": vae_input_dim,
        "vae_hidden_dims": list(vae_hidden_dims),
        "vae_embed_dim": vae_embed_dim,
        "vae_codebook_size": vae_codebook_size,
        "vae_n_layers": vae_n_layers,
        "commitment_weight": commitment_weight,
    }

    start_epoch = 1
    optimizer_step = 0
    if resume_checkpoint is not None:
        resume_path = Path(resume_checkpoint).expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        resume_state = load_training_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            generator=generator,
        )
        if resume_state["seed"] != seed:
            raise ValueError(
                f"resume seed {resume_state['seed']} does not match configured seed {seed}"
            )
        start_epoch = resume_state["start_epoch"]
        optimizer_step = resume_state["optimizer_step"]
        expected_resume_step = (start_epoch - 1) * steps_per_epoch(
            len(training_dataset), batch_size
        )
        if optimizer_step != expected_resume_step:
            raise ValueError(
                "resume optimizer step is incompatible with the configured corpus: "
                f"expected {expected_resume_step}, found {optimizer_step}"
            )
    if start_epoch > epochs:
        raise ValueError(
            f"resume starts at epoch {start_epoch}, beyond configured epoch {epochs}"
        )

    started_at = time.perf_counter()
    losses_finite = True
    last_losses: dict[str, float] = {}
    last_checkpoint: Path | None = None
    last_diagnostics: dict[str, Any] = {}

    for completed_epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_totals = {
            "loss": 0.0,
            "reconstruction_loss": 0.0,
            "rqvae_loss": 0.0,
        }
        batch_count = 0
        epoch_loader = build_epoch_dataloader(
            training_dataset,
            batch_size=batch_size,
            generator=generator,
        )
        for batch in epoch_loader:
            device_batch = batch_to(batch, device)
            optimizer.zero_grad(set_to_none=True)
            model_output = model(device_batch, gumbel_t=0.2)
            step_losses = {
                "loss": float(model_output.loss.detach().cpu().item()),
                "reconstruction_loss": float(
                    model_output.reconstruction_loss.detach().cpu().item()
                ),
                "rqvae_loss": float(model_output.rqvae_loss.detach().cpu().item()),
            }
            if not all(math.isfinite(value) for value in step_losses.values()):
                losses_finite = False
                raise FloatingPointError(
                    f"non-finite RQ-VAE loss at optimizer step {optimizer_step + 1}: "
                    f"{step_losses}"
                )
            model_output.loss.backward()
            optimizer.step()
            optimizer_step += 1
            batch_count += 1
            for key, value in step_losses.items():
                epoch_totals[key] += value

        last_losses = {
            key: value / batch_count for key, value in epoch_totals.items()
        }
        checkpoint_due = (
            completed_epoch % checkpoint_every_epochs == 0
            or completed_epoch == epochs
        )
        diagnostics_due = (
            completed_epoch % diagnostics_every_epochs == 0
            or completed_epoch == epochs
        )
        if checkpoint_due:
            last_checkpoint = checkpoint_path_for(
                save_root,
                completed_epoch=completed_epoch,
                optimizer_step=optimizer_step,
            )
            if last_checkpoint.exists() or last_checkpoint.is_symlink():
                raise FileExistsError(
                    f"refusing to overwrite immutable checkpoint {last_checkpoint}"
                )
            save_training_checkpoint(
                last_checkpoint,
                model=model,
                optimizer=optimizer,
                generator=generator,
                completed_epoch=completed_epoch,
                optimizer_step=optimizer_step,
                seed=seed,
                config=config_payload,
            )

        if diagnostics_due:
            three_token_ids = _encode_corpus_three_token_ids(
                model=model,
                dataset=canonical_dataset,
                device=device,
                batch_size=batch_size,
            )
            last_diagnostics = compute_corpus_diagnostics(
                three_token_ids,
                losses_finite=losses_finite,
            )
            last_diagnostics.update(
                {
                    "completed_epoch": completed_epoch,
                    "optimizer_step": optimizer_step,
                }
            )
            diagnostics_path = save_root / (
                f"diagnostics_epoch_{completed_epoch:05d}.json"
            )
            if diagnostics_path.exists() or diagnostics_path.is_symlink():
                raise FileExistsError(
                    f"refusing to overwrite diagnostics {diagnostics_path}"
                )
            _write_json(diagnostics_path, last_diagnostics)

        elapsed_seconds = time.perf_counter() - started_at
        progress_payload = {
            "completed_epoch": completed_epoch,
            "optimizer_step": optimizer_step,
            "losses": last_losses,
            "elapsed_seconds": elapsed_seconds,
        }
        print(json.dumps(progress_payload, sort_keys=True), flush=True)

        if checkpoint_due or diagnostics_due:
            summary = {
                "dataset": "amazon-beauty",
                "dataset_item_count": len(canonical_dataset),
                "training_item_count": len(training_dataset),
                "config": config_payload,
                "external_commit": _external_commit(),
                "completed_epoch": completed_epoch,
                "optimizer_step": optimizer_step,
                "losses": last_losses,
                "losses_finite": losses_finite,
                "elapsed_seconds": elapsed_seconds,
                "device": _device_payload(device),
                "checkpoint_path": (
                    str(last_checkpoint) if last_checkpoint is not None else None
                ),
                "diagnostics": last_diagnostics,
                "diagnostics_path": (
                    str(diagnostics_path) if diagnostics_due else None
                ),
                "evidence_level": _evidence_level(
                    run_mode,
                    bool(last_diagnostics.get("paper_gate_passed", False)),
                ),
            }
            _write_json(save_root / "summary.json", summary)

    if last_checkpoint is None:
        raise RuntimeError("training completed without writing a checkpoint")
    if run_mode == "full" and epochs == 20000:
        publish_final_checkpoint(
            last_checkpoint,
            save_root / "paper_strict_rqvae_final.pt",
            diagnostics=last_diagnostics,
        )

    return summary


if __name__ == "__main__":
    parse_config()
    train()
