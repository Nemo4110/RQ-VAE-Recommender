"""RQ-VAE collapse fix-matrix trainer (bounded diagnostic arms E1-E5).

Reuses the shared common initialization and deterministic epoch plan from the
2026-07-14 collapse diagnosis, and adds configurable anti-collapse treatments:

- dead-code reset (donor = current batch residual points)
- EMA codebook updates (replacing gradient codebook updates)
- encoder/decoder re-initialization with kaiming_normal_(relu) gain
- codebook L2 normalization (SimVQ-style anchored geometry)
- linear LR warmup

Training forward is eager: compiled and eager gradients were verified identical
on 2026-07-28 (cos >= 0.998 for codebook, exact for encoder/decoder), and the
eager path exposes per-layer residuals/ids needed by EMA and reset.

Evidence level: diagnostic only. Not a paper reproduction.
"""

from __future__ import annotations

import json
import math
import random
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
from paper_rqvae import compute_corpus_diagnostics
from rqvae_collapse_diagnostics import epoch_permutation

COMMON_INIT = "/root/autodl-tmp/recsys-roi-study/runs/tiger/collapse-diagnosis-beauty-20260714/common/common_initialization.pt"

_SNAPSHOT_STEPS = (1, 2, 3, 12)
_SNAPSHOT_EPOCHS = (1, 2, 5, 10, 25, 50, 100, 250, 500)


def build_fix_model(
    *,
    encoder_init_scheme: str,
    codebook_normalize: bool,
    seed: int,
    device: torch.device,
    init_batch_x: torch.Tensor | None,
) -> RqVae:
    """Build the paper-shape RQ-VAE.

    encoder_init_scheme="common" loads the shared post-k-means state.
    encoder_init_scheme="kaiming_relu" re-initializes encoder/decoder Linear
    weights with kaiming_normal_(nonlinearity="relu") and re-runs k-means on
    the provided init batch.
    """
    model = RqVae(
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
    if encoder_init_scheme == "common":
        artifact = torch.load(COMMON_INIT, map_location="cpu", weights_only=False)
        model.load_state_dict(artifact["post_kmeans_model_state"])
        for layer in model.layers:
            layer.kmeans_initted = True
    elif encoder_init_scheme == "kaiming_relu":
        if init_batch_x is None:
            raise ValueError("kaiming_relu init requires the k-means init batch")
        generator = torch.Generator().manual_seed(seed)
        for module in list(model.encoder.modules()) + list(model.decoder.modules()):
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(
                    module.weight, nonlinearity="relu",
                )
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            model.get_semantic_ids(init_batch_x.to(device), gumbel_t=0.2)
        assert all(layer.kmeans_initted for layer in model.layers)
    else:
        raise ValueError(f"unsupported init scheme: {encoder_init_scheme}")
    return model.to(device)


class EmaCodebook:
    """EMA codebook state; replaces gradient updates for embedding weights."""

    def __init__(self, model: RqVae, decay: float) -> None:
        self.decay = decay
        self.cluster_size = []
        self.embed_avg = []
        for layer in model.layers:
            weight = layer.embedding.weight
            self.cluster_size.append(torch.zeros(weight.shape[0], device=weight.device))
            self.embed_avg.append(weight.data.clone())

    @torch.no_grad()
    def update(self, model: RqVae, residuals: list[torch.Tensor], sem_ids: list[torch.Tensor]) -> None:
        for idx, (layer, res, ids) in enumerate(zip(model.layers, residuals, sem_ids)):
            encodings = torch.nn.functional.one_hot(ids, layer.n_embed).float()
            counts = encodings.sum(0)
            embed_sum = encodings.T @ res
            self.cluster_size[idx].mul_(self.decay).add_(counts, alpha=1 - self.decay)
            self.embed_avg[idx].mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size[idx].sum()
            smoothed = (self.cluster_size[idx] + 1e-5) / (n + layer.n_embed * 1e-5) * n
            layer.embedding.weight.data = self.embed_avg[idx] / smoothed.unsqueeze(1)


@torch.no_grad()
def dead_code_reset(
    model: RqVae,
    residuals: list[torch.Tensor],
    sem_ids: list[torch.Tensor],
) -> list[int]:
    reset_counts = []
    for layer, res, ids in zip(model.layers, residuals, sem_ids):
        used = torch.zeros(layer.n_embed, dtype=torch.bool, device=res.device)
        used[ids.unique()] = True
        dead = (~used).nonzero(as_tuple=True)[0]
        if len(dead) == 0:
            reset_counts.append(0)
            continue
        donors = res[torch.randperm(len(res), device=res.device)[: len(dead)]]
        layer.embedding.weight.data[dead] = donors
        reset_counts.append(int(len(dead)))
    return reset_counts


def eager_forward(model: RqVae, x: torch.Tensor):
    """RqVae.forward math without torch.compile (n_cat_feats=0 path)."""
    quantized = model.get_semantic_ids(x, gumbel_t=0.2)
    x_hat = model.decode(quantized.embeddings.sum(axis=-1))
    recon = ((x_hat - x) ** 2).sum(axis=-1)
    loss = (recon + quantized.quantize_loss).mean()
    residuals = list(quantized.residuals.unbind(dim=-1))
    sem_ids = list(quantized.sem_ids.unbind(dim=-1))
    return loss, recon.mean(), quantized.quantize_loss.mean(), residuals, sem_ids


@torch.no_grad()
def corpus_snapshot(model: RqVae, dataset: Dataset, device: torch.device) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0)
    ids_all, norms = [], []
    for batch in loader:
        x = batch_to(batch, device).x
        norms.append(model.encode(x).norm(dim=1).cpu())
        ids_all.append(model.get_semantic_ids(x, gumbel_t=0.2).sem_ids.cpu())
    model.train(was_training)
    sem_ids = torch.cat(ids_all)
    n = torch.cat(norms)
    gates = compute_corpus_diagnostics(sem_ids)
    return {
        "used_counts": [int(sem_ids[:, i].unique().numel()) for i in range(3)],
        "unique_three_token_count": gates["unique_three_token_count"],
        "unique_four_token_count": gates["unique_four_token_count"],
        "max_bucket_size": gates["max_bucket_size"],
        "codebook_usage": gates["codebook_usage"],
        "collision_gate_passed": gates["collision_gate_passed"],
        "usage_gate_passed": gates["usage_gate_passed"],
        "paper_gate_passed": gates["paper_gate_passed"],
        "encoder_norm_p50": float(n.median()),
        "hard_collapse": bool(gates["max_bucket_size"] > 256 * 10),
    }


@gin.configurable
def train(
    arm_name: str = "E1_reset50",
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    warmup_steps: int = 0,
    dead_code_reset_every: int = 0,
    ema_codebook: bool = False,
    ema_decay: float = 0.99,
    encoder_init_scheme: str = "common",
    vae_codebook_normalize: bool = False,
    epochs: int = 500,
    batch_size: int = 1024,
    seed: int = 20260701,
    dataset_folder: str = "dataset/amazon-p5-st5",
    dataset_split: str = "beauty",
    save_dir_root: str = "out/rqvae-fix-matrix/",
) -> dict[str, Any]:
    if batch_size != 1024:
        raise ValueError("fix matrix keeps the diagnostic batch size 1024")
    save_root = Path(save_dir_root).expanduser().resolve() / arm_name
    save_root.mkdir(parents=True, exist_ok=True)
    snapshot_path = save_root / "snapshots.jsonl"
    if snapshot_path.exists():
        raise FileExistsError(f"refusing to overwrite {snapshot_path}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda")
    dataset = ItemData(
        root=dataset_folder,
        dataset=RecDataset.AMAZON,
        force_process=False,
        train_test_split="all",
        split=dataset_split,
    )
    if len(dataset) != 12101:
        raise ValueError(f"expected 12101 items, found {len(dataset)}")

    init_idx = epoch_permutation(len(dataset), seed, 1)[:batch_size].tolist()
    init_batch_x = dataset[init_idx].x

    model = build_fix_model(
        encoder_init_scheme=encoder_init_scheme,
        codebook_normalize=vae_codebook_normalize,
        seed=seed,
        device=device,
        init_batch_x=init_batch_x,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.999),
        eps=1e-8, weight_decay=weight_decay,
    )
    ema = EmaCodebook(model, ema_decay) if ema_codebook else None

    config_payload = {
        "arm_name": arm_name,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
        "dead_code_reset_every": dead_code_reset_every,
        "ema_codebook": ema_codebook,
        "ema_decay": ema_decay,
        "encoder_init_scheme": encoder_init_scheme,
        "vae_codebook_normalize": vae_codebook_normalize,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "dataset_folder": dataset_folder,
        "dataset_split": dataset_split,
    }

    started = time.perf_counter()
    optimizer_step = 0

    def append_snapshot(trigger: str, completed_epoch: int | None) -> dict[str, Any]:
        snap = corpus_snapshot(model, dataset, device)
        snap["optimizer_step"] = optimizer_step
        snap["completed_epoch"] = completed_epoch
        snap["trigger"] = trigger
        with snapshot_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(snap, sort_keys=True) + "\n")
        return snap

    first_snapshot = append_snapshot("optimizer_step:0", None)
    last_snapshot = first_snapshot
    ever_hard_collapse = bool(first_snapshot["hard_collapse"])
    epoch_losses: dict[str, float] = {}

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = epoch_permutation(len(dataset), seed, epoch)
        loader = DataLoader(
            Subset(dataset, permutation.tolist()),
            batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0,
        )
        totals = {"loss": 0.0, "reconstruction_loss": 0.0, "rqvae_loss": 0.0}
        batches = 0
        for batch in loader:
            if warmup_steps > 0:
                cur_lr = learning_rate * min(1.0, (optimizer_step + 1) / warmup_steps)
                for group in optimizer.param_groups:
                    group["lr"] = cur_lr
            device_batch = batch_to(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, recon, qloss, residuals, sem_ids = eager_forward(model, device_batch.x)
            loss.backward()
            if ema is not None:
                for layer in model.layers:
                    layer.embedding.weight.grad = None
            optimizer.step()
            if ema is not None:
                ema.update(model, residuals, sem_ids)
            if dead_code_reset_every > 0 and (optimizer_step + 1) % dead_code_reset_every == 0:
                dead_code_reset(model, residuals, sem_ids)
            optimizer_step += 1
            batches += 1
            totals["loss"] += float(loss.item())
            totals["reconstruction_loss"] += float(recon.item())
            totals["rqvae_loss"] += float(qloss.item())
            if optimizer_step in _SNAPSHOT_STEPS:
                last_snapshot = append_snapshot(f"optimizer_step:{optimizer_step}", None)
                ever_hard_collapse |= bool(last_snapshot["hard_collapse"])
        epoch_losses = {key: value / batches for key, value in totals.items()}
        if epoch in _SNAPSHOT_EPOCHS or epoch % 500 == 0 or epoch == epochs:
            last_snapshot = append_snapshot(f"epoch:{epoch}", epoch)
            ever_hard_collapse |= bool(last_snapshot["hard_collapse"])
        if epoch % 2000 == 0 or epoch == epochs:
            torch.save(
                {"model": model.state_dict(), "completed_epoch": epoch,
                 "optimizer_step": optimizer_step, "config": config_payload},
                save_root / f"checkpoint_epoch_{epoch:05d}_step_{optimizer_step:07d}.pt",
            )
        if epoch % 50 == 0:
            print(json.dumps({"arm": arm_name, "epoch": epoch,
                              "losses": epoch_losses,
                              "used": last_snapshot["used_counts"],
                              "triples": last_snapshot["unique_three_token_count"]}),
                  flush=True)

    summary = {
        "study": "RQ-VAE collapse fix matrix (bounded diagnostic)",
        "evidence_level": "diagnostic_bounded",
        "arm": arm_name,
        "config": config_payload,
        "optimizer_step": optimizer_step,
        "final_losses": epoch_losses,
        "final_assignment": last_snapshot,
        "ever_hard_collapse": ever_hard_collapse,
        "paper_gate_passed": bool(last_snapshot["paper_gate_passed"]),
        "elapsed_seconds": time.perf_counter() - started,
        "snapshot_jsonl_path": str(snapshot_path),
        "common_initialization": (
            COMMON_INIT if encoder_init_scheme == "common" else None
        ),
    }
    (save_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps({"arm": arm_name, "done": True,
                      "paper_gate_passed": summary["paper_gate_passed"],
                      "ever_hard_collapse": ever_hard_collapse,
                      "final_used": last_snapshot["used_counts"],
                      "final_triples": last_snapshot["unique_three_token_count"],
                      "final_max_bucket": last_snapshot["max_bucket_size"]}),
          flush=True)
    return summary


if __name__ == "__main__":
    parse_config()
    train()
