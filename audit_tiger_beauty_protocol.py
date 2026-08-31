#!/usr/bin/env python
"""Emit a reproducible, read-only Beauty data/protocol audit for TIGER."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data.amazon import AmazonReviews


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_tensor(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return _sha256_bytes(array.tobytes())


def _sha256_texts(values: Sequence[Any]) -> str:
    normalized = [str(value) for value in values]
    payload = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    return _sha256_bytes(payload.encode("utf-8"))


def _valid_ids(values: Sequence[int]) -> list[int]:
    return [int(value) for value in values if int(value) >= 0]


def validation_target_probability(train_history_length: int, max_seq_len: int = 20) -> float:
    """Exact probability that current random-window sampling selects validation."""

    if train_history_length < 3:
        raise ValueError("train history must contain at least three items")
    if max_seq_len < 3:
        raise ValueError("max_seq_len must be at least three")

    # SeqData appends validation to a train history of length L, chooses a start
    # uniformly from [0, L - 2], then end uniformly from [start + 3,
    # start + max_seq_len + 1].  The validation item at index L is the target
    # exactly when the exclusive slice endpoint is at least L + 1.
    probabilities = []
    for start in range(train_history_length - 1):
        lower = start + 3
        upper = start + max_seq_len + 1
        choice_count = upper - lower + 1
        favorable = max(0, upper - max(lower, train_history_length + 1) + 1)
        probabilities.append(favorable / choice_count)
    return sum(probabilities) / len(probabilities)


def _summary(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "minimum": 0, "maximum": 0, "mean": 0.0, "median": 0.0}
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "count": int(tensor.numel()),
        "minimum": int(tensor.min().item()),
        "maximum": int(tensor.max().item()),
        "mean": float(tensor.mean().item()),
        "median": float(tensor.median().item()),
    }


def _git_head() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def audit(*, dataset_root: Path, split: str) -> dict[str, Any]:
    dataset = AmazonReviews(root=str(dataset_root), split=split)
    data = dataset.data
    item = data["item"]
    history = data[("user", "rated", "item")]["history"]
    train = history["train"]
    test = history["test"]

    train_histories = [_valid_ids(row) for row in train["itemId"]]
    test_histories = [_valid_ids(row) for row in test["itemId"]]
    validation_targets = [int(row.item()) for row in train["itemId_fut"]]
    test_targets = [int(row.item()) for row in test["itemId_fut"]]
    train_user_ids = train["userId"].reshape(-1).cpu()
    test_user_ids = test["userId"].reshape(-1).cpu()

    expected_test_history_matches = 0
    for train_history, validation_target, test_history in zip(
        train_histories, validation_targets, test_histories, strict=True
    ):
        expected = (train_history + [validation_target])[-20:]
        if test_history == expected:
            expected_test_history_matches += 1

    train_catalog = set(item_id for row in train_histories for item_id in row)
    train_catalog.update(validation_targets)
    test_target_catalog = set(test_targets)
    all_item_ids = set(range(int(item["x"].shape[0])))
    raw_split_dir = Path(dataset.raw_dir) / split
    raw_file_hashes = {
        filename: _sha256_file(raw_split_dir / filename)
        for filename in ("sequential_data.txt", "datamaps.json", "meta.json.gz")
        if (raw_split_dir / filename).is_file()
    }
    probabilities = [validation_target_probability(len(row)) for row in train_histories]

    return {
        "schema_version": 1,
        "scope": "TIGER Beauty local processed-data and sampler audit",
        "git_head": _git_head(),
        "dataset_root": str(dataset_root),
        "split": split,
        "processed_path": str(dataset.processed_paths[0]),
        "raw_file_sha256": raw_file_hashes,
        "item_universe": {
            "item_count": int(item["x"].shape[0]),
            "feature_shape": list(item["x"].shape),
            "feature_sha256": _sha256_tensor(item["x"]),
            "text_count": int(len(item["text"])),
            "text_sha256": _sha256_texts(item["text"]),
            "random_item_is_train_count": int(item["is_train"].sum().item()),
            "random_item_is_eval_count": int((~item["is_train"]).sum().item()),
            "interaction_train_or_validation_catalog_count": len(train_catalog),
            "test_target_catalog_count": len(test_target_catalog),
            "test_targets_outside_train_or_validation_catalog_count": len(
                test_target_catalog - train_catalog
            ),
            "unreferenced_item_count": len(
                all_item_ids - train_catalog - test_target_catalog
            ),
        },
        "content_embedding_provenance": {
            "implementation_model_id": "sentence-transformers/sentence-t5-xxl",
            "implementation_serialization": (
                "Title: {title}; Brand: {brand}; Categories: {categories[0]}; Price: {price}; "
            ),
            "unknown_from_processed_artifact": [
                "exact model revision",
                "SentenceTransformer pooling and normalization defaults at preprocessing time",
                "whether this serialization is text-identical to the TIGER authors' pipeline",
            ],
        },
        "users_and_splits": {
            "train_user_count": int(train_user_ids.numel()),
            "test_user_count": int(test_user_ids.numel()),
            "same_user_order": bool(torch.equal(train_user_ids, test_user_ids)),
            "train_user_id_sha256": _sha256_tensor(train_user_ids),
            "test_user_id_sha256": _sha256_tensor(test_user_ids),
            "train_history_lengths": _summary([len(row) for row in train_histories]),
            "test_history_lengths_after_cap": _summary([len(row) for row in test_histories]),
            "test_history_equals_last20_train_plus_validation_count": expected_test_history_matches,
            "test_history_equals_last20_train_plus_validation_rate": (
                expected_test_history_matches / len(train_histories) if train_histories else 0.0
            ),
        },
        "current_train_subsample_policy": {
            "implementation": "train history + validation item, then random prefix-to-next-item window",
            "validation_is_appended_before_window_sampling": True,
            "expected_validation_target_probability": {
                "mean": float(sum(probabilities) / len(probabilities)) if probabilities else 0.0,
                "minimum": float(min(probabilities)) if probabilities else 0.0,
                "maximum": float(max(probabilities)) if probabilities else 0.0,
            },
            "paper_protocol_status": (
                "The paper names the penultimate item validation but does not state whether "
                "the decoder retrains on validation before test; this local sampler therefore "
                "requires explicit interpretation and must not be silently called exact."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/amazon-p5"))
    parser.add_argument("--split", default="beauty")
    parser.add_argument(
        "--output", type=Path, default=Path("out/audits/tiger_beauty_protocol_audit.json")
    )
    args = parser.parse_args()
    result = audit(dataset_root=args.dataset_root, split=args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
