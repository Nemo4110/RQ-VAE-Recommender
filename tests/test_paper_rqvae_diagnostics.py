from pathlib import Path

import pytest
import torch

from paper_rqvae import compute_corpus_diagnostics
from paper_rqvae import publish_final_checkpoint


def test_diagnostics_require_fixed_256_and_usage_gates() -> None:
    three = torch.tensor([[0, 0, 0], [0, 0, 0], [1, 1, 1]])

    diagnostics = compute_corpus_diagnostics(three)

    assert diagnostics["fixed_collision_cardinality"] == 256
    assert diagnostics["losses_finite"] is True
    assert diagnostics["max_bucket_size"] == 2
    assert diagnostics["unique_four_token_count"] == 3
    assert diagnostics["collision_gate_passed"] is True
    assert diagnostics["usage_gate_passed"] is False
    assert diagnostics["paper_gate_passed"] is False


def test_diagnostics_fail_when_losses_are_non_finite() -> None:
    three = torch.tensor(
        [[index, index, index] for index in range(205)],
        dtype=torch.long,
    )

    diagnostics = compute_corpus_diagnostics(three, losses_finite=False)

    assert diagnostics["collision_gate_passed"] is True
    assert diagnostics["usage_gate_passed"] is True
    assert diagnostics["losses_finite"] is False
    assert diagnostics["paper_gate_passed"] is False


def test_diagnostics_pass_with_205_used_entries_per_codebook() -> None:
    three = torch.tensor(
        [[index, index, index] for index in range(205)],
        dtype=torch.long,
    )

    diagnostics = compute_corpus_diagnostics(three)

    assert diagnostics["codebook_usage"] == pytest.approx([205 / 256] * 3)
    assert diagnostics["max_bucket_size"] == 1
    assert diagnostics["unique_four_token_count"] == 205
    assert diagnostics["paper_gate_passed"] is True


def test_final_checkpoint_is_published_only_after_gate(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint_epoch_20000.pt"
    checkpoint.write_bytes(b"checkpoint")
    link = tmp_path / "paper_strict_rqvae_final.pt"

    with pytest.raises(ValueError, match="paper gate"):
        publish_final_checkpoint(
            checkpoint,
            link,
            diagnostics={"paper_gate_passed": False},
        )

    publish_final_checkpoint(
        checkpoint,
        link,
        diagnostics={"paper_gate_passed": True},
    )
    assert link.is_symlink()
    assert link.resolve() == checkpoint.resolve()
