from __future__ import annotations

import pytest

from modules.checkpointing import resolve_checkpoint_steps


def test_checkpoint_schedule_combines_periodic_explicit_and_final_steps() -> None:
    assert resolve_checkpoint_steps(
        iterations=20,
        save_model_every=7,
        save_model_at=[2, 7, 17],
    ) == frozenset({2, 7, 14, 17, 20})


def test_checkpoint_schedule_rejects_out_of_range_explicit_steps() -> None:
    with pytest.raises(ValueError, match=r"invalid steps: \[0, 11\]"):
        resolve_checkpoint_steps(
            iterations=10,
            save_model_every=100,
            save_model_at=[0, 11],
        )
