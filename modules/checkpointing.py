from __future__ import annotations


def resolve_checkpoint_steps(
    iterations: int, save_model_every: int, save_model_at: list[int] | None = None
) -> frozenset[int]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if save_model_every <= 0:
        raise ValueError("save_model_every must be positive")

    explicit_steps = set(save_model_at or [])
    invalid_steps = sorted(
        step for step in explicit_steps if step <= 0 or step > iterations
    )
    if invalid_steps:
        raise ValueError(
            "save_model_at steps must be in [1, iterations]; "
            f"invalid steps: {invalid_steps}"
        )

    periodic_steps = set(range(save_model_every, iterations + 1, save_model_every))
    return frozenset(periodic_steps | explicit_steps | {iterations})
